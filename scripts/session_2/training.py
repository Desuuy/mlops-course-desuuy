import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("housing_mlflow")


class MLflowSGDRegressor(SGDRegressor):
    """Custom SGDRegressor that logs metrics for each epoch to MLflow"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch_metrics = []

    def fit(self, X, y, **kwargs):
        """Override fit to log metrics for each epoch"""
        # Store original max_iter
        original_max_iter = self.max_iter

        # Fit with epoch-by-epoch tracking
        self._fit_with_epoch_logging(X, y, **kwargs)

        return self

    def _fit_with_epoch_logging(self, X, y, **kwargs):
        """Fit model with epoch-by-epoch metric logging"""
        # Initialize model parameters
        n_samples, n_features = X.shape

        # Initialize weights and bias
        if not hasattr(self, "coef_") or self.coef_ is None:
            self.coef_ = np.zeros(n_features)
        if not hasattr(self, "intercept_") or self.intercept_ is None:
            self.intercept_ = np.array([0.0])

        # Training loop with epoch logging
        for epoch in range(self.max_iter):
            # Make predictions for current epoch
            y_pred = X @ self.coef_ + self.intercept_

            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = mse**0.5

            # Store metrics for this epoch
            epoch_metric = {
                "epoch": epoch + 1,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "rmse": rmse,
            }
            self.epoch_metrics.append(epoch_metric)

            # Log to MLflow every 10 epochs or on the last epoch
            if (epoch + 1) % 10 == 0 or epoch == self.max_iter - 1:
                mlflow.log_metrics(
                    {
                        f"epoch_{epoch+1}_mse": mse,
                        f"epoch_{epoch+1}_mae": mae,
                        f"epoch_{epoch+1}_r2": r2,
                        f"epoch_{epoch+1}_rmse": rmse,
                    },
                    step=epoch + 1,
                )

                logger.info(
                    f"Epoch {epoch+1}/{self.max_iter}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}"
                )

            # Check for early stopping
            if epoch > 0:
                prev_mse = self.epoch_metrics[-2]["mse"]
                if abs(mse - prev_mse) < self.tol:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} (convergence reached)"
                    )
                    break

        # Log final epoch metrics
        final_metrics = self.epoch_metrics[-1]
        mlflow.log_metrics(
            {
                "final_epoch": final_metrics["epoch"],
                "final_mse": final_metrics["mse"],
                "final_mae": final_metrics["mae"],
                "final_r2": final_metrics["r2"],
                "final_rmse": final_metrics["rmse"],
            }
        )

        logger.info(f"Training completed after {final_metrics['epoch']} epochs")
        logger.info(
            f"Final metrics - MSE: {final_metrics['mse']:.4f}, R²: {final_metrics['r2']:.4f}"
        )

        return self


def create_training_curves(epoch_metrics, save_path):
    """Create and save training curves plots"""
    if not epoch_metrics:
        return None

    epochs = [m["epoch"] for m in epoch_metrics]
    mse_values = [m["mse"] for m in epoch_metrics]
    mae_values = [m["mae"] for m in epoch_metrics]
    r2_values = [m["r2"] for m in epoch_metrics]
    rmse_values = [m["rmse"] for m in epoch_metrics]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Metrics Over Epochs", fontsize=16)

    # MSE plot
    axes[0, 0].plot(epochs, mse_values, "b-", linewidth=2)
    axes[0, 0].set_title("Mean Squared Error")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, alpha=0.3)

    # MAE plot
    axes[0, 1].plot(epochs, mae_values, "r-", linewidth=2)
    axes[0, 1].set_title("Mean Absolute Error")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].grid(True, alpha=0.3)

    # R² plot
    axes[1, 0].plot(epochs, r2_values, "g-", linewidth=2)
    axes[1, 0].set_title("R² Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("R²")
    axes[1, 0].grid(True, alpha=0.3)

    # RMSE plot
    axes[1, 1].plot(epochs, rmse_values, "m-", linewidth=2)
    axes[1, 1].set_title("Root Mean Squared Error")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Training curves saved to: {save_path}")
    return save_path


# Set MLflow tracking URI (you can change this to a remote server)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
EXPERIMENT_NAME = "housing_price_prediction"
mlflow.set_experiment(EXPERIMENT_NAME)


def train():
    # Paths
    PROJECT_ROOT = Path(os.getcwd())
    DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
    ARTIFACT_DIR = PROJECT_ROOT / "scripts" / "session_2"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = ARTIFACT_DIR / "housing_linear_mlflow.joblib"

    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    logger.info("Preparing features and target...")
    # Identify target and basic features from the CSV header
    TARGET = "Price"
    NUM_FEATURES = [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population",
    ]

    X = df[NUM_FEATURES]
    y = df[TARGET]

    logger.info("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model parameters
    model_params = {
        "max_iter": 5000,
        "tol": 1e-4,
        "learning_rate": "optimal",
        "random_state": 42,
    }

    logger.info("Building pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", MLflowSGDRegressor(**model_params)),
        ]
    )

    # Start MLflow run
    with mlflow.start_run(run_name="housing_linear_regression") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", NUM_FEATURES)
        mlflow.log_param("target", TARGET)

        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("num_features", len(NUM_FEATURES))

        logger.info("Training model...")
        model.fit(X_train, y_train)

        # Evaluate model performance
        logger.info("Evaluating model performance...")

        # Make predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for training set
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = train_mse**0.5

        # Calculate metrics for test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = test_mse**0.5

        # Log training metrics
        logger.info("Training set metrics:")
        logger.info(f"  MSE: {train_mse:.4f}")
        logger.info(f"  MAE: {train_mae:.4f}")
        logger.info(f"  R²: {train_r2:.4f}")
        logger.info(f"  RMSE: {train_rmse:.4f}")

        # Log test metrics
        logger.info("Test set metrics:")
        logger.info(f"  MSE: {test_mse:.4f}")
        logger.info(f"  MAE: {test_mae:.4f}")
        logger.info(f"  R²: {test_r2:.4f}")
        logger.info(f"  RMSE: {test_rmse:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_mse": train_mse,
                "train_mae": train_mae,
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
            }
        )

        # Log model performance summary
        logger.info("Model performance summary:")
        logger.info(f"  Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"  Training RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # Create and log training curves
        regressor = model.named_steps["regressor"]
        if hasattr(regressor, "epoch_metrics") and regressor.epoch_metrics:
            training_curves_path = ARTIFACT_DIR / "training_curves.png"
            create_training_curves(regressor.epoch_metrics, training_curves_path)
            mlflow.log_artifact(str(training_curves_path), "plots")
            logger.info("Training curves logged to MLflow")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="housing_price_predictor",
        )

        # Log additional artifacts
        mlflow.log_artifact(str(DATA_PATH), "data")

        # Save model locally as well
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to: {MODEL_PATH}")

        # Log model path
        mlflow.log_artifact(str(MODEL_PATH), "artifacts")

        logger.info("MLflow run completed successfully!")
        logger.info("View the run in MLflow UI: mlflow ui")
        logger.info(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train()
