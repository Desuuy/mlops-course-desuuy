import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("housing_eval_mlflow")

# Set MLflow tracking URI (should match training script)
mlflow.set_tracking_uri("file:./mlruns")


def evaluate_model(model_name="housing_price_predictor", version="latest"):
    """
    Evaluate a model from MLflow Model Registry

    Args:
        model_name: Name of the registered model
        version: Version of the model to load (default: "latest")
    """
    # Paths
    PROJECT_ROOT = Path(os.getcwd())
    DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"

    logger.info(f"Loading test data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Prepare features and target (same as training)
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

    logger.info(f"Loading model '{model_name}' version '{version}' from MLflow...")

    try:
        # Load model from MLflow Model Registry
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model from: {model_uri}")

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = mse**0.5

        # Log metrics
        logger.info("Model evaluation results:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")

        # Log to MLflow as a new run
        with mlflow.start_run(run_name=f"evaluation_{model_name}_{version}") as run:
            mlflow.log_metrics(
                {
                    "eval_mse": mse,
                    "eval_mae": mae,
                    "eval_r2": r2,
                    "eval_rmse": rmse,
                }
            )

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "model_version": version,
                    "dataset_size": len(df),
                    "num_features": len(NUM_FEATURES),
                }
            )

            logger.info(f"Evaluation logged to MLflow run: {run.info.run_id}")

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "run_id": run.info.run_id,
        }

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Make sure the model is registered in MLflow Model Registry")
        logger.error("Run the training script first to register the model")
        return None


def list_registered_models():
    """List all registered models in MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        logger.info("Registered models in MLflow:")
        for model in models:
            logger.info(f"  Model: {model.name}")
            for version in model.latest_versions:
                logger.info(
                    f"    Version: {version.version} (Stage: {version.current_stage})"
                )

    except Exception as e:
        logger.error(f"Error listing models: {e}")


if __name__ == "__main__":
    # List available models first
    list_registered_models()

    # Evaluate the latest version of the housing price predictor
    results = evaluate_model()

    if results:
        logger.info("Evaluation completed successfully!")
        logger.info(
            f"Model performance - R²: {results['r2']:.4f}, RMSE: {results['rmse']:.4f}"
        )
    else:
        logger.error("Evaluation failed!")
