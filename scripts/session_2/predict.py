import logging

import mlflow
import mlflow.sklearn
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("housing_predict_mlflow")

# Set MLflow tracking URI (should match training script)
mlflow.set_tracking_uri("file:./mlruns")


def load_model_from_mlflow(model_name="housing_price_predictor", version="latest"):
    """
    Load a model from MLflow Model Registry

    Args:
        model_name: Name of the registered model
        version: Version of the model to load (default: "latest")

    Returns:
        Loaded model object
    """
    logger.info(f"Loading model '{model_name}' version '{version}' from MLflow...")

    try:
        # Load model from MLflow Model Registry
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model from: {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Make sure the model is registered in MLflow Model Registry")
        logger.error("Run the training script first to register the model")
        return None


def create_sample_data():
    """
    Create 2 sample data points for prediction

    Returns:
        DataFrame with sample data
    """
    # Sample data points with realistic values for housing features
    sample_data = {
        "Avg. Area Income": [80000, 120000],
        "Avg. Area House Age": [5, 12],
        "Avg. Area Number of Rooms": [6, 8],
        "Avg. Area Number of Bedrooms": [3, 4],
        "Area Population": [50000, 75000],
    }

    df = pd.DataFrame(sample_data)
    logger.info("Created sample data:")
    logger.info(
        f"Sample 1: Income=${df.iloc[0]['Avg. Area Income']:,.0f}, "
        f"Age={df.iloc[0]['Avg. Area House Age']} years, "
        f"Rooms={df.iloc[0]['Avg. Area Number of Rooms']}, "
        f"Bedrooms={df.iloc[0]['Avg. Area Number of Bedrooms']}, "
        f"Population={df.iloc[0]['Area Population']:,.0f}"
    )

    logger.info(
        f"Sample 2: Income=${df.iloc[1]['Avg. Area Income']:,.0f}, "
        f"Age={df.iloc[1]['Avg. Area House Age']} years, "
        f"Rooms={df.iloc[1]['Avg. Area Number of Rooms']}, "
        f"Bedrooms={df.iloc[1]['Avg. Area Number of Bedrooms']}, "
        f"Population={df.iloc[1]['Area Population']:,.0f}"
    )

    return df


def make_predictions(model, sample_data):
    """
    Make predictions using the loaded model

    Args:
        model: Loaded MLflow model
        sample_data: DataFrame with sample data

    Returns:
        Array of predictions
    """
    logger.info("Making predictions on sample data...")

    try:
        # Make predictions
        predictions = model.predict(sample_data)

        # Log predictions
        logger.info("Prediction results:")
        for i, pred in enumerate(predictions):
            logger.info(f"  Sample {i+1}: Predicted Price = ${pred:,.2f}")

        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None


def predict_housing_prices(model_name="housing_price_predictor", version="latest"):
    """
    Main function to load model and make predictions on sample data

    Args:
        model_name: Name of the registered model
        version: Version of the model to load
    """
    logger.info("Starting housing price prediction...")

    # Load model from MLflow
    model = load_model_from_mlflow(model_name, version)
    if model is None:
        return None

    # Create sample data
    sample_data = create_sample_data()

    # Make predictions
    predictions = make_predictions(model, sample_data)
    if predictions is None:
        return None

    # Create results summary
    results = {
        "model_name": model_name,
        "model_version": version,
        "sample_data": sample_data,
        "predictions": predictions,
    }

    logger.info("Prediction completed successfully!")
    return results


def list_available_models():
    """List all registered models in MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        logger.info("Available models in MLflow:")
        if not models:
            logger.info("  No registered models found")
            return

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
    list_available_models()

    # Make predictions
    results = predict_housing_prices()

    if results:
        logger.info("\n" + "=" * 50)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Version: {results['model_version']}")
        logger.info("\nPredictions:")

        for i, (_, row) in enumerate(results["sample_data"].iterrows()):
            pred = results["predictions"][i]
            logger.info(f"  Sample {i+1}: ${pred:,.2f}")
            logger.info(
                f"    Features: Income=${row['Avg. Area Income']:,.0f}, "
                f"Age={row['Avg. Area House Age']}y, "
                f"Rooms={row['Avg. Area Number of Rooms']}, "
                f"Bedrooms={row['Avg. Area Number of Bedrooms']}, "
                f"Pop={row['Area Population']:,.0f}"
            )
    else:
        logger.error("Prediction failed!")
