import mlflow.sklearn
import pandas as pd

model_name = "housing_prediction"
model_version = "1"
alias = "the_best"

model_uri = f"models:/{model_name}/{model_version}"
# model_uri = f"models:/{model_name}@{alias}"

model = mlflow.sklearn.load_model(model_uri)


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

    return df


predictions = model.predict(create_sample_data())
print(f"Predicted price: ${predictions[0]:,.2f}")
