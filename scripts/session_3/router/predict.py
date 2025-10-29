import mlflow.sklearn
import pandas as pd
from fastapi import APIRouter

from scripts.session_3.schemas.request import HousingPredictionRequest
from scripts.session_3.schemas.response import HousingPredictionResponse

model_name = "housing_prediction"
model_version = "1"
alias = "the_best"

model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.sklearn.load_model(model_uri)

housing_router = APIRouter(prefix="/housing")


# /housing/predict
@housing_router.post("/predict", response_model=HousingPredictionResponse)
def func_predict(request: HousingPredictionRequest) -> HousingPredictionResponse:
    input_data = {
        "Avg. Area Income": [request.average_area_income],
        "Avg. Area House Age": [request.average_area_house_age],
        "Avg. Area Number of Rooms": [request.average_area_number_of_rooms],
        "Avg. Area Number of Bedrooms": [request.average_area_number_of_bedrooms],
        "Area Population": [request.area_population],
    }
    df = pd.DataFrame(input_data)
    predictions = model.predict(df)
    return HousingPredictionResponse(predicted_price=predictions[0])
