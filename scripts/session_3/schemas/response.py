from pydantic import BaseModel


class HousingPredictionResponse(BaseModel):
    predicted_price: float
