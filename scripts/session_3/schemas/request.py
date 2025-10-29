from pydantic import BaseModel


class HousingPredictionRequest(BaseModel):
    average_area_income: float
    average_area_house_age: float
    average_area_number_of_rooms: float
    average_area_number_of_bedrooms: float
    area_population: float
