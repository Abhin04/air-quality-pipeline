from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_aqi_random_forest.pkl"

FEATURE_COLUMNS = [
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_7",
    "aqi_rolling_mean_3",
    "aqi_rolling_mean_7",
    "aqi_rolling_std_7",
]


class AQIPredictionRequest(BaseModel):
    aqi: float
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    year: int
    month: int
    day: int
    day_of_week: int
    day_of_year: int
    is_weekend: int
    aqi_lag_1: float
    aqi_lag_3: float
    aqi_lag_7: float
    aqi_rolling_mean_3: float
    aqi_rolling_mean_7: float
    aqi_rolling_std_7: float


class AQIPredictionResponse(BaseModel):
    predicted_aqi: float


app = FastAPI(
    title="Mumbai AQI Prediction API",
    description="Next-day AQI prediction using the trained Random Forest model.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Mumbai AQI Prediction API",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=AQIPredictionResponse)
def predict(request: AQIPredictionRequest):
    input_data = pd.DataFrame(
        [request.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_aqi": float(prediction)
    }