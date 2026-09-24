from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_predict_endpoint():
    payload = {
        "aqi": 211.0,
        "pm25": 102.74,
        "pm10": 120.87,
        "no2": 124.0,
        "so2": 115.52,
        "co": 1.54,
        "o3": 174.28,
        "year": 2025,
        "month": 1,
        "day": 30,
        "day_of_week": 3,
        "day_of_year": 30,
        "is_weekend": 0,
        "aqi_lag_1": 181.0,
        "aqi_lag_3": 157.0,
        "aqi_lag_7": 137.0,
        "aqi_rolling_mean_3": 145.6666666667,
        "aqi_rolling_mean_7": 176.8571428571,
        "aqi_rolling_std_7": 56.590424901,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_aqi" in result
    assert isinstance(result["predicted_aqi"], float)
    assert result["predicted_aqi"] > 0