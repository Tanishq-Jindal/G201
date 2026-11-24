import math

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


# Sample valid ATM input
def valid_input():
    return {
        "nearby_shops": 25,
        "days_to_salary": 5,
        "footfall_index": 70,
        "distance_to_market_km": 1.5,
        "avg_ticket_rs": 2500,
        "other_atms_1km": 3,
        "atm_type": "mall",
        "area": "Market"
    }


# Test 1: Health endpoint
def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Test 2: Valid prediction
def test_valid_prediction():

    response = client.post(
        "/predict",
        json=valid_input()
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert math.isfinite(response.json()["prediction"])


# Test 3: Invalid ATM count
def test_invalid_atm_count():

    data = valid_input()
    data["other_atms_1km"] = 30

    response = client.post("/predict", json=data)

    assert response.status_code == 422


# Test 4: Negative distance
def test_negative_distance():

    data = valid_input()
    data["distance_to_market_km"] = -5

    response = client.post("/predict", json=data)

    assert response.status_code == 422


# Test 5: Missing required feature
def test_missing_feature():

    data = valid_input()
    del data["avg_ticket_rs"]

    response = client.post("/predict", json=data)

    assert response.status_code == 422


# Test 6: Invalid ATM category
def test_invalid_atm_type():

    data = valid_input()
    data["atm_type"] = "unknown"

    response = client.post("/predict", json=data)

    assert response.status_code == 422