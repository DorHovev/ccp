import pytest
from fastapi.testclient import TestClient
from api_service.main import app

client = TestClient(app)
import requests

def test_predict_valid():
    payload = {
        "TotalCharges": 1000.0,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction_value" in data
        assert data["prediction_value"] in [0, 1]

def test_predict_missing_field():
    payload = {
        "TotalCharges": 1000.0,
        # Missing 'Month_to_month'
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_predict_invalid_type():
    payload = {
        "TotalCharges": "not_a_number",
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_predict_minimum_values():
    payload = {
        "TotalCharges": 0.0,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 0,
        "tenure": 0
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction_value" in data

def test_predict_maximum_values():
    payload = {
        "TotalCharges": 100000.0,
        "Month_to_month": 0,
        "One_year": 1,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 100
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction_value" in data

@pytest.mark.parametrize("payload", [
    # Negative tenure
    {"TotalCharges": 100.0, "Month_to_month": 1, "One_year": 0, "Two_year": 0, "PhoneService": 1, "tenure": -1},
    # PhoneService not binary
    {"TotalCharges": 100.0, "Month_to_month": 1, "One_year": 0, "Two_year": 0, "PhoneService": 2, "tenure": 10},
    # All contract types zero
    {"TotalCharges": 100.0, "Month_to_month": 0, "One_year": 0, "Two_year": 0, "PhoneService": 1, "tenure": 10},
])
def test_predict_invalid_inputs(payload):
    with TestClient(app) as client:
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

def test_predict_missing_totalcharges():
    payload = {
        # "TotalCharges": 100.0,  # Missing
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 10
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data 