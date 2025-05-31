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