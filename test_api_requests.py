import requests
import time

url = "http://localhost:8000/predict/"

payloads = [
    {
        "TotalCharges": 1000.0,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    },
    {
        "TotalCharges": 250.5,
        "Month_to_month": 0,
        "One_year": 1,
        "Two_year": 0,
        "PhoneService": 0,
        "tenure": 24
    },
    {
        "TotalCharges": 5000.0,
        "Month_to_month": 0,
        "One_year": 0,
        "Two_year": 1,
        "PhoneService": 1,
        "tenure": 36
    },
    {
        "TotalCharges": 0.0,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 0,
        "tenure": 0
    },
    {
        "TotalCharges": 99999.9,
        "Month_to_month": 0,
        "One_year": 1,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 100
    }
]

for i, payload in enumerate(payloads, 1):
    response = requests.post(url, json=payload)
    print(f"Request {i} status code:", response.status_code)
    print(f"Request {i} response:", response.json())
    time.sleep(1)  # Small delay to make requests visible in monitoring 