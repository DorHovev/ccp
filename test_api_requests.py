import requests

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Error/edge case payloads
error_payloads = [
    # Missing field
    {
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    },
    # Wrong type
    {
        "TotalCharges": "not_a_number",
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    },
    # Negative value
    {
        "TotalCharges": -100.0,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": -5
    },
    # Extra field
    {
        "TotalCharges": 1000.0,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12,
        "ExtraField": 123
    },
    # Null value
    {
        "TotalCharges": None,
        "Month_to_month": 1,
        "One_year": 0,
        "Two_year": 0,
        "PhoneService": 1,
        "tenure": 12
    }
]

# Combine and multiply payloads for stress test
all_payloads = payloads * 100 + error_payloads * 20  # 100 valid, 20 of each error
random.shuffle(all_payloads)

def send_request(i, payload):
    try:
        response = requests.post(url, json=payload)
        print(f"Request {i} status: {response.status_code}, response: {response.json()}")
    except Exception as e:
        print(f"Request {i} failed: {e}")

# Use ThreadPoolExecutor for parallel requests (stress test)
num_workers = 10
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(send_request, i, payload) for i, payload in enumerate(all_payloads, 1)]
    for future in as_completed(futures):
        pass  # All output is printed in send_request

print("Stress test complete.") 