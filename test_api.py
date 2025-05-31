import requests

url = "http://localhost:8000/predict/"

payload = {
    "TotalCharges": 1000.0,
    "Month_to_month": 1,
    "One_year": 0,
    "Two_year": 0,
    "PhoneService": 1,
    "tenure": 12
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())