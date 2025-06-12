import os
import joblib
from loguru import logger

MODEL_PATH =  "churn_model.pickle"
MODEL_COLUMNS_ORDERED = [
    'TotalCharges',
    'Month-to-month',
    'One year',
    'Two year',
    'PhoneService',
    'tenure'
]

churn_prediction_model = None

def load_model():
    global churn_prediction_model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.critical(f"Model file not found at startup: {MODEL_PATH}")
            churn_prediction_model = None
            return
        churn_prediction_model = joblib.load(MODEL_PATH)
        logger.info(f"Churn prediction model loaded successfully from {MODEL_PATH}.")
    except Exception as e:
        logger.critical(f"Failed to load model at startup from {MODEL_PATH}: {e}", exc_info=True)
        churn_prediction_model = None