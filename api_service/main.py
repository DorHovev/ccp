from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, ValidationInfo
import joblib
import pandas as pd
import os
import sys # For loguru stderr
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from prometheus_client import Counter, Histogram

# --- Logger Setup --- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=LOG_FORMAT, level=LOG_LEVEL, enqueue=True)

# --- FastAPI App Initialization --- #
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn based on input features.",
    version="1.1.0"
)

# Instrumentator for Prometheus metrics
Instrumentator().instrument(app).expose(app)

# --- Custom Prometheus Metrics for API --- #
API_PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Total number of predictions made by the API"
)
API_PREDICTION_ERRORS_TOTAL = Counter(
    "api_prediction_errors_total",
    "Total number of prediction errors encountered by the API"
)
API_PREDICTION_LATENCY_SECONDS = Histogram(
    "api_prediction_latency_seconds",
    "Prediction request latency in seconds"
)

# --- Pydantic Input Model with Validation --- #
class PredictionFeatures(BaseModel):
    TotalCharges: float
    Month_to_month: int
    One_year: int
    Two_year: int
    PhoneService: int # Expecting 0 or 1
    tenure: int

    @field_validator('PhoneService')
    @classmethod
    def phone_service_must_be_binary(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError('PhoneService must be 0 (No) or 1 (Yes)')
        return v

    @field_validator('Month_to_month', 'One_year', 'Two_year')
    @classmethod
    def contract_type_must_be_binary(cls, v: int, info: ValidationInfo) -> int:
        if v not in [0, 1]:
            raise ValueError(f'{info.field_name} must be 0 or 1 (one-hot encoded)')
        return v
    
    @field_validator('tenure')
    @classmethod
    def tenure_must_be_positive(cls,v: int) -> int:
        if v < 0:
            raise ValueError('tenure must be non-negative')
        return v

# --- Model Loading --- #
MODEL_FILE_PATH = "churn_model.pickle"
MODEL_COLUMNS_ORDERED = [
    'TotalCharges',
    'Month-to-month',
    'One year',
    'Two year',
    'PhoneService',
    'tenure'
]
churn_prediction_model = None

@app.on_event("startup")
def load_model():
    global churn_prediction_model
    try:
        if not os.path.exists(MODEL_FILE_PATH):
            logger.critical(f"Model file not found at startup: {MODEL_FILE_PATH}")
            # Application will still start, but /predict will fail until model is present
            churn_prediction_model = None 
            return
        churn_prediction_model = joblib.load(MODEL_FILE_PATH)
        logger.info(f"Churn prediction model loaded successfully from {MODEL_FILE_PATH}.")
    except Exception as e:
        logger.critical(f"Failed to load model at startup from {MODEL_FILE_PATH}: {e}", exc_info=True)
        churn_prediction_model = None # Ensure model is None if loading fails

# --- API Endpoints --- #
@app.get("/health", tags=["Health"])
async def health_check():
    """Check the health of the API."""
    if churn_prediction_model is None:
        logger.warning("Health check: Model is not loaded.")
        # Still return 200, but indicate model status
        return {"status": "healthy", "model_loaded": False, "message": "API is running, but model is not loaded."}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict/", tags=["Predictions"])
@API_PREDICTION_LATENCY_SECONDS.time()
async def predict_churn(input_data: PredictionFeatures):
    """
    Predict customer churn based on input features.
    Requires one-hot encoded 'Contract' type columns and binary 'PhoneService'.
    """
    if churn_prediction_model is None:
        logger.error("Prediction attempt failed: Model is not loaded.")
        API_PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")

    try:
        logger.debug(f"Received input for prediction: {input_data.dict()}")
        # Create DataFrame in the exact order expected by the model
        data_dict = {
            'TotalCharges': [input_data.TotalCharges],
            'Month-to_month': [input_data.Month_to_month],
            'One_year': [input_data.One_year],
            'Two_year': [input_data.Two_year],
            'PhoneService': [input_data.PhoneService],
            'tenure': [input_data.tenure]
        }
        input_df = pd.DataFrame(data_dict, columns=MODEL_COLUMNS_ORDERED)
        
        logger.debug(f"DataFrame for prediction: \n{input_df.to_string()}")

        prediction = churn_prediction_model.predict(input_df)
        probabilities = churn_prediction_model.predict_proba(input_df)

        churn_probability = probabilities[0][1] # Assuming class 1 is churn
        no_churn_probability = probabilities[0][0]

        logger.info(f"Prediction successful for input. Churn: {int(prediction[0])}, Prob: {churn_probability:.4f}")
        API_PREDICTIONS_TOTAL.inc()

        return {
            "customer_input": input_data.dict(),
            "prediction_label": "Churn" if int(prediction[0]) == 1 else "No Churn",
            "prediction_value": int(prediction[0]),
            "probability_churn": float(f"{churn_probability:.4f}"),
            "probability_no_churn": float(f"{no_churn_probability:.4f}")
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        API_PREDICTION_ERRORS_TOTAL.inc()
        # Log the input that caused the error for debugging, be careful with sensitive data
        logger.debug(f"Input data that caused error: {input_data.dict()}") 
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# To run this app (outside Docker for testing):
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Customer Churn Prediction API locally on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000) 