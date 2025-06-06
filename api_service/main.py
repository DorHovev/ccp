from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator, ValidationInfo, model_validator
import joblib
import pandas as pd
import os
import sys # For loguru stderr
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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

DATA_VALIDATION_FAILURES = Counter(
    "api_data_validation_failures_total",
    "Number of API inputs that failed validation",
    ["validation_type"]
)

PREDICTION_CLASS_COUNT = Counter(
    "api_prediction_class_total",
    "Total predictions by class",
    ["class_label"]
)

FEATURE_MEAN = Gauge(
    "api_feature_mean_value",
    "Mean value of a feature in API requests",
    ["feature_name"]
)

UNHANDLED_EXCEPTIONS = Counter(
    "api_unhandled_exceptions_total",
    "Critical exceptions in API",
    ["exception_type"]
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

    @classmethod
    def validate_contract_one_hot(cls, values):
        # Ensure exactly one of the contract types is 1
        # In Pydantic v2, values is a model instance, not a dict
        contract_fields = [values.Month_to_month, values.One_year, values.Two_year]
        if contract_fields.count(1) != 1:
            raise ValueError('Exactly one of Month_to_month, One_year, or Two_year must be 1 (one-hot encoded)')
        return values

    # Register as a root validator
    _validate_contract_one_hot = model_validator(mode="after")(validate_contract_one_hot)

# --- Model Loading --- #
MODEL_PATH = os.getenv("MODEL_PATH", "churn_model.pickle")
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
        if not os.path.exists(MODEL_PATH):
            logger.critical(f"Model file not found at startup: {MODEL_PATH}")
            # Application will still start, but /predict will fail until model is present
            churn_prediction_model = None 
            return
        churn_prediction_model = joblib.load(MODEL_PATH)
        logger.info(f"Churn prediction model loaded successfully from {MODEL_PATH}.")
    except Exception as e:
        logger.critical(f"Failed to load model at startup from {MODEL_PATH}: {e}", exc_info=True)
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
async def predict_churn(input_data: PredictionFeatures):
    """
    Predict customer churn based on input features.
    Requires one-hot encoded 'Contract' type columns and binary 'PhoneService'.
    """
    with API_PREDICTION_LATENCY_SECONDS.time():
        if churn_prediction_model is None:
            logger.error("Prediction attempt failed: Model is not loaded.")
            API_PREDICTION_ERRORS_TOTAL.inc()
            raise HTTPException(status_code=503, detail="Model not available. Please try again later.")

        try:
            logger.debug(f"Received input for prediction: {input_data.dict()}")
            # Create DataFrame in the exact order expected by the model
            data_dict = {
                'TotalCharges': [input_data.TotalCharges],
                'Month-to-month': [input_data.Month_to_month],
                'One year': [input_data.One_year],
                'Two year': [input_data.Two_year],
                'PhoneService': [input_data.PhoneService],
                'tenure': [input_data.tenure]
            }
            input_df = pd.DataFrame(data_dict, columns=MODEL_COLUMNS_ORDERED)
            
            logger.debug(f"DataFrame for prediction: \n{input_df.to_string()}")

            for col in input_df.columns:
                FEATURE_MEAN.labels(feature_name=col).set(input_df[col].mean())

            prediction = churn_prediction_model.predict(input_df)
            probabilities = churn_prediction_model.predict_proba(input_df)

            churn_probability = probabilities[0][1] # Assuming class 1 is churn
            no_churn_probability = probabilities[0][0]

            logger.info(f"Prediction successful for input. Churn: {int(prediction[0])}, Prob: {churn_probability:.4f}")
            API_PREDICTIONS_TOTAL.inc()

            label = int(prediction[0])
            PREDICTION_CLASS_COUNT.labels(class_label=str(label)).inc()

            return {
                "customer_input": input_data.dict(),
                "prediction_label": "Churn" if int(prediction[0]) == 1 else "No Churn",
                "prediction_value": int(prediction[0]),
                "probability_churn": float(f"{churn_probability:.4f}"),
                "probability_no_churn": float(f"{no_churn_probability:.4f}")
            }
        except Exception as e:
            UNHANDLED_EXCEPTIONS.labels(exception_type=type(e).__name__).inc()
            logger.error(f"Error during prediction: {e}", exc_info=True)
            API_PREDICTION_ERRORS_TOTAL.inc()
            # Log the input that caused the error for debugging, be careful with sensitive data
            logger.debug(f"Input data that caused error: {input_data.dict()}") 
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    DATA_VALIDATION_FAILURES.labels(validation_type="pydantic").inc()
    # Convert all error messages to strings to avoid non-serializable objects
    errors = exc.errors()
    for err in errors:
        if 'msg' in err and not isinstance(err['msg'], str):
            err['msg'] = str(err['msg'])
        if 'ctx' in err and err['ctx']:
            for k, v in err['ctx'].items():
                if not isinstance(v, str):
                    err['ctx'][k] = str(v)
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "body": exc.body},
    )

# To run this app (outside Docker for testing):
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Customer Churn Prediction API locally on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000) 