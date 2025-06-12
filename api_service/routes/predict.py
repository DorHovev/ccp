from fastapi import APIRouter, HTTPException
from ..models import PredictionFeatures
import api_service.model_loader as model_loader
from ..metrics import API_PREDICTION_LATENCY_SECONDS, API_PREDICTION_ERRORS_TOTAL, API_PREDICTIONS_TOTAL, PREDICTION_CLASS_COUNT, FEATURE_MEAN, UNHANDLED_EXCEPTIONS
from ..logger import logger
import pandas as pd

router = APIRouter()

@router.post("/predict/", tags=["Predictions"])
async def predict_churn(input_data: PredictionFeatures):
    """
    Predict customer churn based on input features.
    Requires one-hot encoded 'Contract' type columns and binary 'PhoneService'.
    """
    with API_PREDICTION_LATENCY_SECONDS.time():
        if model_loader.churn_prediction_model is None:
            logger.error("Prediction attempt failed: Model is not loaded.")
            API_PREDICTION_ERRORS_TOTAL.inc()
            raise HTTPException(status_code=503, detail="Model not available. Please try again later.")

        try:
            logger.debug(f"Received input for prediction: {input_data.dict()}")
            data_dict = {
                'TotalCharges': [input_data.TotalCharges],
                'Month-to-month': [input_data.Month_to_month],
                'One year': [input_data.One_year],
                'Two year': [input_data.Two_year],
                'PhoneService': [input_data.PhoneService],
                'tenure': [input_data.tenure]
            }
            input_df = pd.DataFrame(data_dict, columns=model_loader.MODEL_COLUMNS_ORDERED)
            logger.debug(f"DataFrame for prediction: \n{input_df.to_string()}")
            for col in input_df.columns:
                FEATURE_MEAN.labels(feature_name=col).set(input_df[col].mean())
            prediction = model_loader.churn_prediction_model.predict(input_df)
            probabilities = model_loader.churn_prediction_model.predict_proba(input_df)
            churn_probability = probabilities[0][1]
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
            logger.debug(f"Input data that caused error: {input_data.dict()}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") 