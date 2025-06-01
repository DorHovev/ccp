import joblib
import pandas as pd
from batch_processor import config
from batch_processor.monitoring import logger, record_error, PREDICTIONS_MADE_TOTAL

class ModelPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model successfully loaded from {self.model_path}")
        except FileNotFoundError:
            logger.critical(f"Model file not found at {self.model_path}. Predictions will fail.")
            record_error("model_load_failure", f"File not found: {self.model_path}")
            # self.model remains None, predict method will handle this
        except Exception as e:
            logger.critical(f"Error loading model from {self.model_path}: {e}")
            record_error("model_load_failure", f"Loading error: {e}")
            # self.model remains None

    def predict(self, features_df: pd.DataFrame):
        if self.model is None:
            logger.error("Model is not loaded. Cannot make predictions.")
            # Return empty arrays or raise an exception based on how the caller should handle this
            return [], [] # Empty predictions and probabilities
        
        if not isinstance(features_df, pd.DataFrame):
            logger.error("Input to predict must be a Pandas DataFrame.")
            record_error("prediction_input_error", "Input was not a DataFrame")
            return [], []
            
        if features_df.empty:
            logger.info("Received an empty DataFrame for prediction. No predictions to make.")
            return [], []

        # Ensure columns are in the order the model expects
        try:
            ordered_features_df = features_df[config.MODEL_COLUMNS_ORDERED]
        except KeyError as e:
            logger.error(f"Missing expected columns for prediction: {e}. Columns available: {features_df.columns.tolist()}")
            record_error("prediction_input_error", f"Missing columns: {e}")
            return [],[]

        try:
            predictions = self.model.predict(ordered_features_df)
            probabilities = self.model.predict_proba(ordered_features_df)
            PREDICTIONS_MADE_TOTAL.inc(len(predictions))
            logger.info(f"Successfully made {len(predictions)} predictions.")
            return predictions.tolist(), probabilities.tolist() # Return as lists
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            record_error("prediction_runtime_error", f"Error: {e}")
            return [], [] 