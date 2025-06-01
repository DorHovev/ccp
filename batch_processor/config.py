import os
from pathlib import Path

# --- General Configuration ---
APP_NAME = "BatchChurnPrediction"
MODEL_VERSION = "batch_v1.1" # Example version

# --- Database Configuration --- #
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pass")
DB_HOST = os.getenv("DB_HOST", "db") # Service name in Docker Compose
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "mlops_db")
DATABASE_URL = f"postgresql://user:pass@localhost:5432/mlops_db"

# --- Model and Data Paths --- #
MODEL_PATH = os.getenv("MODEL_PATH", "./churn_model.pickle")

# Use a folder for input CSVs
INPUT_CSV_FOLDER = os.getenv("INPUT_CSV_FOLDER", "input_data")
INPUT_CSV_FILES = [str(p) for p in Path(INPUT_CSV_FOLDER).rglob("*.csv")]

# --- Monitoring Configuration --- #
PROMETHEUS_PUSHGATEWAY = os.getenv("PROMETHEUS_PUSHGATEWAY", "pushgateway:9091")
PROMETHEUS_JOB_NAME = "batch_churn_prediction_job"

# --- Logging Configuration --- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# --- Preprocessing Configuration --- #
# Values from the notebook, make these configurable if they might change
TOTAL_CHARGES_DEFAULT_FILL = 2279.0 
# Example: if tenure mean from training dataset was, say, 32.34
TENURE_DEFAULT_MEAN_FILL = 32.34 

# Columns expected by the model after all preprocessing
# This order is critical for the RandomForestClassifier
MODEL_COLUMNS_ORDERED = [
    'TotalCharges',
    'Month-to-month',
    'One year',
    'Two year',
    'PhoneService',
    'tenure'
]

# Raw columns needed from the database to generate MODEL_COLUMNS_ORDERED
RAW_FEATURES_FOR_PREPROCESSING = ['customerid', 'totalcharges', 'contract', 'phoneservice', 'tenure'] 