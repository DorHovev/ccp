import sys
from loguru import logger
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
import os
from batch_processor import config # Use relative import for config

# --- Logger Setup with Loguru --- #
logger.remove()
logger.add(
    sys.stderr,
    format=config.LOG_FORMAT,
    level=config.LOG_LEVEL,
    enqueue=True,  # Useful for async or multiprocessing scenarios
)

# --- Prometheus Metrics Setup --- #
METRICS_REGISTRY = CollectorRegistry()

# --- Batch Job Metrics --- #
BATCH_JOB_LAST_SUCCESS_TIMESTAMP = Gauge(
    "batch_job_last_success_timestamp_seconds",
    "Timestamp of the last successful batch job completion",
    registry=METRICS_REGISTRY
)
BATCH_JOB_DURATION_SECONDS = Histogram(
    "batch_job_duration_seconds",
    "Duration of the batch job processing in seconds",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600), # Example buckets
    registry=METRICS_REGISTRY
)
BATCH_JOB_ERRORS_TOTAL = Counter(
    "batch_job_errors_total",
    "Total number of errors encountered during the batch job",
    ["error_type", "details"], # Added details label for more context
    registry=METRICS_REGISTRY
)

# --- Data Processing Metrics --- #
ROWS_LOADED_FROM_CSV_TOTAL = Counter(
    "batch_job_rows_loaded_from_csv_total",
    "Total number of new rows loaded from CSV files into the database",
    ["csv_file"],
    registry=METRICS_REGISTRY
)
ROWS_SKIPPED_DUPLICATES_TOTAL = Counter(
    "batch_job_rows_skipped_duplicates_total",
    "Total number of duplicate rows skipped during CSV loading",
    ["csv_file"],
    registry=METRICS_REGISTRY
)
ROWS_MISSING_CUSTOMERID_TOTAL = Counter(
    "batch_job_rows_missing_customerid_total",
    "Total number of rows skipped due to missing customerID during CSV loading",
    ["csv_file"],
    registry=METRICS_REGISTRY
)
DATA_CONVERSION_ERRORS_TOTAL = Counter(
    "batch_job_data_conversion_errors_total",
    "Total number of data conversion errors during CSV loading",
    ["csv_file", "column_name"],
    registry=METRICS_REGISTRY
)
ROWS_FETCHED_FOR_PREPROCESSING_TOTAL = Gauge(
    "batch_job_rows_fetched_for_preprocessing_total",
    "Number of rows fetched from the database for preprocessing in the current run",
    registry=METRICS_REGISTRY
)
ROWS_AFTER_PREPROCESSING_TOTAL = Gauge(
    "batch_job_rows_after_preprocessing_total",
    "Number of rows after preprocessing in the current run",
    registry=METRICS_REGISTRY
)

# --- Model Prediction Metrics --- #
PREDICTIONS_MADE_TOTAL = Counter(
    "batch_job_predictions_made_total",
    "Total number of predictions made by the model",
    registry=METRICS_REGISTRY
)
PREDICTIONS_PERSISTED_TOTAL = Counter(
    "batch_job_predictions_persisted_total",
    "Total number of predictions successfully persisted to the database",
    registry=METRICS_REGISTRY
)

# --- Push Metrics Function --- #
def push_metrics_to_gateway():
    """Pushes all registered metrics to the configured Pushgateway."""
    try:
        push_to_gateway(
            config.PROMETHEUS_PUSHGATEWAY, 
            job=config.PROMETHEUS_JOB_NAME, 
            registry=METRICS_REGISTRY
        )
        logger.info(f"Successfully pushed metrics to Pushgateway at {config.PROMETHEUS_PUSHGATEWAY} for job {config.PROMETHEUS_JOB_NAME}")
    except Exception as e:
        logger.error(f"Could not push metrics to Pushgateway: {e}")
        # This is a critical monitoring failure, might need a separate alert
        # For now, we'll log it. If this metric itself was a counter, it would be ironic.

# --- Utility for recording errors --- #
def record_error(error_type: str, details: str = "N/A"):
    """Increments the error counter and logs the error."""
    logger.error(f"Error Type: {error_type}, Details: {details}")
    BATCH_JOB_ERRORS_TOTAL.labels(error_type=error_type, details=details).inc() 