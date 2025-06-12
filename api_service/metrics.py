from prometheus_client import Counter, Histogram, Gauge

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