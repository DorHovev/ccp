from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from .logger import logger
from .model_loader import load_model
from .routes import health, predict
from .exception_handlers import validation_exception_handler
from fastapi.exceptions import RequestValidationError

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn based on input features.",
    version="1.1.0"
)

Instrumentator().instrument(app).expose(app)

# Register routers
app.include_router(health.router)
app.include_router(predict.router)

# Register exception handler
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Startup event for model loading
def on_startup():
    load_model()

app.add_event_handler("startup", on_startup)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Customer Churn Prediction API locally on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000) 