from fastapi import APIRouter
from ..model_loader import churn_prediction_model
from ..logger import logger

router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check():
    """Check the health of the API."""
    if churn_prediction_model is None:
        logger.warning("Health check: Model is not loaded.")
        return {"status": "healthy", "model_loaded": False, "message": "API is running, but model is not loaded."}
    return {"status": "healthy", "model_loaded": True} 