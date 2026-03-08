from fastapi import APIRouter
from app.core.startup import ModelRegistry

router = APIRouter()

@router.get("/health")
def health():
    """
    Health check endpoint that includes model status.
    """
    model_loaded = ModelRegistry.model is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "Customer Churn Prediction API",
        "model_loaded": model_loaded,
        "model_path": ModelRegistry.model_path if model_loaded else None,
        "model_type": type(ModelRegistry.model).__name__ if model_loaded else None
    }

