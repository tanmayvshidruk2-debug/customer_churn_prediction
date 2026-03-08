from fastapi import APIRouter, HTTPException
import logging
from app.core.startup import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/model-version")
def get_model_version():
    """
    Get the current model version and metadata.
    
    Returns information about the loaded model including:
    - Model path and version (timestamp)
    - Model type and performance metrics
    - Preprocessing configuration
    """
    if ModelRegistry.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check service initialization."
        )
    
    try:
        version_info = {
            "model_loaded": True,
            "model_path": ModelRegistry.model_path,
            "model_type": type(ModelRegistry.model).__name__,
            "model_version": ModelRegistry.model_path.split("model_")[-1] if ModelRegistry.model_path else None,
            "exclude_columns": ModelRegistry.exclude_columns,
            "has_encoders": ModelRegistry.encoders is not None,
            "num_encoders": len(ModelRegistry.encoders) if ModelRegistry.encoders else 0
        }
        
        # Add metadata if available
        if ModelRegistry.metadata:
            version_info["metadata"] = ModelRegistry.metadata
        
        logger.info(f"Model version info requested: {version_info['model_version']}")
        
        return version_info
    
    except Exception as e:
        logger.error(f"Error retrieving model version: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model version: {str(e)}"
        )
