from fastapi import APIRouter, HTTPException
import pandas as pd
import logging
from typing import Optional

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from app.services.prediction_service import PredictionService
from app.core.startup import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

# Prediction service initialized during app startup
prediction_service: Optional[PredictionService] = None

def get_prediction_service() -> PredictionService:
    """
    Get or initialize prediction service using loaded model and artifacts.
    Called on first prediction request.
    """
    global prediction_service
    
    if prediction_service is None:
        if ModelRegistry.model is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction service not available. Model not loaded during startup."
            )
        
        # Initialize prediction service with artifacts loaded from model_registry
        prediction_service = PredictionService(
            model=ModelRegistry.model,
            encoders=ModelRegistry.encoders,
            exclude_columns=ModelRegistry.exclude_columns
        )
        logger.info("✓ Prediction service initialized with loaded model artifacts")
    
    return prediction_service

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a churn prediction for a customer.
    
    Uses preprocessing logic from src.features transformers:
    - Excludes specified columns
    - Imputes missing values
    - Generates engineered features
    - Encodes categorical variables
    
    Args:
        PredictionRequest: Customer features
    
    Returns:
        PredictionResponse: Prediction result with probabilities
    
    Example:
        {
            "tenure_months": 12,
            "total_charges": 1500.50,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "support_calls": 3,
            "late_payments": 0
        }
    """
    try:
        service = get_prediction_service()
        
        # Convert request to DataFrame
        data_dict = request.dict()
        
        logger.debug(f"Processing prediction request: {data_dict}")
        
        # Make prediction
        result = service.predict_single(data_dict)
        
        logger.debug(f"Prediction result: {result}")
        
        return PredictionResponse(
            prediction=int(result["prediction"]),
            churn_probability=float(result["churn_probability"]),
            no_churn_probability=float(result["no_churn_probability"])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


