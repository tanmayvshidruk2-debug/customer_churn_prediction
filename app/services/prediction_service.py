import pandas as pd
import logging
from typing import Dict
import traceback

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Service for making predictions using the trained XGBoost model.
    Uses preprocessing aligned with src.features transformers.
    """
    
    def __init__(self, model=None, encoders=None, exclude_columns=None):
        """
        Initialize prediction service.
        
        Args:
            model: Trained model object
            encoders: Dictionary of fitted encoders for categorical variables
            exclude_columns: List of columns to exclude from prediction
        """
        self.model = model
        self.encoders = encoders
        self.exclude_columns = exclude_columns or [
            "customer_id", "age", "monthly_charges", "Date",
            "avg_monthly_value", "support_call_ratio", "payment_risk"
        ]
        
        # Initialize preprocessing service with loaded artifacts
        from app.services.preprocessing_service import PreprocessingService
        self.preprocessing_service = PreprocessingService(
            encoders=self.encoders,
            exclude_columns=self.exclude_columns
        )
        
        if self.model:
            logger.info(f"PredictionService initialized with model: {type(self.model).__name__}")
    
    def predict(
        self,
        data: pd.DataFrame,
        return_probability: bool = True
    ) -> Dict:
        """
        Make predictions on input data.
        
        Args:
            data: DataFrame with customer features
            return_probability: If True, return probability; if False, return class prediction
        
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize with a valid model.")
        
        try:
            logger.debug(f"Preprocessing {len(data)} samples")
            # Preprocess data using src transformers pipeline
            processed_data = self.preprocessing_service.preprocess(data)
            
            logger.debug(f"Making predictions on {len(processed_data)} samples")
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            result = {
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                "n_samples": len(processed_data)
            }
            
            # Add probabilities if model supports predict_proba
            if return_probability and hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(processed_data)
                result["churn_probability"] = probabilities[:, 1].tolist()
                result["no_churn_probability"] = probabilities[:, 0].tolist()
            
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def predict_single(self, data_dict: Dict) -> Dict:
        """
        Make prediction for a single sample.
        
        Args:
            data_dict: Dictionary with customer features
        
        Returns:
            Dictionary with prediction and probability
        """
        df = pd.DataFrame([data_dict])
        result = self.predict(df, return_probability=True)
        
        return {
            "prediction": result["predictions"][0],
            "churn_probability": result.get("churn_probability", [0])[0],
            "no_churn_probability": result.get("no_churn_probability", [1])[0]
        }

        

