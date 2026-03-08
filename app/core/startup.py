"""
Startup initialization for FastAPI app.
Loads model, encoders, and metadata from model_registry.
"""
import os
import joblib
import logging
from typing import Optional, Dict, Any
import yaml

logger = logging.getLogger(__name__)

# Global state
class ModelRegistry:
    model = None
    encoders = None
    metadata = None
    model_path = None
    exclude_columns = None

def get_latest_model_path() -> Optional[str]:
    """Get the path to the latest trained model in model_registry"""
    model_registry = "model_registry"
    
    if not os.path.exists(model_registry):
        logger.warning(f"Model registry directory '{model_registry}' not found")
        return None
    
    # Get all model directories sorted by timestamp (sorted lexicographically since format is YYYYMMDD_HHMMSS)
    models = sorted([d for d in os.listdir(model_registry) if d.startswith("model_")])
    
    if not models:
        logger.warning("No model directories found in model_registry")
        return None
    
    latest_model_dir = models[-1]
    model_path = os.path.join(model_registry, latest_model_dir, "model.pkl")
    model_dir = os.path.join(model_registry, latest_model_dir)
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        return None
    
    logger.info(f"Latest model found: {latest_model_dir}")
    return model_dir

def load_model(model_dir: str) -> Any:
    """Load the trained model"""
    model_path = os.path.join(model_dir, "model.pkl")
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def load_encoders(model_dir: str) -> Optional[Dict]:
    """Load categorical encoders from model directory"""
    encoders_path = os.path.join(model_dir, "encoders.pkl")
    try:
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            logger.info(f"✓ Encoders loaded from {encoders_path}")
            return encoders
        else:
            logger.warning(f"Encoders file not found at {encoders_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load encoders: {e}")
        raise

def load_metadata(model_dir: str) -> Optional[Dict]:
    """Load model metadata from YAML file"""
    metadata_path = os.path.join(model_dir, "model.yaml")
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                logger.info(f"✓ Metadata loaded from {metadata_path}")
                return metadata
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise

def initialize_model_registry():
    """
    Initialize model registry by loading latest model, encoders, and metadata.
    Stores in global ModelRegistry class.
    """
    logger.info("Initializing model registry...")
    
    model_dir = get_latest_model_path()
    if not model_dir:
        raise ValueError("No valid model found in model_registry")
    
    # Load model
    ModelRegistry.model = load_model(model_dir)
    
    # Load encoders
    ModelRegistry.encoders = load_encoders(model_dir)
    
    # Load metadata
    ModelRegistry.metadata = load_metadata(model_dir)
    
    # Store model path for reference
    ModelRegistry.model_path = model_dir
    
    # Set exclude columns from config or defaults
    ModelRegistry.exclude_columns = [
        "customer_id", "age", "monthly_charges", "Date",
        "avg_monthly_value", "support_call_ratio", "payment_risk"
    ]
    
    logger.info(f"Model registry initialized. Using model from: {model_dir}")
    logger.info(f"Model type: {type(ModelRegistry.model).__name__}")
    logger.info(f"Exclude columns: {ModelRegistry.exclude_columns}")
