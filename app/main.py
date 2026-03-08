from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import os
from app.api.v1.router import router as api_router
from app.core.startup import initialize_model_registry

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for FastAPI app.
    Loads model, encoders, and preprocessing artifacts from model_registry on startup.
    """
    # Startup
    logger.info("Starting up Customer Churn API...")
    try:
        initialize_model_registry()
        logger.info("✓ Model registry initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize model registry: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Churn API...")

app = FastAPI(
    title="Customer Churn API",
    description="API for predicting customer churn",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"status": "ok", "service": "Customer Churn Prediction API"}
