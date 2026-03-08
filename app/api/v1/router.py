from fastapi import APIRouter
from app.api.v1.endpoints import predict, health, model_info

router = APIRouter()
router.include_router(predict.router, tags=["Prediction"])
router.include_router(health.router, tags=["Health"])
router.include_router(model_info.router, tags=["Model"])

