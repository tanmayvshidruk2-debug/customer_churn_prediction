from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    churn_probability: float  # Probability of churn (0 to 1)
    no_churn_probability: float  # Probability of no churn (0 to 1)
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "churn_probability": 0.75,
                "no_churn_probability": 0.25
            }
        }

