from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    """
    Request schema for churn prediction.
    Excludes: customer_id, age, monthly_charges
    """
    tenure_months: int
    total_charges: float
    contract_type: str
    internet_service: str
    support_calls: int
    late_payments: int
    
    class Config:
        schema_extra = {
            "example": {
                "tenure_months": 12,
                "total_charges": 1500.50,
                "contract_type": "month-to-month",
                "internet_service": "fiber_optic",
                "support_calls": 3,
                "late_payments": 0
            }
        }

