"""
Comprehensive test suite for Customer Churn Prediction API.

Tests the following endpoints:
- GET /api/v1/health - Health check with model status
- GET /api/v1/model-version - Model version and metadata
- POST /api/v1/predict - Churn prediction
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.core.startup import ModelRegistry
from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock XGBoost model for testing"""
    model = Mock()
    model.predict = Mock(return_value=np.array([0, 1, 0]))
    model.predict_proba = Mock(return_value=np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1]
    ]))
    model.__class__.__name__ = "XGBClassifier"
    return model


@pytest.fixture
def mock_encoders():
    """Create mock LabelEncoders for categorical columns"""
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    categorical_cols = ["contract_type", "internet_service"]
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit with some sample data
        le.fit(["month-to-month", "one-year", "two-year"] 
               if col == "contract_type" 
               else ["fiber_optic", "dsl", "cable"])
        encoders[col] = le
    
    return encoders


@pytest.fixture
def setup_model_registry(mock_model, mock_encoders):
    """Setup ModelRegistry with mock data before tests"""
    ModelRegistry.model = mock_model
    ModelRegistry.encoders = mock_encoders
    ModelRegistry.metadata = {
        "model_type": "xgboost",
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.78,
        "f1": 0.80,
        "roc_auc": 0.88
    }
    ModelRegistry.model_path = "model_registry/model_20260308_152718"
    ModelRegistry.exclude_columns = [
        "customer_id", "age", "monthly_charges", "Date",
        "avg_monthly_value", "support_call_ratio", "payment_risk"
    ]
    
    yield
    
    # Cleanup after tests
    ModelRegistry.model = None
    ModelRegistry.encoders = None
    ModelRegistry.metadata = None
    ModelRegistry.model_path = None


# ============================================================================
# HEALTH CHECK ENDPOINT TESTS
# ============================================================================

class TestHealthCheckEndpoint:
    """Test suite for GET /api/v1/health endpoint"""
    
    def test_health_check_with_model_loaded(self, client, setup_model_registry):
        """Test health check returns healthy status when model is loaded"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "Customer Churn Prediction API"
        assert data["model_loaded"] is True
        assert data["model_type"] == "XGBClassifier"
        assert "model_path" in data
    
    def test_health_check_model_path_in_response(self, client, setup_model_registry):
        """Test that health check includes model path"""
        response = client.get("/api/v1/health")
        data = response.json()
        
        assert data["model_path"] == "model_registry/model_20260308_152718"
    
    def test_health_check_model_type_in_response(self, client, setup_model_registry):
        """Test that health check includes correct model type"""
        response = client.get("/api/v1/health")
        data = response.json()
        
        assert data["model_type"] == "XGBClassifier"
    
    def test_health_check_without_model(self, client):
        """Test health check returns degraded status when model not loaded"""
        ModelRegistry.model = None
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["model_path"] is None
    
    def test_health_check_response_structure(self, client, setup_model_registry):
        """Test that health check response has all required fields"""
        response = client.get("/api/v1/health")
        data = response.json()
        
        required_fields = ["status", "service", "model_loaded", "model_path", "model_type"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ============================================================================
# MODEL VERSION ENDPOINT TESTS
# ============================================================================

class TestModelVersionEndpoint:
    """Test suite for GET /api/v1/model-version endpoint"""
    
    def test_get_model_version_success(self, client, setup_model_registry):
        """Test successful model version retrieval"""
        response = client.get("/api/v1/model-version")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_loaded"] is True
        assert data["model_type"] == "XGBClassifier"
        assert data["model_version"] == "20260308_152718"
    
    def test_get_model_version_includes_metadata(self, client, setup_model_registry):
        """Test that model version includes metadata"""
        response = client.get("/api/v1/model-version")
        data = response.json()
        
        assert "metadata" in data
        assert data["metadata"]["accuracy"] == 0.85
        assert data["metadata"]["f1"] == 0.80
        assert data["metadata"]["roc_auc"] == 0.88
    
    def test_get_model_version_exclude_columns(self, client, setup_model_registry):
        """Test that model version includes exclude columns configuration"""
        response = client.get("/api/v1/model-version")
        data = response.json()
        
        assert "exclude_columns" in data
        assert "customer_id" in data["exclude_columns"]
        assert "age" in data["exclude_columns"]
        assert len(data["exclude_columns"]) == 7
    
    def test_get_model_version_encoders_info(self, client, setup_model_registry):
        """Test that model version includes encoder information"""
        response = client.get("/api/v1/model-version")
        data = response.json()
        
        assert data["has_encoders"] is True
        assert data["num_encoders"] == 2
    
    def test_get_model_version_without_model(self, client):
        """Test model version endpoint returns 503 when model not loaded"""
        ModelRegistry.model = None
        
        response = client.get("/api/v1/model-version")
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    def test_get_model_version_response_structure(self, client, setup_model_registry):
        """Test that model version response has all required fields"""
        response = client.get("/api/v1/model-version")
        data = response.json()
        
        required_fields = [
            "model_loaded", "model_path", "model_type", 
            "model_version", "exclude_columns", "has_encoders", "num_encoders"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ============================================================================
# PREDICT ENDPOINT TESTS
# ============================================================================

class TestPredictEndpoint:
    """Test suite for POST /api/v1/predict endpoint"""
    
    @pytest.fixture
    def valid_prediction_request(self):
        """Valid request data for prediction"""
        return {
            "tenure_months": 12,
            "total_charges": 1500.50,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "support_calls": 3,
            "late_payments": 0
        }
    
    def test_predict_success(self, client, setup_model_registry, valid_prediction_request):
        """Test successful prediction"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "churn_probability" in data
        assert "no_churn_probability" in data
        assert data["prediction"] in [0, 1]
    
    def test_predict_probabilities_sum_to_one(self, client, setup_model_registry, valid_prediction_request):
        """Test that probabilities sum to 1.0"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        data = response.json()
        
        total_prob = data["churn_probability"] + data["no_churn_probability"]
        assert abs(total_prob - 1.0) < 0.001
    
    def test_predict_probability_range(self, client, setup_model_registry, valid_prediction_request):
        """Test that probabilities are in valid range [0, 1]"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        data = response.json()
        
        assert 0 <= data["churn_probability"] <= 1
        assert 0 <= data["no_churn_probability"] <= 1
    
    def test_predict_with_various_inputs(self, client, setup_model_registry):
        """Test predictions with different input values"""
        test_cases = [
            {
                "tenure_months": 1,
                "total_charges": 100.0,
                "contract_type": "month-to-month",
                "internet_service": "dsl",
                "support_calls": 0,
                "late_payments": 5
            },
            {
                "tenure_months": 60,
                "total_charges": 5000.0,
                "contract_type": "two-year",
                "internet_service": "cable",
                "support_calls": 1,
                "late_payments": 0
            }
        ]
        
        for request_data in test_cases:
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "churn_probability" in data
    
    def test_predict_missing_required_field(self, client, setup_model_registry):
        """Test prediction fails with missing required field"""
        invalid_request = {
            "tenure_months": 12,
            "total_charges": 1500.50,
            # Missing contract_type
            "internet_service": "fiber_optic",
            "support_calls": 3,
            "late_payments": 0
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_invalid_data_type(self, client, setup_model_registry):
        """Test prediction fails with invalid data type"""
        invalid_request = {
            "tenure_months": "invalid",  # Should be int
            "total_charges": 1500.50,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "support_calls": 3,
            "late_payments": 0
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_predict_response_schema(self, client, setup_model_registry, valid_prediction_request):
        """Test that prediction response matches PredictionResponse schema"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        data = response.json()
        
        # Verify response matches schema
        assert isinstance(data["prediction"], int)
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["no_churn_probability"], float)
    

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_health_then_predict_flow(self, client, setup_model_registry):
        """Test typical client flow: check health, then make prediction"""
        # First, check health
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Then make prediction
        prediction_request = {
            "tenure_months": 12,
            "total_charges": 1500.50,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "support_calls": 3,
            "late_payments": 0
        }
        
        predict_response = client.post("/api/v1/predict", json=prediction_request)
        assert predict_response.status_code == 200
    
    def test_all_endpoints_accessibility(self, client, setup_model_registry):
        """Test that all endpoints are accessible"""
        endpoints = [
            ("GET", "/api/v1/health", None),
            ("GET", "/api/v1/model-version", None),
            ("POST", "/api/v1/predict", {
                "tenure_months": 12,
                "total_charges": 1500.50,
                "contract_type": "month-to-month",
                "internet_service": "fiber_optic",
                "support_calls": 3,
                "late_payments": 0
            })
        ]
        
        for method, endpoint, data in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=data)
            
            assert response.status_code in [200, 503], f"Endpoint {endpoint} failed"


# ============================================================================
# PERFORMANCE / STRESS TESTS
# ============================================================================

class TestAPIPerformance:
    """Performance and stress tests for API"""
    
    def test_multiple_predictions(self, client, setup_model_registry):
        """Test API handles multiple prediction requests"""
        request_data = {
            "tenure_months": 12,
            "total_charges": 1500.50,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "support_calls": 3,
            "late_payments": 0
        }
        
        num_requests = 10
        for _ in range(num_requests):
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
    
    def test_health_check_performance(self, client, setup_model_registry):
        """Test health check endpoint performance"""
        num_requests = 20
        
        for _ in range(num_requests):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
