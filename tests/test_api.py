"""
Tests for the diabetes triage API.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Only import if model exists
def test_model_exists():
    """Test that model file exists."""
    model_path = Path("models/model.pkl")
    assert model_path.exists(), "Model file must exist before running API tests"


# Import app only after confirming model exists
try:
    from src.api import app
    client = TestClient(app)
    API_AVAILABLE = True
except RuntimeError:
    API_AVAILABLE = False
    client = None


@pytest.mark.skipif(not API_AVAILABLE, reason="Model not available")
class TestAPI:
    """Test suite for API endpoints."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
    
    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        payload = {
            "age": 0.02,
            "sex": -0.044,
            "bmi": 0.06,
            "bp": -0.03,
            "s1": -0.02,
            "s2": 0.03,
            "s3": -0.02,
            "s4": 0.02,
            "s5": 0.02,
            "s6": -0.001
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert isinstance(data["prediction"], (int, float))
    
    def test_predict_missing_field(self):
        """Test prediction with missing required field."""
        payload = {
            "age": 0.02,
            "sex": -0.044,
            # Missing other fields
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_type(self):
        """Test prediction with invalid data type."""
        payload = {
            "age": "not a number",
            "sex": -0.044,
            "bmi": 0.06,
            "bp": -0.03,
            "s1": -0.02,
            "s2": 0.03,
            "s3": -0.02,
            "s4": 0.02,
            "s5": 0.02,
            "s6": -0.001
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_training_script():
    """Test that training script can be imported."""
    try:
        from src import train
        assert hasattr(train, "main")
        assert hasattr(train, "MODEL_VERSION")
    except ImportError as e:
        pytest.fail(f"Failed to import training script: {e}")