"""
FastAPI service for diabetes progression prediction.
"""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load model at startup
MODEL_PATH = Path("models/model.pkl")
VERSION_PATH = Path("models/version.txt")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Load version
model_version = "unknown"
if VERSION_PATH.exists():
    with open(VERSION_PATH, "r") as f:
        model_version = f.read().strip()

# Create FastAPI app
app = FastAPI(
    title="Diabetes Triage Service",
    description="ML service for predicting diabetes progression risk",
    version=model_version,
)


class PatientFeatures(BaseModel):
    """Input schema for patient features."""

    age: float = Field(..., description="Age (normalized)")
    sex: float = Field(..., description="Sex (normalized)")
    bmi: float = Field(..., description="Body mass index (normalized)")
    bp: float = Field(..., description="Average blood pressure (normalized)")
    s1: float = Field(..., description="Total serum cholesterol (normalized)")
    s2: float = Field(..., description="Low-density lipoproteins (normalized)")
    s3: float = Field(..., description="High-density lipoproteins (normalized)")
    s4: float = Field(..., description="Total cholesterol / HDL (normalized)")
    s5: float = Field(..., description="Log of serum triglycerides (normalized)")
    s6: float = Field(..., description="Blood sugar level (normalized)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.02,
                "sex": -0.044,
                "bmi": 0.06,
                "bp": -0.03,
                "s1": -0.02,
                "s2": 0.03,
                "s3": -0.02,
                "s4": 0.02,
                "s5": 0.02,
                "s6": -0.001,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: float = Field(..., description="Predicted progression score")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_version": model_version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PatientFeatures):
    """
    Predict diabetes progression risk score.

    Higher scores indicate greater risk of disease progression.
    """
    try:
        # Convert to DataFrame with correct feature names
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])

        # Make prediction
        prediction = model.predict(df)[0]

        return {"prediction": float(prediction), "model_version": model_version}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Diabetes Triage Service",
        "version": model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }
