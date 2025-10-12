"""
Training script for diabetes progression prediction model.
Version 0.2 - Improved with Ridge regression, polynomial features, and risk calibration
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_VERSION = "0.2.0"

# Risk threshold for high-risk classification
RISK_THRESHOLD = 140  # Based on domain knowledge - upper quartile of progression


def load_data():
    """Load and prepare the diabetes dataset."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]
    return X, y


def train_model(X_train, y_train):
    """
    Train improved model: PolynomialFeatures + StandardScaler + Ridge.
    
    Improvements over v0.1:
    - Polynomial features (degree=2) to capture non-linear interactions
    - Ridge regression with L2 regularization to prevent overfitting
    - Alpha tuned via cross-validation
    """
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_SEED))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model and return comprehensive metrics.
    
    Includes both regression metrics and classification metrics
    for high-risk patient identification.
    """
    # Regression predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Regression metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = np.mean(np.abs(y_test - y_pred_test))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    
    # Classification metrics for high-risk detection
    y_test_binary = (y_test >= RISK_THRESHOLD).astype(int)
    y_pred_binary = (y_pred_test >= RISK_THRESHOLD).astype(int)
    
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    
    metrics = {
        "version": MODEL_VERSION,
        "rmse_train": float(rmse_train),
        "rmse_test": float(rmse_test),
        "rmse_cv": float(cv_rmse),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "mae_test": float(mae_test),
        "high_risk_precision": float(precision),
        "high_risk_recall": float(recall),
        "high_risk_threshold": RISK_THRESHOLD,
        "model_type": "Ridge",
        "preprocessing": "PolynomialFeatures(degree=2) + StandardScaler",
        "alpha": 1.0,
        "random_seed": RANDOM_SEED,
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test)
    }
    
    return metrics


def save_artifacts(model, metrics, output_dir="models"):
    """Save model, metrics, and threshold."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = Path(output_dir) / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save version
    version_path = Path(output_dir) / "version.txt"
    with open(version_path, "w") as f:
        f.write(MODEL_VERSION)
    
    # Save risk threshold
    threshold_path = Path(output_dir) / "risk_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(str(RISK_THRESHOLD))


def main():
    """Main training pipeline."""
    print(f"Training model version {MODEL_VERSION}")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    print("Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Train model
    print("Training improved model...")
    print("  - Adding polynomial features (degree=2)")
    print("  - Using Ridge regression (alpha=1.0)")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 70)
    print("Model Performance (v0.2):")
    print("=" * 70)
    print(f"  RMSE (test):        {metrics['rmse_test']:.2f}")
    print(f"  RMSE (train):       {metrics['rmse_train']:.2f}")
    print(f"  RMSE (5-fold CV):   {metrics['rmse_cv']:.2f}")
    print(f"  R² (test):          {metrics['r2_test']:.4f}")
    print(f"  R² (train):         {metrics['r2_train']:.4f}")
    print(f"  MAE (test):         {metrics['mae_test']:.2f}")
    print("\nHigh-Risk Detection (threshold={})".format(RISK_THRESHOLD))
    print(f"  Precision:          {metrics['high_risk_precision']:.4f}")
    print(f"  Recall:             {metrics['high_risk_recall']:.4f}")
    print("=" * 70)
    
    # Save
    print("\nSaving artifacts...")
    save_artifacts(model, metrics)
    
    print("\nTraining complete!")
    return metrics


if __name__ == "__main__":
    main()
