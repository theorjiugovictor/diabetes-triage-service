"""
Training script for diabetes progression prediction model.
Version 0.1 - Baseline with StandardScaler + LinearRegression
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_VERSION = "0.1.0"


def load_data():
    """Load and prepare the diabetes dataset."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]
    return X, y


def train_model(X_train, y_train):
    """Train baseline model: StandardScaler + LinearRegression."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    metrics = {
        "version": MODEL_VERSION,
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "model_type": "LinearRegression",
        "preprocessing": "StandardScaler",
        "random_seed": RANDOM_SEED,
        "n_test_samples": len(y_test)
    }
    
    return metrics


def save_artifacts(model, metrics, output_dir="models"):
    """Save model and metrics."""
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


def main():
    """Main training pipeline."""
    print(f"Training model version {MODEL_VERSION}")
    print("=" * 50)
    
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
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nModel Performance:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    
    # Save
    print("\nSaving artifacts...")
    save_artifacts(model, metrics)
    
    print("\nTraining complete!")
    return metrics


if __name__ == "__main__":
    main()