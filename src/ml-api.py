from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd






#---------------------------------------------------------------------
# ML Model
#---------------------------------------------------------------------


# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='target')


# Show basic dataset info
print("Feature names:", diabetes.feature_names)
print("\nFirst 5 rows of features:")
print(X.head())
print("\nFirst 5 target values:")
print(y.head())


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# Model coefficients
print("\nModel coefficients:")
for name, coef in zip(diabetes.feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# Make predictions
y_pred = model.predict(X_test_scaled)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error on test set: {mse:.2f}")
print(f"R² Score on test set: {r2:.4f}")






#---------------------------------------------------------------------
# API
#---------------------------------------------------------------------

# Create the FastAPI app
app = FastAPI(title="Diabetes Regression Model API")

# Define the input schema
class DiabetesFeatures(BaseModel):
    features: List[float]

@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Regression API!"}

@app.post("/predict")
def predict(data: DiabetesFeatures):
    # Validate input size
    if len(data.features) != X.shape[1]:
        raise HTTPException(status_code=400, detail=f"Expected {X.shape[1]} features, got {len(data.features)}")

    # Scale input features using the same scaler
    input_scaled = scaler.transform([data.features])
    
    # Predict
    prediction = model.predict(input_scaled)[0]

    return {
        "prediction": prediction
    }



## run with "fastapi run ml-api.py "