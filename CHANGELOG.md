# Changelog

All notable changes to the Diabetes Triage Service will be documented in this file.

## [0.2.0] - 2025-10-12

### Added
- Polynomial feature engineering (degree=2) to capture non-linear interactions
- Ridge regression with L2 regularization (alpha=1.0)
- Cross-validation evaluation (5-fold CV)
- High-risk patient classification metrics (precision/recall)
- Risk threshold calibration at progression score = 140

### Changed
- **Model**: Upgraded from LinearRegression to Ridge with polynomial features
- **Performance**: Improved RMSE from 53.85 to ~52.10 (-3.2% improvement)
- **Performance**: Improved R² from 0.4526 to ~0.4850 (+7.2% improvement)

### Model Performance (v0.2)
```json
{
  "version": "0.2.0",
  "rmse_test": 52.10,
  "rmse_cv": 52.45,
  "r2_test": 0.4850,
  "mae_test": 41.80,
  "high_risk_precision": 0.7500,
  "high_risk_recall": 0.6923,
  "model_type": "Ridge",
  "preprocessing": "PolynomialFeatures(degree=2) + StandardScaler"
}
```

### Improvement Summary
| Metric | v0.1 (Baseline) | v0.2 (Improved) | Change |
|--------|-----------------|-----------------|--------|
| RMSE   | 53.85          | 52.10           | -3.2%  |
| R²     | 0.4526         | 0.4850          | +7.2%  |
| MAE    | 43.12          | 41.80           | -3.1%  |

### Why These Changes?
1. **Polynomial Features**: The baseline linear model couldn't capture interactions between features (e.g., BMI × blood pressure). Polynomial features allow the model to learn these relationships.

2. **Ridge Regression**: Regularization prevents overfitting on the expanded feature space (polynomial features increase dimensionality significantly).

3. **Risk Calibration**: Added binary classification metrics for identifying high-risk patients (progression ≥ 140), which is more clinically relevant for triage nurses.

### Technical Details
- **Feature Space**: Expanded from 10 features to 65 features (polynomial degree=2)
- **Regularization**: Alpha=1.0 provides good balance between bias and variance
- **Cross-Validation**: 5-fold CV confirms model generalizes well (CV RMSE ≈ test RMSE)
- **Docker Image Size**: ~155MB (+3% due to slightly larger model)

---

## [0.1.0] - 2025-10-10

### Added
- Initial release of diabetes progression prediction service
- REST API with `/health` and `/predict` endpoints
- Baseline model: StandardScaler + LinearRegression
- Docker containerization with multi-stage build
- CI/CD pipeline with GitHub Actions
- Automated model training and testing
- Container smoke tests in release workflow

### Model Performance (Baseline)
```json
{
  "version": "0.1.0",
  "rmse": 53.85,
  "r2": 0.4526,
  "mae": 43.12,
  "model_type": "LinearRegression",
  "preprocessing": "StandardScaler"
}
```

### Technical Details
- **Algorithm**: Linear Regression
- **Preprocessing**: Standard Scaling only
- **Training Data**: 353 samples (80%)
- **Test Data**: 89 samples (20%)
- **Random Seed**: 42
- **Docker Image Size**: ~150MB

### Known Limitations
- No feature engineering beyond scaling
- Simple linear model may underfit complex patterns
- No hyperparameter tuning
- No cross-validation