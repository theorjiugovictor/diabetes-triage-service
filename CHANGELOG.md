# Changelog

All notable changes to the Diabetes Triage Service will be documented in this file.

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
