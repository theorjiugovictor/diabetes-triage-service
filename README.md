# Diabetes Triage Service

ML service for predicting diabetes progression risk to help prioritize patient follow-ups in a virtual diabetes clinic.

## Overview

This service provides a REST API that predicts disease progression scores based on patient vitals and lab results. Higher scores indicate greater risk of disease progression, helping triage nurses prioritize follow-up calls.

## Quick Start

### Using Docker (Recommended)

Pull and run the latest release:

```bash
# Pull the image
docker pull ghcr.io/theorjiugovictor/diabetes-triage-service:v0.1.0

# Run the service
docker run -p 8000:8000 ghcr.io/theorjiugovictor/diabetes-triage-service:v0.1.0
```

The API will be available at `http://localhost:8000`

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/theorjiugovictor/diabetes-triage-service.git
   cd diabetes-triage-service
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python src/train.py
   ```

4. **Run the API server**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_version": "0.1.0"
}
```

### Predict Progression Score

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "prediction": 152.5,
  "model_version": "0.1.0"
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Features

### Input Schema

All features are normalized values from patient records:

- `age`: Age (normalized)
- `sex`: Sex (normalized)
- `bmi`: Body mass index (normalized)
- `bp`: Average blood pressure (normalized)
- `s1`: Total serum cholesterol (normalized)
- `s2`: Low-density lipoproteins (normalized)
- `s3`: High-density lipoproteins (normalized)
- `s4`: Total cholesterol / HDL ratio (normalized)
- `s5`: Log of serum triglycerides (normalized)
- `s6`: Blood sugar level (normalized)

### Response Format

- `prediction`: Numeric progression score (higher = greater risk)
- `model_version`: Version of the model used for prediction

## Model Details

### Version 0.1.0 (Baseline)

- **Algorithm**: Linear Regression
- **Preprocessing**: Standard Scaling
- **Training/Test Split**: 80/20
- **Random Seed**: 42 (for reproducibility)

See `CHANGELOG.md` for detailed metrics and improvements across versions.

## Development Workflow

### Creating a New Release

1. Update `MODEL_VERSION` in `src/train.py`
2. Make improvements and update `CHANGELOG.md`
3. Commit changes
4. Create and push a tag:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The GitHub Actions workflow will automatically:
- Train the model
- Run tests
- Build Docker image
- Run smoke tests
- Push to GitHub Container Registry
- Create a GitHub Release

### CI/CD Pipeline

- **PR/Push**: Linting, tests, training smoke test, artifact upload
- **Tag (v*)**: Full build, container tests, registry push, release creation

## Project Structure

```
diabetes-triage-service/
├── .github/workflows/     # GitHub Actions workflows
├── src/
│   ├── train.py          # Training script
│   ├── api.py            # FastAPI service
│   └── utils.py          # Utility functions
├── tests/                # Test suite
├── models/               # Trained models (generated)
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── CHANGELOG.md        # Version history
```

## Reproducibility

- Python version: 3.11
- Dependencies pinned in `requirements.txt`
- Random seeds set for deterministic training
- All training data from scikit-learn (stable dataset)

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

The CI pipeline will validate your changes automatically.
