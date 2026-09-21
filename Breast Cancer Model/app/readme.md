# Breast Cancer Prediction API

A FastAPI-based REST API for breast cancer prediction using machine learning.

## Overview

This API serves a machine learning model that predicts whether a breast tumor is benign or malignant based on various features extracted from diagnostic images.

## Project Structure

```
app/
├── server.py         # FastAPI application
├── model.joblib      # Trained ML model
├── Dockerfile        # Docker configuration
└── requirements.txt  # Python dependencies
```

## Setup and Installation

### Local Development

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn server:app --reload
```

### Docker Deployment

1. Build the Docker image:

```bash
docker build -t breast-cancer-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 breast-cancer-api
```

## API Endpoints

### GET /

Health check endpoint that returns a welcome message.

### POST /predict

Predicts whether a tumor is benign or malignant.

**Request Body:**

```json
{
    "features": [array_of_features]
}
```

**Response:**

```json
{
    "predicted_class": "Benign" or "Malignant"
}
```

## Requirements

- Python 3.9+
- FastAPI
- Scikit-learn
- NumPy
- Uvicorn
- Docker (optional)

## Model Information

The model is trained to classify breast cancer tumors as either:

- Benign (B): Non-cancerous
- Malignant (M): Cancerous

## License

[Your chosen license]

## Authors

[Your name]
