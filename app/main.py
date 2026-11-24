import json
import time
import joblib
import pandas as pd

from datetime import datetime, timezone
from fastapi import FastAPI

from app.schemas import (
    PredictionInput,
    PredictionOutput
)

from src.mp014.preprocessing import (
    ROOT,
    load_config,
    clean_features
)


# Load project configuration
config = load_config()


# Load champion model
model_path = ROOT / config["artifacts"]["model_path"]

model = joblib.load(model_path)


# Initialize FastAPI application
app = FastAPI(
    title="ATM Cash Demand Prediction API",
    description="ML-powered ATM cash demand prediction service",
    version="1.0.0"
)


# Health check endpoint
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    start_time = time.perf_counter()

    # Convert input into DataFrame
    input_df = pd.DataFrame([
        data.model_dump()
    ])

    # Clean input features
    cleaned_data = clean_features(input_df)

    # Generate prediction
    prediction = float(
        model.predict(cleaned_data)[0]
    )

    # Calculate prediction latency
    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    # Structured JSON logging
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "prediction",
        "endpoint": "/predict",
        "model_name": config["mlflow"]["registered_model_name"],
        "model_alias": "champion",
        "prediction": prediction,
        "latency_ms": latency_ms,
        "status_code": 200
    }

    # Print one JSON log per prediction
    print(json.dumps(log_entry), flush=True)

    # Return prediction
    return PredictionOutput(
        prediction=prediction
    )