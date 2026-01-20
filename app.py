import os
import joblib
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Artifacts
MODEL_PATH = "data/model/lgbm_model.pkl"
PROCESSOR_PATH = "data/model/feature_processor.pkl"
SCALER_PATH = "data/model/scaler.pkl"
ENCODERS_PATH = "data/model/encoders.pkl"

app = FastAPI(title="KKBox Churn Prediction API", version="2.0.0")

# Initialize Prometheus Monitoring
Instrumentator().instrument(app).expose(app)

# Global variables for artifacts
model = None
processor = None
scaler = None
encoders = None

@app.on_event("startup")
def load_models():
    global model, processor, scaler, encoders
    try:
        model = joblib.load(MODEL_PATH)
        processor = joblib.load(PROCESSOR_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        logger.info(" All MLOps artifacts loaded successfully.")
    except Exception as e:
        logger.error(f" Failed to load artifacts: {e}")

# Pydantic model for input validation
from typing import Optional

class ChurnInput(BaseModel):
    # msno is now optional. If not provided, we use a placeholder.
    msno: Optional[str] = "unknown_user" 
    city: int
    bd: int
    gender: str
    registered_via: int
    total_transactions: int
    total_payment: float
    total_cancel_count: int
    promo_transaction_count: int
    active_days: int
    total_secs_played: float
    total_unique_songs: int
    total_songs_played: int
    total_songs_100_percent: int
    # Optional Trend fields
    active_days_first_half: int = 0
    active_days_second_half: int = 0
    total_secs_first_half: float = 0.0
    total_secs_second_half: float = 0.0

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(input_data: ChurnInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # 2. Use the SAME processor from training (Handles Ratios, Age Bins, Cleaning)
        processed_df = processor.transform(input_df)

        # 3. Apply Label Encoders (Ensures category 1 in train is category 1 in serve)
        for col, le in encoders.items():
            if col in processed_df.columns:
                # Handle unseen labels by mapping to a default/str
                processed_df[col] = processed_df[col].astype(str)
                processed_df[col] = le.transform(processed_df[col])

        # 4. Scale numeric features
        numeric_cols = scaler.feature_names_in_
        processed_df[numeric_cols] = scaler.transform(processed_df[numeric_cols])

        # 5. Predict (Ensure column order matches training)
        feature_order = model.feature_name_
        final_df = processed_df[feature_order]
        
        probability = model.predict_proba(final_df)[:, 1][0]
        prediction = bool(probability > 0.5)

        return {
            "msno": input_data.msno,
            "churn_probability": float(round(probability, 4)),
            "prediction": "Churn" if prediction else "Stay",
            "model_version": "v2-lgbm"
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))