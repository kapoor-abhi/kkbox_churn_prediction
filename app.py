import os
import joblib
import pandas as pd
import json
import logging
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

# 1. Prometheus Instrumentator
from prometheus_fastapi_instrumentator import Instrumentator

# 2. Import Custom Processor
from src.components.data_processor import KKBoxFeatureEngineering

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KKBox_API")

# --- Global Variables ---
pipeline = None
TRAFFIC_LOG_FILE = "data/live_traffic.jsonl"

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles Startup and Shutdown logic.
    """
    # --- STARTUP LOGIC ---
    # Load the ML Pipeline
    global pipeline
    model_path = "data/model/production_pipeline.pkl"
    try:
        logger.info(f"🚀 Loading Production Pipeline from {model_path}...")
        pipeline = joblib.load(model_path)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

    yield  # Application runs here...

    # --- SHUTDOWN LOGIC ---
    logger.info("🛑 Application is shutting down...")

# --- Initialize App with Lifespan ---
app = FastAPI(title="KKBox Churn Prediction API", version="3.0.0", lifespan=lifespan)

# --- CRITICAL FIX: Instrumentator must be in Global Scope ---
# This adds the middleware immediately, before the app "starts"
Instrumentator().instrument(app).expose(app)
logger.info("📊 Prometheus metrics exposed at /metrics")


# --- Input Schema ---
class ChurnInput(BaseModel):
    msno: str = Field(default="unknown_user", description="User ID")
    city: int = Field(default=1, description="City ID")
    bd: int = Field(default=28, description="Age")
    gender: str = Field(default="unknown", description="Gender")
    registered_via: int = 7
    total_transactions: int
    total_payment: float
    total_cancel_count: int
    promo_transaction_count: int
    avg_plan_days: float
    days_since_last_transaction: int = Field(default=-1)
    total_secs_played: float
    total_unique_songs: int
    total_songs_played: int
    total_songs_100_percent: int
    active_days: int
    active_days_first_half: int = 0
    active_days_second_half: int = 0
    total_secs_first_half: float = 0.0
    total_secs_second_half: float = 0.0

# --- Helper Functions ---
def log_traffic(input_data: dict, prediction: float, output_label: bool):
    try:
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input": input_data,
            "prediction_prob": prediction,
            "prediction_class": int(output_label)
        }
        os.makedirs(os.path.dirname(TRAFFIC_LOG_FILE), exist_ok=True)
        with open(TRAFFIC_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log traffic: {e}")

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict")
async def predict(input_data: ChurnInput, background_tasks: BackgroundTasks):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        data_dict = input_data.dict()
        features_dict = {k: v for k, v in data_dict.items() if k != 'msno'}
        input_df = pd.DataFrame([features_dict])

        probability = pipeline.predict_proba(input_df)[:, 1][0]
        prediction = bool(probability > 0.5)

        background_tasks.add_task(log_traffic, data_dict, float(probability), prediction)

        return {
            "msno": input_data.msno,
            "churn_probability": round(float(probability), 4),
            "prediction": "Churn" if prediction else "Stay",
            "model_version": "v3-pipeline"
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))