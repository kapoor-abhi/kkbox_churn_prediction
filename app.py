import os
import joblib
import pandas as pd
import json
import logging
import datetime
from contextlib import asynccontextmanager  # <--- NEW IMPORT
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

# 1. Prometheus Instrumentator
from prometheus_fastapi_instrumentator import Instrumentator

# 2. Import Custom Processor (Crucial for joblib)
from src.components.data_processor import KKBoxFeatureEngineering

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KKBox_API")

# --- Global Variables ---
pipeline = None
TRAFFIC_LOG_FILE = "data/live_traffic.jsonl"

# --- Lifespan Manager (The Modern Replacement for on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the Startup and Shutdown logic of the API.
    """
    # --- STARTUP LOGIC ---
    
    # 1. Expose Prometheus Metrics
    Instrumentator().instrument(app).expose(app)
    logger.info("📊 Prometheus metrics exposed at /metrics")

    # 2. Load the ML Pipeline
    global pipeline
    model_path = "data/model/production_pipeline.pkl"
    try:
        logger.info(f"🚀 Loading Production Pipeline from {model_path}...")
        pipeline = joblib.load(model_path)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # We don't crash here so that /metrics keeps working, 
        # but /predict will return 503 Service Unavailable.

    # Yield control back to the application
    yield

    # --- SHUTDOWN LOGIC ---
    logger.info("🛑 Application is shutting down...")
    # (Optional: Close DB connections or flush logs here if needed)


# --- Initialize App with Lifespan ---
app = FastAPI(title="KKBox Churn Prediction API", version="3.0.0", lifespan=lifespan)


# --- Input Schema ---
class ChurnInput(BaseModel):
    # Identity (Not used for prediction, but good for logging)
    msno: str = Field(default="unknown_user", description="User ID")
    
    # Demographics
    city: int = Field(default=1, description="City ID")
    bd: int = Field(default=28, description="Age. Default is median age.")
    gender: str = Field(default="unknown", description="Gender (male/female). Use 'unknown' if missing.")
    registered_via: int = 7
    
    # Transaction Aggregates
    total_transactions: int
    total_payment: float
    total_cancel_count: int
    promo_transaction_count: int
    avg_plan_days: float
    days_since_last_transaction: int = Field(default=-1, description="-1 implies never transacted")
    
    # Usage Aggregates
    total_secs_played: float
    total_unique_songs: int
    total_songs_played: int
    total_songs_100_percent: int
    active_days: int
    
    # Trend Features
    active_days_first_half: int = Field(default=0, description="Activity count in T-30 to T-15 days")
    active_days_second_half: int = Field(default=0, description="Activity count in T-15 to T-0 days")
    total_secs_first_half: float = 0.0
    total_secs_second_half: float = 0.0


# --- Helper Functions ---
def log_traffic(input_data: dict, prediction: float, output_label: bool):
    """
    Async logging for Drift Detection.
    """
    try:
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input": input_data,
            "prediction_prob": prediction,
            "prediction_class": int(output_label)
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(TRAFFIC_LOG_FILE), exist_ok=True)

        with open(TRAFFIC_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log traffic: {e}")


# --- Endpoints ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": pipeline is not None,
        "metrics_enabled": True
    }

@app.post("/predict")
async def predict(input_data: ChurnInput, background_tasks: BackgroundTasks):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Check server logs.")

    try:
        # 1. Prepare Data
        data_dict = input_data.dict()
        # Remove metadata not used in training (msno)
        features_dict = {k: v for k, v in data_dict.items() if k != 'msno'}
        
        input_df = pd.DataFrame([features_dict])

        # 2. Predict
        # The Pipeline handles Ratios, Scaling, and Categorical Casting automatically
        probability = pipeline.predict_proba(input_df)[:, 1][0]
        prediction = bool(probability > 0.5)

        # 3. Log (Background Task)
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