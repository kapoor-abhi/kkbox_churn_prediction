import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import logging

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Configuration ---
ARTIFACTS_DIR = "data/model"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler_fold_1.pkl")

# --- 3. Global Artifacts Storage ---
artifacts = {}

# --- 4. FastAPI App Initialization ---
app = FastAPI(
    title="KKBox Churn Prediction API (Simplified)",
    description="An API that takes minimal user inputs, performs feature engineering, and predicts churn.",
    version="2.0.0"
)

# --- 5. Lifespan Events (Startup/Shutdown) ---
@app.on_event("startup")
async def load_artifacts():
    """Loads model and artifacts on server startup."""
    logging.info("Loading model and artifacts...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        artifacts['model'] = model
        artifacts['scaler'] = scaler
        artifacts['model_columns'] = model.feature_name_
        artifacts['numeric_columns_to_scale'] = scaler.get_feature_names_out()
        logging.info("Artifacts loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error loading artifacts: {e}. Ensure pipeline has been run.")
        artifacts['model'] = None

# --- 6. SIMPLIFIED Pydantic Input Model ---
# This model only asks for the raw, fundamental features.
class UserInput(BaseModel):
    # Member features
    city: int = Field(..., example=1, description="City of the user.")
    bd: int = Field(..., example=30, description="Age of the user.")
    gender: str = Field(..., example="male", description="Gender ('male', 'female', or 'unknown').")
    registered_via: int = Field(..., example=7, description="Registration method ID.")
    
    # Base Transaction features
    total_transactions: int = Field(..., example=12, gt=0) # Greater than 0 to avoid division by zero
    total_payment: float = Field(..., example=1788.0)
    avg_plan_days: int = Field(..., example=30)
    total_cancel_count: int = Field(..., example=0)
    promo_transaction_count: int = Field(..., example=1)
    days_since_last_transaction: int = Field(..., example=20)

    # Base User log features
    total_secs_played: float = Field(..., example=250000.0)
    total_unique_songs: int = Field(..., example=800)
    total_songs_played: int = Field(..., example=1000, gt=0) # Greater than 0
    total_songs_100_percent: int = Field(..., example=850)
    active_days: int = Field(..., example=28)
    days_since_last_listen: int = Field(..., example=3)

# --- 7. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running."}

@app.post("/predict")
async def predict_churn(user_input: UserInput):
    """
    Predicts customer churn from raw features.
    This endpoint performs all necessary feature engineering internally.
    """
    if not artifacts.get('model'):
        raise HTTPException(status_code=503, detail="Model is not available.")

    logging.info("Received prediction request. Starting feature engineering...")
    
    # --- Internal Feature Engineering Pipeline ---
    # 1. Convert Pydantic model to a pandas DataFrame
    input_df = pd.DataFrame([user_input.dict()])
    epsilon = 1e-6 # To prevent division by zero

    # 2. Create Derived Transaction Features
    input_df['avg_payment_value'] = input_df['total_payment'] / (input_df['total_transactions'] + epsilon)
    input_df['cancel_rate'] = input_df['total_cancel_count'] / (input_df['total_transactions'] + epsilon)
    input_df['promo_ratio'] = input_df['promo_transaction_count'] / (input_df['total_transactions'] + epsilon)

    # 3. Create Derived User Log Features
    input_df['avg_secs_played_daily'] = input_df['total_secs_played'] / (input_df['active_days'] + epsilon)
    input_df['avg_unique_songs_daily'] = input_df['total_unique_songs'] / (input_df['active_days'] + epsilon)
    input_df['completion_rate'] = input_df['total_songs_100_percent'] / (input_df['total_songs_played'] + epsilon)
    input_df['uniqueness_rate'] = input_df['total_unique_songs'] / (input_df['total_songs_played'] + epsilon)

    # 4. Create Advanced/Defaulted Features
    input_df['activity_trend_abs'] = 0.0  # Default value
    input_df['secs_trend_ratio'] = 1.0    # Default value (no change)

    # 5. Create Categorical Features for Encoding
    age_bins = [0, 18, 25, 35, 50, 80]
    age_labels = ['0-18', '19-25', '26-35', '36-50', '51-80']
    input_df['age_group'] = pd.cut(input_df['bd'], bins=age_bins, labels=age_labels, right=False)
    input_df['age_group'] = input_df['age_group'].cat.add_categories('Unknown').fillna('Unknown')
    input_df['gender'] = input_df['gender'].str.lower().fillna('unknown')
    
    # 6. Perform One-Hot Encoding
    categorical_features = ['gender', 'city', 'registered_via', 'age_group']
    input_df = pd.get_dummies(input_df, columns=categorical_features, dummy_na=False)

    # 7. Align columns with the model's training data (CRITICAL STEP)
    model_columns = artifacts['model_columns']
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 8. Apply the StandardScaler
    cols_to_scale = [col for col in artifacts['numeric_columns_to_scale'] if col in input_df.columns]
    if cols_to_scale:
        input_df[cols_to_scale] = artifacts['scaler'].transform(input_df[cols_to_scale])
    
    logging.info("Feature engineering complete. Making prediction.")

    # --- Prediction ---
    try:
        prediction_proba = artifacts['model'].predict_proba(input_df)[:, 1]
        churn_probability = float(prediction_proba[0])
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Could not make a prediction.")

    # --- Return Response ---
    return {
        "churn_probability": round(churn_probability, 4),
        "will_churn": bool(churn_probability > 0.5),
        "model_version": "2.0"
    }

# --- 8. Main execution block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)