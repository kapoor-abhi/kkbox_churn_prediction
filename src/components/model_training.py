import os
import yaml
import pandas as pd
import numpy as np
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import os
# --- ADD THESE LINES ---
from dotenv import load_dotenv
load_dotenv() # Load secrets from .env file
# -----------------------

import yaml
# ... rest of your imports ...

# Import our custom processor (The logic we just fixed)
from src.components.data_processor import KKBoxFeatureEngineering

# Connect to Local MLflow (running in Docker)
mlflow.set_tracking_uri("http://localhost:5001")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(config_path, params_path):
    # 1. Load Configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # 2. Load Gold Data (The result of Dask Feature Engineering)
    # This file has approx 22 columns (Aggregations only)
    featured_path = os.path.join(config['data_ingestion']['root_dir'], 'featured', 'master_features.parquet')
    logging.info(f"⏳ Loading data from {featured_path}...")
    
    try:
        df = pd.read_parquet(featured_path)
    except FileNotFoundError:
        logging.error("❌ Data not found! Did you run 'src/components/feature_engineering.py'?")
        return

    # 3. Prepare X and y
    target = params['training']['target_col'] # 'is_churn'
    unused_cols = params['training']['unused_cols'] # ['msno', 'bd', etc] if specified
    
    # Drop target and strictly unused columns (like IDs)
    # We keep 'bd' and others if they are needed by the Processor
    X = df.drop(columns=[target] + unused_cols, errors='ignore')
    y = df[target]

    logging.info(f"📊 Input Feature Shape (Before Processing): {X.shape}")

    # 4. Define the Production Pipeline
    # Step A: KKBoxFeatureEngineering -> Imputes missing values, creates Ratios, casts to Category
    # Step B: LGBMClassifier -> Trains the model
    # Note: We do NOT use StandardScaler because LightGBM handles unscaled data well, 
    # and it simplifies the pipeline for categorical features.
    
    pipeline = Pipeline([
        ('feature_eng', KKBoxFeatureEngineering()),
        ('model', LGBMClassifier(**params['lgbm_params']))
    ])

    # 5. Start MLflow Experiment
    experiment_name = "KKBox_Churn_Production"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        logging.info(f"🚀 Started MLflow Run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params(params['lgbm_params'])
        mlflow.log_param("input_rows", X.shape[0])

        # --- Phase 1: Cross-Validation (Metric Evaluation) ---
        logging.info("⚔️  Starting Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=params['training']['n_splits'], 
                              shuffle=True, 
                              random_state=params['training']['random_state'])
        
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Clone pipeline to ensure a fresh start for each fold
            from sklearn.base import clone
            fold_pipe = clone(pipeline)
            
            # Fit
            fold_pipe.fit(X_train, y_train)
            
            # Verification (Fold 0 only): Check how many features the model actually sees
            if fold == 0:
                # Access the model step inside the pipeline
                model_step = fold_pipe.named_steps['model']
                n_features = model_step.n_features_in_
                feature_names = model_step.feature_name_
                logging.info(f"✅ Verified: Model trained on {n_features} features (Expanded from {X.shape[1]}).")
                logging.info(f"   Sample Features: {feature_names[:5]}...")

            # Evaluate
            preds = fold_pipe.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            auc_scores.append(score)
            
            logging.info(f"   Fold {fold+1} AUC: {score:.4f}")
            mlflow.log_metric(f"fold_{fold+1}_auc", score)

        mean_auc = np.mean(auc_scores)
        logging.info(f"🏆 Mean CV AUC: {mean_auc:.4f}")
        mlflow.log_metric("mean_auc", mean_auc)

        # --- Phase 2: Production Retraining (Refit) ---
        logging.info("🔄 Retraining Pipeline on FULL dataset for Production...")
        
        # This learns the global Median Age and trains on 100% of data
        pipeline.fit(X, y)
        
        # --- Phase 3: Save Artifacts ---
        model_dir = os.path.join(config['data_ingestion']['root_dir'], 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Local File (For DVC)
        local_model_path = os.path.join(model_dir, 'production_pipeline.pkl')
        joblib.dump(pipeline, local_model_path)
        logging.info(f"💾 Saved local pipeline to: {local_model_path}")
        
        # Save to MLflow (MinIO) - This is the "Registry" version
        logging.info("☁️  Uploading to MLflow Artifact Store (MinIO)...")
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model",
            registered_model_name="KKBox_Churn_Pipeline",
            input_example=X.head(1) # Helps MLflow understand schema
        )
        
        logging.info("✨ Training Complete. Pipeline is ready for Deployment.")

if __name__ == "__main__":
    train_model("config/config.yaml", "config/params.yaml")