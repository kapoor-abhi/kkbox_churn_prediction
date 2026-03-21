import os
import yaml
import numpy as np
import pandas as pd
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from dotenv import load_dotenv

load_dotenv()

from src.components.data_processor import KKBoxFeatureEngineering

mlflow.set_tracking_uri("file:./mlruns")

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
    target = params['training']['target_col']
    unused_cols = params['training']['unused_cols']
    X = df.drop(columns=[target] + unused_cols, errors='ignore')
    y = df[target]

    logging.info(f"📊 Input Feature Shape (Before Processing): {X.shape}")

    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    lgbm_params = params['lgbm_params'].copy()
    lgbm_params.setdefault('scale_pos_weight', scale_pos_weight)
    lgbm_params.setdefault('random_state', params['training']['random_state'])

    pipeline = Pipeline([
        ('feature_eng', KKBoxFeatureEngineering()),
        ('model', LGBMClassifier(**lgbm_params))
    ])

    experiment_name = "KKBox_Churn_Production"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        logging.info(f"🚀 Started MLflow Run ID: {run.info.run_id}")
        
        mlflow.log_params(lgbm_params)
        mlflow.log_param("input_rows", X.shape[0])
        mlflow.log_metric("train_positive_rate", float(y.mean()))

        logging.info("⚔️  Starting Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=params['training']['n_splits'], 
                              shuffle=True, 
                              random_state=params['training']['random_state'])
        
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            from sklearn.base import clone
            fold_pipe = clone(pipeline)
            fold_pipe.fit(X_train, y_train)
            
            if fold == 0:
                model_step = fold_pipe.named_steps['model']
                n_features = model_step.n_features_in_
                feature_names = model_step.feature_name_
                logging.info(f"✅ Verified: Model trained on {n_features} features (Expanded from {X.shape[1]}).")
                logging.info(f"   Sample Features: {feature_names[:5]}...")

            preds = fold_pipe.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            auc_scores.append(score)
            
            logging.info(f"   Fold {fold+1} AUC: {score:.4f}")
            mlflow.log_metric(f"fold_{fold+1}_auc", score)

        mean_auc = np.mean(auc_scores)
        logging.info(f"🏆 Mean CV AUC: {mean_auc:.4f}")
        mlflow.log_metric("mean_auc", mean_auc)

        logging.info("🔄 Retraining Pipeline on FULL dataset for Production...")
        pipeline.fit(X, y)
        
        model_dir = os.path.join(config['data_ingestion']['root_dir'], 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        local_model_path = os.path.join(model_dir, 'production_pipeline.pkl')
        joblib.dump(pipeline, local_model_path)
        logging.info(f"💾 Saved local pipeline to: {local_model_path}")
        
        logging.info("☁️  Uploading to MLflow Artifact Store (MinIO)...")
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model",
            registered_model_name="KKBox_Churn_Pipeline",
            input_example=X.head(1)
        )
        
        logging.info("✨ Training Complete. Pipeline is ready for Deployment.")

if __name__ == "__main__":
    train_model("config/config.yaml", "config/params.yaml")
