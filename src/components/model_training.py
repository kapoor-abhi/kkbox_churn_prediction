import os
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
import joblib
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(config_path, params_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # 1. Load Featured Data
    featured_path = os.path.join(config['data_ingestion']['root_dir'], 'featured', 'master_features.parquet')
    df = pd.read_parquet(featured_path)
    
    # 2. Prepare Data
    target = params['training']['target_col']
    features = [c for c in df.columns if c not in params['training']['unused_cols']]
    
    X = df[features]
    y = df[target]

    # Handle Categorical Columns (Label Encoding is better for LGBM than OHE)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # 3. MLflow Tracking
    mlflow.set_experiment("KKBox_Churn_Production")
    
    with mlflow.start_run():
        logging.info("MLflow Tracking Started.")
        mlflow.log_params(params['lgbm_params'])
        mlflow.log_param("n_splits", params['training']['n_splits'])

        skf = StratifiedKFold(n_splits=params['training']['n_splits'], shuffle=True, random_state=params['training']['random_state'])
        
        fold_auc_scores = []
        best_iteration_list = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logging.info(f"Training Fold {fold+1}...")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Scale numeric features (Prevent Leakage)
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            # Remove categorical columns from numeric list if they were encoded
            numeric_cols = [c for c in numeric_cols if c not in cat_cols]
            
            scaler = StandardScaler()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

            # Calculate Scale Weight for imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            model = lgb.LGBMClassifier(**params['lgbm_params'], scale_pos_weight=scale_pos_weight)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50)]
            )

            preds = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, preds)
            fold_auc_scores.append(auc)
            best_iteration_list.append(model.best_iteration_)
            
            mlflow.log_metric(f"fold_{fold+1}_auc", auc)
            
            # Save the scaler and encoder from fold 1 as representative (Industry practice)
            if fold == 0:
                model_dir = os.path.join(config['data_ingestion']['root_dir'], 'model')
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
                joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))

        # 4. Log Final Results
        mean_auc = np.mean(fold_auc_scores)
        logging.info(f"Mean CV AUC: {mean_auc:.5f}")
        mlflow.log_metric("mean_auc", mean_auc)

        # Log the last model to MLflow registry
        mlflow.lightgbm.log_model(model, "model", registered_model_name="Churn_LGBM_Model")
        
        # Save model locally for DVC
        joblib.dump(model, os.path.join(model_dir, 'lgbm_model.pkl'))
        logging.info("Training complete. Artifacts saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--params', type=str, required=True)
    args = parser.parse_args()
    train_model(args.config, args.params)