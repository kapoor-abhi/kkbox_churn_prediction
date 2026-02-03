import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

def inspect_model(config_path):
    # 1. Load Config and Model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = os.path.join(config['data_ingestion']['root_dir'], 'model', 'production_pipeline.pkl')
    print(f"🔍 Loading model from: {model_path}")
    
    pipeline = joblib.load(model_path)
    
    # 2. Extract the LGBMClassifier
    # Note: 'model' is the name of the step in your Pipeline definition
    lgbm_model = pipeline.named_steps['model']
    
    # 3. Get Feature Importance
    importance = lgbm_model.feature_importances_
    feature_names = lgbm_model.feature_name_
    
    # 4. Create a DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # 5. Display the "Villains"
    print("\n========= 🚨 TOP 10 FEATURES DRIVING YOUR MODEL 🚨 =========")
    print(feat_imp.head(10))
    print("============================================================\n")
    
    # 6. Check for ID Leakage
    if 'msno' in feat_imp['Feature'].values:
        print("❌ CRITICAL WARNING: 'msno' (User ID) is being used as a feature!")
        print("   This forces the model to memorize specific users instead of behavior.")
        print("   FIX: Add 'msno' to 'unused_cols' in params.yaml immediately.")

if __name__ == "__main__":
    inspect_model("config/config.yaml")