import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import yaml
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load the Gold Data (Feature Engineered)
    featured_path = os.path.join(config['data_ingestion']['root_dir'], 'featured', 'master_features.parquet')
    logging.info(f"🕵️‍♀️ Validating data integrity for: {featured_path}")
    
    try:
        df = pd.read_parquet(featured_path)
    except FileNotFoundError:
        logging.error("❌ Data file not found. Run feature engineering first.")
        sys.exit(1)

    # 2. Define the Schema (The "Rules of the Road")
    schema = DataFrameSchema({
        # --- Critical Identity ---
        "msno": Column(str, nullable=False),
        
        # --- Transaction Rules ---
        # Payments cannot be negative. 
        "total_payment": Column(float, Check.greater_than_or_equal_to(0)),
        # Transactions cannot be negative.
        "total_transactions": Column(int, Check.greater_than_or_equal_to(0)),
        # Plan days usually between 0 and 400 (roughly a year + buffer)
        "avg_plan_days": Column(float, Check.in_range(0, 450)),
        
        # --- Usage Rules ---
        # Seconds played cannot be negative
        "total_secs_played": Column(float, Check.greater_than_or_equal_to(0)),
        # Active days cannot exceed the number of days in a month (roughly)
        # Since we aggregated over history, let's say max 3000 days, but definitely > 0
        "active_days": Column(int, Check.greater_than_or_equal_to(0)),
        
        # --- Logic Checks ---
        # Trend data cannot be negative
        "active_days_first_half": Column(int, Check.greater_than_or_equal_to(0)),
        
        # --- Target ---
        # Churn is binary (0 or 1)
        "is_churn": Column(int, Check.isin([0, 1]))
    })

    try:
        # 3. Run Validation
        schema.validate(df, lazy=True) # lazy=True reports ALL errors, not just the first one
        logging.info("✅ Data Validation Passed! The dataset is safe for training.")
        
    except pa.errors.SchemaErrors as err:
        logging.error("❌ Data Validation FAILED!")
        logging.error(f"Failure Cases:\n{err.failure_cases}")
        # Stop the pipeline! Do not train on garbage.
        sys.exit(1)

if __name__ == "__main__":
    validate_data("config/config.yaml")