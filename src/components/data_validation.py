import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
import yaml
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

# We check for types and existence, but we will handle the "crazy" values 
# in the Feature Engineering/Processor stage.
members_schema = DataFrameSchema({
    "msno": Column(str),
    "city": Column(int, nullable=True),
    "bd": Column(int, nullable=True), 
    "gender": Column(str, nullable=True),
    "registered_via": Column(int, nullable=True),
})

def validate_all_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['data_ingestion']['raw_data_dir']
    
    try:
        logging.info("Validating Members data types...")
        members_df = pd.read_csv(os.path.join(raw_dir, config['data_ingestion']['members_data_file']))
        members_schema.validate(members_df)
        
        logging.info("✅ Data Validation Passed (Types are correct).")
    except Exception as e:
        logging.error(f"❌ Data Validation Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_all_data("config/config.yaml")