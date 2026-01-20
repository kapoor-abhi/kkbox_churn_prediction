import os
import yaml
import pandas as pd
import dask.dataframe as dd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['data_ingestion']['raw_data_dir']
    processed_dir = os.path.join(config['data_ingestion']['root_dir'], 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Dictionary of files to process
    files = {
        'train': config['data_ingestion']['train_data_file'],
        'members': config['data_ingestion']['members_data_file'],
        'transactions': config['data_ingestion']['transactions_data_file'],
        'user_logs': config['data_ingestion']['user_logs_data_file'] # Added this back
    }

    for key, filename in files.items():
        logging.info(f"Processing {filename}...")
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, f"{key}.parquet")

        if key == 'user_logs':
            # Use Dask for the large user_logs file
            logging.info("Using Dask for heavy user_logs file...")
            ddf = dd.read_csv(input_path)
            # Standardize columns
            ddf.columns = [c.lower().strip() for c in ddf.columns]
            ddf.to_parquet(output_path, engine='pyarrow')
        else:
            # Use Pandas for smaller files
            df = pd.read_csv(input_path)
            df.columns = [c.lower().strip() for c in df.columns]
            df.to_parquet(output_path, index=False)
            
        logging.info(f"Successfully saved {key} to {output_path}")

if __name__ == "__main__":
    ingest_data("config/config.yaml")