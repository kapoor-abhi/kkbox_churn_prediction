import os
import yaml
import dask.dataframe as dd
import logging
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Setup MinIO Connection Options
    storage_options = {
        "key": os.getenv("MINIO_ROOT_USER", "admin"),
        "secret": os.getenv("MINIO_ROOT_PASSWORD", "password123"),
        "client_kwargs": {
            "endpoint_url": "http://localhost:9000"
        }
    }

    raw_bucket = config['data_ingestion']['raw_data_dir']
    processed_dir = os.path.join(config['data_ingestion']['root_dir'], 'processed')
    
    # Dask writes folders by default, so we clean the specific output first if it exists
    os.makedirs(processed_dir, exist_ok=True)

    files = {
        'train': config['data_ingestion']['train_data_file'],
        'members': config['data_ingestion']['members_data_file'],
        'transactions': config['data_ingestion']['transactions_data_file'],
        'user_logs': config['data_ingestion']['user_logs_data_file']
    }

    for key, filename in files.items():
        input_path = f"{raw_bucket}/{filename}"
        output_path = os.path.join(processed_dir, f"{key}.parquet")

        logging.info(f"🚀 Streaming {key} from Data Lake ({input_path})...")

        try:
            # 2. Read with Dask (Lazy evaluation - doesn't load RAM yet)
            # blocksize="64MB" splits large files into manageable chunks
            ddf = dd.read_csv(input_path, storage_options=storage_options, blocksize="64MB", dtype={'gender': 'object', 'msno': 'object'})
            
            # Standardize columns
            ddf.columns = [c.lower().strip() for c in ddf.columns]

            # 3. Write to Parquet (Compute happens here)
            # Dask will create a folder (e.g., train.parquet/) containing partition files.
            # This is the standard "Big Data" format.
            if os.path.exists(output_path):
                shutil.rmtree(output_path) # Clean old run
                
            ddf.to_parquet(
                output_path, 
                engine='pyarrow', 
                compression='zstd', 
                write_index=False
            )
            
            logging.info(f"✅ Successfully ingested {key} to {output_path}")

        except Exception as e:
            logging.error(f"❌ Failed to ingest {key}: {e}")
            raise e

if __name__ == "__main__":
    ingest_data("config/config.yaml")