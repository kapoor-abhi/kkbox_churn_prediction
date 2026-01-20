import os
import argparse
import yaml
import pandas as pd
import numpy as np
import dask.dataframe as dd
import logging
import joblib
from data_processor import KKBoxFeatureEngineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_feature_engineering(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = os.path.join(config['data_ingestion']['root_dir'], 'processed')
    featured_dir = os.path.join(config['data_ingestion']['root_dir'], 'featured')
    os.makedirs(featured_dir, exist_ok=True)

    # 1. Load Data
    logging.info("Loading processed parquet files...")
    train_df = pd.read_parquet(os.path.join(processed_dir, 'train.parquet'))
    members_df = pd.read_parquet(os.path.join(processed_dir, 'members.parquet'))
    transactions_df = pd.read_parquet(os.path.join(processed_dir, 'transactions.parquet'))
    user_logs_dd = dd.read_parquet(os.path.join(processed_dir, 'user_logs.parquet'))

    # 2. Transaction Engineering
    logging.info("Engineering transaction features...")
    snapshot_date = pd.to_datetime(transactions_df['transaction_date']).max() + pd.Timedelta(days=1)
    promo_counts = transactions_df[transactions_df['actual_amount_paid'] == 0].groupby('msno').size().reset_index(name='promo_transaction_count')
    
    trans_features = transactions_df.groupby('msno').agg(
        total_transactions=('msno', 'count'),
        total_payment=('actual_amount_paid', 'sum'),
        avg_plan_days=('payment_plan_days', 'mean'),
        total_cancel_count=('is_cancel', 'sum'),
        last_transaction_date=('transaction_date', 'max')
    ).reset_index()
    trans_features = pd.merge(trans_features, promo_counts, on='msno', how='left').fillna(0)
    trans_features['days_since_last_transaction'] = (snapshot_date - pd.to_datetime(trans_features['last_transaction_date'])).dt.days
    trans_features.drop(columns=['last_transaction_date'], inplace=True)

    # 3. User Logs (Dask)
    logging.info("Engineering usage features with Dask...")
    user_logs_dd['total_songs_daily'] = user_logs_dd['num_25'] + user_logs_dd['num_50'] + user_logs_dd['num_75'] + user_logs_dd['num_985'] + user_logs_dd['num_100']
    
    usage_agg = user_logs_dd.groupby('msno').agg({
        'total_secs': ['sum'],
        'num_unq': ['sum'],
        'total_songs_daily': ['sum'],
        'num_100': ['sum'],
        'date': ['count']
    }).compute()
    usage_agg.columns = ['total_secs_played', 'total_unique_songs', 'total_songs_played', 'total_songs_100_percent', 'active_days']
    usage_agg = usage_agg.reset_index()

    # 4. Trends (Advanced Logic from original)
    logging.info("Calculating activity trends...")
    user_logs_dd['date'] = dd.to_datetime(user_logs_dd['date'])
    mid_month = user_logs_dd['date'].max().compute() - pd.Timedelta(days=15)
    
    first_half = user_logs_dd[user_logs_dd['date'] < mid_month].groupby('msno').agg(
        active_days_first_half=('date', 'count'),
        total_secs_first_half=('total_secs', 'sum')
    ).compute().reset_index()
    
    second_half = user_logs_dd[user_logs_dd['date'] >= mid_month].groupby('msno').agg(
        active_days_second_half=('date', 'count'),
        total_secs_second_half=('total_secs', 'sum')
    ).compute().reset_index()

    # 5. Master Merge & Unified Processor
    logging.info("Final merging and applying processor...")
    df = train_df.merge(members_df, on='msno', how='left')
    df = df.merge(trans_features, on='msno', how='left')
    df = df.merge(usage_agg, on='msno', how='left')
    df = df.merge(first_half, on='msno', how='left')
    df = df.merge(second_half, on='msno', how='left')

    processor = KKBoxFeatureEngineering()
    processor.fit(df)
    df_featured = processor.transform(df)

    # Final cleanup
    num_cols = df_featured.select_dtypes(include=[np.number]).columns
    df_featured[num_cols] = df_featured[num_cols].fillna(0)

    # 6. Save
    output_path = os.path.join(featured_dir, 'master_features.parquet')
    df_featured.to_parquet(output_path, index=False)
    joblib.dump(processor, os.path.join(config['data_ingestion']['root_dir'], 'model', 'feature_processor.pkl'))
    logging.info(f"Complete features saved. Shape: {df_featured.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    run_feature_engineering(args.config)