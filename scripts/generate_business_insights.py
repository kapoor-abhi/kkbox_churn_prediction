import os
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_data():
    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    dataset_path = os.path.join(config['data_ingestion']['root_dir'], 'featured', 'master_features.parquet')
    df = pd.read_parquet(dataset_path)
    # Also need train.parquet for getting the transactions. Wait, master_features.parquet has everything we aggregated.
    # What about payment methods?
    # `payment_method_id` wasn't included in master_features in the original feature_engineering.py (it only did count).
    # But let's work with what we have in master_df.
    return df

def run_insights():
    df = load_data()
    
    print("="*60)
    print("KKBOX CHURN PREDICTION - BUSINESS INSIGHTS")
    print("="*60)
    print(f"\nTotal Analyzed Users: {len(df):,}")
    print(f"Overall Churn Rate: {df['is_churn'].mean():.2%}\n")
    
    print("--- 1. Ghost Customers vs Active Customers ---")
    ghost_users = df[df['total_transactions'] == 0]
    active_users = df[df['total_transactions'] > 0]
    print(f"Ghost Users (No transactions): {len(ghost_users):,} -> Churn Rate: {ghost_users['is_churn'].mean():.2%}")
    print(f"Active Users (1+ transactions): {len(active_users):,} -> Churn Rate: {active_users['is_churn'].mean():.2%}\n")
    
    print("--- 2. Promotional Sensitivity ---")
    df['promo_bin'] = pd.cut(df['promo_transaction_count'], bins=[-1, 0, 2, 999], labels=['0 Promos', '1-2 Promos', '> 2 Promos'])
    for name, group in df.groupby('promo_bin'):
        print(f"{name}: {len(group):,} users -> Churn Rate: {group['is_churn'].mean():.2%}")
    print()

    print("--- 3. Impact of Auto-Renew Behavior ---")
    df['auto_renew_flag'] = df['auto_renew_ratio'] > 0.5
    for state, group in df.groupby('auto_renew_flag'):
        label = "Majority Auto-Renew" if state else "Manual Renewal"
        print(f"{label}: {len(group):,} users -> Churn Rate: {group['is_churn'].mean():.2%}")
    print()
    
    print("--- 4. Feature Engineering Assertions ---")
    print("- 'registration_age_days': Captures loyalty via time since initialization.")
    print("- 'days_since_last_transaction': Encoded missings as -1. The recency is a highly predictive segment.")
    print("- 'avg_payment_value': Engineered out of total payment over transactions.")
    print("- 'cancel_rate': Determines historic churn volatility.\n")

    print("="*60)

if __name__ == "__main__":
    run_insights()
