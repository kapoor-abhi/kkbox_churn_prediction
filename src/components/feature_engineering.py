import logging
import os

import dask.dataframe as dd
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EPSILON = 1e-6
MAX_SECONDS_PER_DAY = 24 * 60 * 60


def _parse_snapshot_date(config):
    cutoff_date_int = config['data_ingestion'].get('cutoff_date_int', 20170228)
    return pd.to_datetime(str(cutoff_date_int), format='%Y%m%d')


def _safe_fill_columns(df, columns, value):
    existing = [column for column in columns if column in df.columns]
    if existing:
        df[existing] = df[existing].fillna(value)
    return df

def run_feature_engineering(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = os.path.join(config['data_ingestion']['root_dir'], 'processed')
    featured_dir = os.path.join(config['data_ingestion']['root_dir'], 'featured')
    os.makedirs(featured_dir, exist_ok=True)

    snapshot_date = _parse_snapshot_date(config)

    logging.info("⏳ Loading Parquet data using Dask...")
    train_dd = dd.read_parquet(os.path.join(processed_dir, 'train.parquet'))
    members_dd = dd.read_parquet(os.path.join(processed_dir, 'members.parquet'))
    transactions_dd = dd.read_parquet(os.path.join(processed_dir, 'transactions.parquet'))
    user_logs_dd = dd.read_parquet(os.path.join(processed_dir, 'user_logs.parquet'))

    cutoff_date_int = config['data_ingestion'].get('cutoff_date_int', 20170228)
    logging.info(
        "🛡️ Applying official snapshot cutoff at %s (train_v2 March-churn observation window).",
        cutoff_date_int,
    )

    transactions_dd = transactions_dd[transactions_dd['transaction_date'] <= cutoff_date_int].copy()
    user_logs_dd = user_logs_dd[user_logs_dd['date'] <= cutoff_date_int].copy()

    logging.info("⚙️  Aggregating Transactions...")
    transactions_dd['is_promo'] = (transactions_dd['actual_amount_paid'] == 0).astype('int8')
    transactions_dd['transaction_date_dt'] = dd.to_datetime(
        transactions_dd['transaction_date'].astype(str),
        format='%Y%m%d',
        errors='coerce',
    )
    transactions_dd['membership_expire_date_dt'] = dd.to_datetime(
        transactions_dd['membership_expire_date'].astype(str),
        format='%Y%m%d',
        errors='coerce',
    )

    trans_core = transactions_dd.groupby('msno').agg({
        'payment_method_id': 'count',
        'actual_amount_paid': 'sum',
        'payment_plan_days': 'mean',
        'is_cancel': 'sum',
        'is_promo': 'sum',
        'is_auto_renew': 'sum',
        'transaction_date_dt': 'max',
        'membership_expire_date_dt': 'max',
    })
    trans_core.columns = [
        'total_transactions',
        'total_payment',
        'avg_plan_days',
        'total_cancel_count',
        'promo_transaction_count',
        'auto_renew_count',
        'last_transaction_date',
        'last_membership_expire_date',
    ]

    logging.info("⚙️  Aggregating User Logs...")
    user_logs_dd['date_dt'] = dd.to_datetime(
        user_logs_dd['date'].astype(str),
        format='%Y%m%d',
        errors='coerce',
    )
    user_logs_dd['total_secs'] = user_logs_dd['total_secs'].clip(lower=0, upper=MAX_SECONDS_PER_DAY)
    user_logs_dd['total_songs_daily'] = (
        user_logs_dd['num_25']
        + user_logs_dd['num_50']
        + user_logs_dd['num_75']
        + user_logs_dd['num_985']
        + user_logs_dd['num_100']
    )
    user_logs_dd['days_from_snapshot'] = (snapshot_date - user_logs_dd['date_dt']).dt.days

    logs_overall = user_logs_dd.groupby('msno').agg({
        'total_secs': 'sum',
        'num_unq': 'sum',
        'total_songs_daily': 'sum',
        'num_100': 'sum',
        'date_dt': 'count',
    })
    logs_overall.columns = [
        'total_secs_played',
        'total_unique_songs',
        'total_songs_played',
        'total_songs_100_percent',
        'active_days',
    ]

    logs_recent_30 = user_logs_dd[user_logs_dd['days_from_snapshot'].between(0, 29)]
    logs_prev_30 = user_logs_dd[user_logs_dd['days_from_snapshot'].between(30, 59)]

    recent_30_agg = logs_recent_30.groupby('msno').agg({'total_secs': 'sum', 'date_dt': 'count'})
    recent_30_agg.columns = ['recent_30_secs', 'recent_30_active_days']

    prev_30_agg = logs_prev_30.groupby('msno').agg({'total_secs': 'sum', 'date_dt': 'count'})
    prev_30_agg.columns = ['previous_30_secs', 'previous_30_active_days']

    logging.info("💾 Computing and Merging (Dask -> Pandas)...")
    df_train = train_dd.compute()
    df_members = members_dd.compute()
    df_trans = trans_core.compute()
    df_logs = logs_overall.compute()
    df_recent_30 = recent_30_agg.compute()
    df_prev_30 = prev_30_agg.compute()

    df_trans = df_trans.reset_index()
    df_logs = df_logs.reset_index()
    df_recent_30 = df_recent_30.reset_index()
    df_prev_30 = df_prev_30.reset_index()
    df_members = df_members.copy()

    if 'registration_init_time' in df_members.columns:
        df_members['registration_init_time'] = pd.to_datetime(
            df_members['registration_init_time'].astype(str),
            format='%Y%m%d',
            errors='coerce',
        )
        df_members['registration_age_days'] = (
            snapshot_date - df_members['registration_init_time']
        ).dt.days.clip(lower=0)

    df_trans['days_since_last_transaction'] = (snapshot_date - df_trans['last_transaction_date']).dt.days
    df_trans['days_until_membership_expire'] = (
        df_trans['last_membership_expire_date'] - snapshot_date
    ).dt.days
    df_trans['avg_payment_value'] = (
        df_trans['total_payment'] / (df_trans['total_transactions'] + EPSILON)
    )
    df_trans['cancel_rate'] = (
        df_trans['total_cancel_count'] / (df_trans['total_transactions'] + EPSILON)
    ).clip(0, 1)
    df_trans['promo_ratio'] = (
        df_trans['promo_transaction_count'] / (df_trans['total_transactions'] + EPSILON)
    ).clip(0, 1)
    df_trans['auto_renew_ratio'] = (
        df_trans['auto_renew_count'] / (df_trans['total_transactions'] + EPSILON)
    ).clip(0, 1)

    df_logs['avg_secs_played_daily'] = df_logs['total_secs_played'] / (df_logs['active_days'] + EPSILON)
    df_logs['avg_unique_songs_daily'] = df_logs['total_unique_songs'] / (df_logs['active_days'] + EPSILON)
    df_logs['completion_rate'] = (
        df_logs['total_songs_100_percent'] / (df_logs['total_songs_played'] + EPSILON)
    ).clip(0, 1)
    df_logs['uniqueness_rate'] = (
        df_logs['total_unique_songs'] / (df_logs['total_songs_played'] + EPSILON)
    ).clip(0, 1)

    logging.info("   -> Merging tables...")
    master_df = df_train.merge(df_members, on='msno', how='left')
    master_df = master_df.merge(df_trans, on='msno', how='left')
    master_df = master_df.merge(df_logs, on='msno', how='left')
    master_df = master_df.merge(df_recent_30, on='msno', how='left')
    master_df = master_df.merge(df_prev_30, on='msno', how='left')

    logging.info("🧹 Performing Final Type Cleanup...")

    fill_0_cols = [
        'total_transactions', 'total_payment', 'total_cancel_count',
        'promo_transaction_count', 'auto_renew_count', 'avg_plan_days',
        'avg_payment_value', 'cancel_rate', 'promo_ratio', 'auto_renew_ratio',
        'days_until_membership_expire',
        'total_secs_played', 'avg_secs_played_daily',
        'total_unique_songs', 'avg_unique_songs_daily', 'total_songs_played',
        'completion_rate', 'uniqueness_rate',
        'total_songs_100_percent', 'active_days',
        'recent_30_secs', 'recent_30_active_days',
        'previous_30_secs', 'previous_30_active_days',
        'registration_age_days',
    ]
    master_df = _safe_fill_columns(master_df, fill_0_cols, 0)
    master_df['days_since_last_transaction'] = master_df['days_since_last_transaction'].fillna(-1)
    master_df['has_transaction_history'] = (master_df['total_transactions'] > 0).astype(int)
    master_df['has_log_history'] = (master_df['active_days'] > 0).astype(int)

    master_df['recent_activity_ratio'] = (
        master_df['recent_30_active_days'] / (master_df['previous_30_active_days'] + EPSILON)
    ).clip(0, 30)
    master_df['recent_listening_ratio'] = (
        master_df['recent_30_secs'] / (master_df['previous_30_secs'] + EPSILON)
    ).clip(0, 30)

    int_cols = [
        'total_transactions', 'total_cancel_count', 'promo_transaction_count',
        'auto_renew_count',
        'total_unique_songs', 'total_songs_played', 'total_songs_100_percent', 'active_days',
        'recent_30_active_days', 'previous_30_active_days',
        'city', 'registered_via', 'days_since_last_transaction', 'is_churn'
    ]
    for col in int_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(-1).astype(int)

    if 'bd' in master_df.columns:
        master_df['bd'] = pd.to_numeric(master_df['bd'], errors='coerce')

    output_file = os.path.join(featured_dir, 'master_features.parquet')
    logging.info(f"✅ Saving Master Table to {output_file}...")
    master_df.to_parquet(output_file, index=False)
    logging.info(f"🎉 Done! Final Shape: {master_df.shape}")

if __name__ == "__main__":
    run_feature_engineering("config/config.yaml")
