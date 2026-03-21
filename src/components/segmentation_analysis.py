import argparse
import json
import logging
import os

import joblib
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.components.data_processor import KKBoxFeatureEngineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _safe_qcut(series, labels):
    ranked = series.rank(method='first')
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels, duplicates='drop').astype(str)
    except ValueError:
        return pd.cut(ranked, bins=len(labels), labels=labels, include_lowest=True).astype(str)


def load_featured_dataset(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    featured_path = os.path.join(
        config['data_ingestion']['root_dir'],
        'featured',
        'master_features.parquet',
    )
    analysis_dir = config['data_ingestion'].get('analysis_dir', os.path.join(config['data_ingestion']['root_dir'], 'analysis'))
    os.makedirs(analysis_dir, exist_ok=True)

    logging.info("Loading featured dataset from %s", featured_path)
    df = pd.read_parquet(featured_path)
    return config, analysis_dir, df


def prepare_analysis_frame(master_df):
    processor = KKBoxFeatureEngineering()
    feature_input = master_df.drop(columns=['is_churn'], errors='ignore')
    processor.fit(feature_input)
    enriched = processor.transform(feature_input)
    if 'msno' in master_df.columns:
        enriched['msno'] = master_df['msno'].values
    if 'is_churn' in master_df.columns:
        enriched['is_churn'] = master_df['is_churn'].values
    return enriched


def build_rfm_segments(df):
    rfm_df = df[['msno', 'is_churn', 'days_since_last_transaction', 'total_transactions', 'total_payment']].copy()
    rfm_df['is_ghost_user'] = (rfm_df['total_transactions'] <= 0).astype(int)

    valid_mask = rfm_df['is_ghost_user'] == 0
    rfm_df.loc[valid_mask, 'R_Score'] = _safe_qcut(
        rfm_df.loc[valid_mask, 'days_since_last_transaction'].astype(float),
        [5, 4, 3, 2, 1],
    )
    rfm_df.loc[valid_mask, 'F_Score'] = _safe_qcut(
        rfm_df.loc[valid_mask, 'total_transactions'].astype(float),
        [1, 2, 3, 4, 5],
    )
    rfm_df.loc[valid_mask, 'M_Score'] = _safe_qcut(
        rfm_df.loc[valid_mask, 'total_payment'].astype(float),
        [1, 2, 3, 4, 5],
    )

    rfm_df[['R_Score', 'F_Score', 'M_Score']] = rfm_df[['R_Score', 'F_Score', 'M_Score']].fillna('0')
    rfm_df['RFM_Vector'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']

    segment_map = {
        r'555|554|545|544': 'Champions',
        r'5[3-5][1-5]|4[4-5][1-5]': 'Loyal / Engaged',
        r'[4-5][1-3][1-5]': 'Promising',
        r'3[4-5][1-5]': 'Potential Loyalists',
        r'3[1-3][1-5]': 'Needs Attention',
        r'[1-2][4-5][4-5]': 'High-Value Churn Risk',
        r'[1-2][4-5][1-3]': 'Discount Sensitive',
        r'[1-2][1-3][4-5]': 'High-Value Sleepers',
        r'[1-2][1-3][1-3]': 'Lost / Hibernating',
        r'000': 'No History',
    }

    rfm_df['RFM_Segment'] = rfm_df['RFM_Vector']
    for pattern, segment_name in segment_map.items():
        rfm_df['RFM_Segment'] = rfm_df['RFM_Segment'].replace(pattern, segment_name, regex=True)
    rfm_df.loc[rfm_df['is_ghost_user'] == 1, 'RFM_Segment'] = 'No History'
    rfm_df.loc[rfm_df['RFM_Segment'] == rfm_df['RFM_Vector'], 'RFM_Segment'] = 'Others'

    segment_profile = (
        rfm_df.groupby('RFM_Segment')
        .agg(
            user_count=('msno', 'count'),
            churn_rate=('is_churn', 'mean'),
            avg_recency_days=('days_since_last_transaction', 'mean'),
            avg_transactions=('total_transactions', 'mean'),
            avg_payment=('total_payment', 'mean'),
        )
        .sort_values(['churn_rate', 'user_count'], ascending=[False, False])
        .reset_index()
    )
    return rfm_df, segment_profile


def build_cluster_views(df, n_clusters=5, dbscan_sample_size=15000):
    cluster_features = [
        'days_since_last_transaction',
        'days_until_membership_expire',
        'total_transactions',
        'total_payment',
        'avg_payment_value',
        'avg_plan_days',
        'cancel_rate',
        'promo_ratio',
        'auto_renew_ratio',
        'total_secs_played',
        'avg_secs_played_daily',
        'total_unique_songs',
        'avg_unique_songs_daily',
        'completion_rate',
        'uniqueness_rate',
        'active_days',
        'recent_30_secs',
        'recent_30_active_days',
        'recent_listening_ratio',
        'recent_activity_ratio',
        'registration_age_days',
    ]

    available_features = [feature for feature in cluster_features if feature in df.columns]
    cluster_df = df[['msno', 'is_churn'] + available_features].copy()
    cluster_df[available_features] = cluster_df[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df[available_features])

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_features)
    cluster_df['PCA_1'] = components[:, 0]
    cluster_df['PCA_2'] = components[:, 1]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_df['kmeans_cluster'] = kmeans.fit_predict(scaled_features)

    kmeans_profile = (
        cluster_df.groupby('kmeans_cluster')
        .agg(
            user_count=('msno', 'count'),
            churn_rate=('is_churn', 'mean'),
            avg_recency_days=('days_since_last_transaction', 'mean'),
            avg_total_payment=('total_payment', 'mean'),
            avg_total_secs=('total_secs_played', 'mean'),
        )
        .sort_values('churn_rate', ascending=False)
        .reset_index()
    )

    sample_size = min(dbscan_sample_size, len(cluster_df))
    sampled_df = cluster_df.sample(n=sample_size, random_state=42).copy()
    dbscan = DBSCAN(eps=0.8, min_samples=25)
    sampled_df['dbscan_cluster'] = dbscan.fit_predict(sampled_df[['PCA_1', 'PCA_2']].values)

    dbscan_profile = (
        sampled_df.groupby('dbscan_cluster')
        .agg(
            user_count=('msno', 'count'),
            churn_rate=('is_churn', 'mean'),
            avg_recency_days=('days_since_last_transaction', 'mean'),
            avg_total_payment=('total_payment', 'mean'),
        )
        .sort_values('churn_rate', ascending=False)
        .reset_index()
    )

    cluster_summary = {
        'cluster_features': available_features,
        'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'dbscan_sample_size': int(sample_size),
        'dbscan_noise_points': int((sampled_df['dbscan_cluster'] == -1).sum()),
    }

    bundle = {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'dbscan': dbscan,
        'cluster_features': available_features,
    }
    return cluster_df, sampled_df, kmeans_profile, dbscan_profile, cluster_summary, bundle


def run_segmentation_analysis(config_path, n_clusters=5, dbscan_sample_size=15000):
    config, analysis_dir, master_df = load_featured_dataset(config_path)
    enriched_df = prepare_analysis_frame(master_df)

    logging.info("Running RFM segmentation...")
    rfm_df, rfm_profile = build_rfm_segments(enriched_df)

    logging.info("Running PCA + clustering...")
    cluster_df, dbscan_sample_df, kmeans_profile, dbscan_profile, cluster_summary, cluster_bundle = build_cluster_views(
        enriched_df,
        n_clusters=n_clusters,
        dbscan_sample_size=dbscan_sample_size,
    )

    analysis_df = enriched_df.merge(
        rfm_df[['msno', 'RFM_Vector', 'RFM_Segment']],
        on='msno',
        how='left',
    ).merge(
        cluster_df[['msno', 'PCA_1', 'PCA_2', 'kmeans_cluster']],
        on='msno',
        how='left',
    )

    output_files = {
        'analysis_dataset': os.path.join(analysis_dir, 'analysis_dataset.parquet'),
        'rfm_profile': os.path.join(analysis_dir, 'rfm_profile.csv'),
        'kmeans_profile': os.path.join(analysis_dir, 'kmeans_profile.csv'),
        'dbscan_profile': os.path.join(analysis_dir, 'dbscan_profile.csv'),
        'dbscan_sample': os.path.join(analysis_dir, 'dbscan_sample.parquet'),
        'cluster_bundle': os.path.join(analysis_dir, 'cluster_bundle.joblib'),
        'summary': os.path.join(analysis_dir, 'analysis_summary.json'),
    }

    analysis_df.to_parquet(output_files['analysis_dataset'], index=False)
    rfm_profile.to_csv(output_files['rfm_profile'], index=False)
    kmeans_profile.to_csv(output_files['kmeans_profile'], index=False)
    dbscan_profile.to_csv(output_files['dbscan_profile'], index=False)
    dbscan_sample_df.to_parquet(output_files['dbscan_sample'], index=False)
    joblib.dump(cluster_bundle, output_files['cluster_bundle'])

    summary = {
        'rows': int(len(analysis_df)),
        'churn_rate': float(analysis_df['is_churn'].mean()),
        'rfm_segments': int(rfm_profile['RFM_Segment'].nunique()),
        'kmeans_clusters': int(kmeans_profile['kmeans_cluster'].nunique()),
        **cluster_summary,
    }
    with open(output_files['summary'], 'w') as file:
        json.dump(summary, file, indent=2)

    logging.info("Saved analysis outputs to %s", analysis_dir)
    return output_files, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KKBOX local EDA, RFM segmentation, and clustering analysis.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the project config file.')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of KMeans clusters.')
    parser.add_argument('--dbscan-sample-size', type=int, default=15000, help='Sample size used for DBSCAN.')
    args = parser.parse_args()

    run_segmentation_analysis(
        config_path=args.config,
        n_clusters=args.n_clusters,
        dbscan_sample_size=args.dbscan_sample_size,
    )
