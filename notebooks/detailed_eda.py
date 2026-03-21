import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_data(config_path="config/config.yaml"):
    logging.info("Loading config and data...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    analysis_dir = config['data_ingestion'].get('analysis_dir', 'data/analysis')
    dataset_path = os.path.join(analysis_dir, 'analysis_dataset.parquet')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run pipeline first.")
        
    df = pd.read_parquet(dataset_path)
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

def run_eda(df, output_dir="notebooks/eda_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("1. Overall Churn Rate")
    overall_churn = df['is_churn'].mean()
    logging.info(f"-> Overall Churn Rate: {overall_churn:.2%}")
    
    logging.info("2. Churn Rate vs High Promo Transactions")
    # Define High Promo Transactions
    if 'promo_transaction_count' in df.columns:
        promo_median = df['promo_transaction_count'].median()
        df['high_promo_txn'] = df['promo_transaction_count'] > promo_median
        promo_churn = df.groupby('high_promo_txn')['is_churn'].agg(['mean', 'count']).reset_index()
        logging.info(f"Churn by High Promo Txn (>{promo_median}):\n{promo_churn}")
        
    logging.info("3. Churn Rate vs High Promo Ratio")
    # Define High Promo Ratio
    if 'promo_ratio' in df.columns:
        df['high_promo_ratio'] = df['promo_ratio'] > 0.5
        ratio_churn = df.groupby('high_promo_ratio')['is_churn'].agg(['mean', 'count']).reset_index()
        logging.info(f"Churn by High Promo Ratio (>50%):\n{ratio_churn}")
        
    logging.info("4. Advanced Insights: Cancellation Rate impact")
    if 'cancel_rate' in df.columns:
        df['high_cancel_rate'] = df['cancel_rate'] > 0
        cancel_churn = df.groupby('high_cancel_rate')['is_churn'].agg(['mean', 'count']).reset_index()
        logging.info(f"Churn by has cancelled before:\n{cancel_churn}")

    logging.info("5. Generating Plotly Figures...")
    
    # Figure 1: Churn by Promo Ratio Bins
    df['promo_ratio_bin'] = pd.cut(df['promo_ratio'], bins=5)
    promo_bin_churn = df.groupby('promo_ratio_bin')['is_churn'].mean().reset_index()
    promo_bin_churn['promo_ratio_bin'] = promo_bin_churn['promo_ratio_bin'].astype(str)
    
    fig1 = px.bar(promo_bin_churn, x='promo_ratio_bin', y='is_churn', 
                  title="Churn Rate across Promo Ratios",
                  labels={'is_churn': 'Churn Rate', 'promo_ratio_bin': 'Promo Ratio Bin'},
                  text_auto='.1%')
    fig1.write_html(os.path.join(output_dir, 'churn_vs_promo_ratio.html'))
    
    # Figure 2: RFM Segment Churn 
    if 'RFM_Segment' in df.columns:
        rfm_churn = df.groupby('RFM_Segment')['is_churn'].agg(['mean', 'count']).reset_index()
        fig2 = px.scatter(rfm_churn, x='count', y='mean', color='RFM_Segment', 
                          size='count', title="RFM Segments: Size vs Churn Rate",
                          labels={'mean': 'Churn Rate', 'count': 'Number of Users'})
        fig2.write_html(os.path.join(output_dir, 'rfm_churn_scatter.html'))
        
    logging.info(f"EDA Complete. HTML reports saved to {output_dir}")

if __name__ == "__main__":
    df = load_data()
    run_eda(df)
