import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from src.components.segmentation_analysis import run_segmentation_analysis


st.set_page_config(
    page_title="KKBOX Churn Prediction & EDA", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    }
    .css-1d391kg, [data-testid="stHeader"] {
        background: transparent !important;
    }
    h1, h2, h3, .stMetric label {
        color: #E94560 !important;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(233, 69, 96, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


@st.cache_data(show_spinner=False)
def load_analysis_data(config_path):
    config = load_config(config_path)
    analysis_dir = config['data_ingestion'].get(
        'analysis_dir',
        os.path.join(config['data_ingestion']['root_dir'], 'analysis'),
    )
    master_features_path = os.path.join(
        config['data_ingestion']['root_dir'],
        'featured',
        'master_features.parquet',
    )
    analysis_dataset_path = os.path.join(analysis_dir, 'analysis_dataset.parquet')
    summary_path = os.path.join(analysis_dir, 'analysis_summary.json')
    rfm_profile_path = os.path.join(analysis_dir, 'rfm_profile.csv')
    kmeans_profile_path = os.path.join(analysis_dir, 'kmeans_profile.csv')
    dbscan_profile_path = os.path.join(analysis_dir, 'dbscan_profile.csv')
    dbscan_sample_path = os.path.join(analysis_dir, 'dbscan_sample.parquet')

    if not os.path.exists(analysis_dataset_path):
        run_segmentation_analysis(config_path)

    with open(summary_path, 'r') as file:
        summary = json.load(file)

    return {
        'master_df': pd.read_parquet(master_features_path),
        'analysis_df': pd.read_parquet(analysis_dataset_path),
        'rfm_profile': pd.read_csv(rfm_profile_path),
        'kmeans_profile': pd.read_csv(kmeans_profile_path),
        'dbscan_profile': pd.read_csv(dbscan_profile_path),
        'dbscan_sample': pd.read_parquet(dbscan_sample_path),
        'summary': summary,
    }


@st.cache_resource(show_spinner=False)
def load_model_pipeline(model_path="data/model/production_pipeline.pkl"):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def render_metrics(summary):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Churn Rate", f"{summary['churn_rate']:.2%}")
    col3.metric("KMeans Clusters", summary['kmeans_clusters'])
    col4.metric("PCA Variance", f"{summary['pca_explained_variance']:.2%}")


def build_prediction_frame(master_df, msno):
    user_matches = master_df[master_df['msno'] == msno]
    if user_matches.empty:
        return None

    user_row = user_matches.iloc[[0]].copy()
    drop_columns = ['msno', 'is_churn', 'registration_init_time']
    return user_row.drop(columns=drop_columns, errors='ignore')


def main():
    st.title("KKBOX Local Churn Analysis Workbench")
    st.caption("Data-card aligned feature review, RFM segmentation, clustering, and model scoring in one local app.")

    config_path = st.sidebar.text_input("Config path", value="config/config.yaml")
    if st.sidebar.button("Refresh analysis outputs"):
        with st.spinner("Rebuilding analysis outputs..."):
            run_segmentation_analysis(config_path)
            load_analysis_data.clear()
            st.success("Analysis outputs refreshed.")

    data = load_analysis_data(config_path)
    master_df = data['master_df']
    analysis_df = data['analysis_df']
    rfm_profile = data['rfm_profile']
    kmeans_profile = data['kmeans_profile']
    dbscan_profile = data['dbscan_profile']
    dbscan_sample = data['dbscan_sample']
    summary = data['summary']

    render_metrics(summary)

    tabs = st.tabs(["Overview", "Advanced EDA", "Feature Explorer", "RFM", "Clusters", "Model Lab"])

    with tabs[0]:
        st.subheader("Dataset snapshot")
        sample_size = st.slider("Preview sample size", min_value=1000, max_value=50000, value=10000, step=1000)
        preview_df = analysis_df.sample(min(sample_size, len(analysis_df)), random_state=42)

        churn_by_history = (
            analysis_df.groupby('has_transaction_history')['is_churn']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'churn_rate', 'count': 'user_count'})
        )

        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(
                px.histogram(
                    preview_df,
                    x='days_since_last_transaction',
                    color='is_churn',
                    nbins=60,
                    title='Recency distribution by churn label',
                ),
                use_container_width=True,
            )
        with right_col:
            st.plotly_chart(
                px.bar(
                    churn_by_history,
                    x='has_transaction_history',
                    y='churn_rate',
                    text='user_count',
                    title='Churn rate by transaction-history flag',
                ),
                use_container_width=True,
            )

        st.dataframe(
            analysis_df[['msno', 'is_churn', 'days_since_last_transaction', 'total_transactions', 'total_payment', 'RFM_Segment', 'kmeans_cluster']].head(25),
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("Advanced EDA & Churn Insights")
        st.markdown("Detailed breakdown of churn ratios across specific interaction classes.")
        
        # 1. Churn vs Promo Transactions
        if 'promo_transaction_count' in analysis_df.columns:
            def categorize_promo(x):
                if x == 0: return '0 Promos'
                elif x <= 2: return '1-2 Promos'
                else: return '> 2'
            analysis_df['promo_bin'] = analysis_df['promo_transaction_count'].apply(categorize_promo)
            churn_by_promo = analysis_df.groupby('promo_bin')['is_churn'].agg(['mean', 'count']).reset_index()
            # Ensure order: 0 Promos, 1-2 Promos, > 2
            churn_by_promo['order'] = churn_by_promo['promo_bin'].map({'0 Promos': 1, '1-2 Promos': 2, '> 2': 3})
            churn_by_promo = churn_by_promo.sort_values('order')
            
        # 2. Churn vs Promo Ratio
        if 'promo_ratio' in analysis_df.columns:
            analysis_df['promo_ratio_bin'] = pd.cut(analysis_df['promo_ratio'], bins=[-0.01, 0.1, 0.5, 1.0], labels=['Low (<10%)', 'Medium (10%-50%)', 'High (>50%)'])
            churn_by_ratio = analysis_df.groupby('promo_ratio_bin')['is_churn'].agg(['mean', 'count']).reset_index()
            
        c1, c2 = st.columns(2)
        with c1:
            fig_p = px.bar(churn_by_promo, x='promo_bin', y='mean', text='count',
                           title="Churn Rate: Count of Promo Txns", color='mean',
                           color_continuous_scale="Reds")
            fig_p.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_p, use_container_width=True)
            
            # Dynamic Insight text
            zero_churn = churn_by_promo[churn_by_promo['promo_bin']=='0 Promos']['mean'].values[0]
            high_churn = churn_by_promo[churn_by_promo['promo_bin']=='> 2']['mean'].values[0] if '> 2' in churn_by_promo['promo_bin'].values else 0
            st.info(f"**Insight:** Users with >2 Promo Transactions have a churn rate of {high_churn:.1%} compared to {zero_churn:.1%} for users with 0 promotions. Promotional hunting is strongly correlated with varying retention.")
            
        with c2:
            fig_r = px.bar(churn_by_ratio, x='promo_ratio_bin', y='mean', text='count',
                           title="Churn Rate: Promo Transaction Ratio", color='mean',
                           color_continuous_scale="Reds")
            fig_r.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_r, use_container_width=True)
            
            low_r_churn = churn_by_ratio[churn_by_ratio['promo_ratio_bin']=='Low (<10%)']['mean'].values[0]
            hi_r_churn = churn_by_ratio[churn_by_ratio['promo_ratio_bin']=='High (>50%)']['mean'].values[0]
            st.info(f"**Insight:** A high promo ratio (>50%) leads to a churn rate of {hi_r_churn:.1%}, compared to {low_r_churn:.1%} for low ratio users. High promo reliance indicates lower organic loyalty.")
            
        st.markdown("---")
        # 3. Churn behavior by Days Since Last Transaction
        if 'days_since_last_transaction' in analysis_df.columns:
            # Filter out negative missing placeholders (-1)
            valid_recency = analysis_df[analysis_df['days_since_last_transaction'] >= 0].copy()
            if not valid_recency.empty:
                v_bins, _ = pd.qcut(valid_recency['days_since_last_transaction'], q=5, retbins=True, duplicates='drop')
                valid_recency['recency_bin'] = pd.cut(valid_recency['days_since_last_transaction'], bins=_, include_lowest=True).astype(str)
                churn_by_recency = valid_recency.groupby('recency_bin')['is_churn'].agg(['mean', 'count']).reset_index()
                
                # Plot
                fig_rec = px.line(churn_by_recency, x='recency_bin', y='mean', markers=True, 
                                  title="Churn Rate vs Recency (Days since last tx - Non-Ghost Users)", text='count')
                fig_rec.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_rec, use_container_width=True)
                st.info("**Insight:** As the days since the last transaction increase, the probability of churn typically changes. Extreme recency often indicates high engagement and lower churn risk among non-ghost users.")

    with tabs[2]:
        st.subheader("Feature explorer")
        numeric_columns = [
            column for column in analysis_df.columns
            if pd.api.types.is_numeric_dtype(analysis_df[column]) and column not in {'is_churn'}
        ]
        selected_feature = st.selectbox("Numeric feature", sorted(numeric_columns), index=0)
        view_df = analysis_df[[selected_feature, 'is_churn']].dropna()
        st.plotly_chart(
            px.box(
                view_df,
                x='is_churn',
                y=selected_feature,
                title=f'{selected_feature} by churn label',
                points=False,
            ),
            use_container_width=True,
        )

        corr_source = analysis_df[numeric_columns + ['is_churn']].sample(min(25000, len(analysis_df)), random_state=42)
        corr_matrix = corr_source.corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr_matrix, aspect='auto', title='Correlation matrix (sampled)'),
            use_container_width=True,
        )

    with tabs[2]:
        st.subheader("RFM segmentation")
        st.plotly_chart(
            px.bar(
                rfm_profile.sort_values('churn_rate', ascending=False),
                x='churn_rate',
                y='RFM_Segment',
                color='user_count',
                orientation='h',
                title='RFM segment churn profile',
            ),
            use_container_width=True,
        )

        selected_segment = st.selectbox("Inspect segment", rfm_profile['RFM_Segment'].tolist())
        segment_df = analysis_df[analysis_df['RFM_Segment'] == selected_segment]
        st.write(f"Users in segment: {len(segment_df):,}")
        st.dataframe(
            segment_df[['msno', 'is_churn', 'days_since_last_transaction', 'total_transactions', 'total_payment']].head(50),
            use_container_width=True,
        )

    with tabs[3]:
        st.subheader("PCA and cluster views")
        sampled_projection = analysis_df.sample(min(30000, len(analysis_df)), random_state=42)
        st.plotly_chart(
            px.scatter(
                sampled_projection,
                x='PCA_1',
                y='PCA_2',
                color='kmeans_cluster',
                hover_data=['is_churn', 'RFM_Segment'],
                title='KMeans clusters in PCA space',
                opacity=0.6,
            ),
            use_container_width=True,
        )

        left_col, right_col = st.columns(2)
        with left_col:
            st.dataframe(kmeans_profile, use_container_width=True)
        with right_col:
            st.dataframe(dbscan_profile, use_container_width=True)

        st.plotly_chart(
            px.scatter(
                dbscan_sample,
                x='PCA_1',
                y='PCA_2',
                color='dbscan_cluster',
                hover_data=['is_churn'],
                title='DBSCAN sample view',
                opacity=0.7,
            ),
            use_container_width=True,
        )

    with tabs[4]:
        st.subheader("Model scoring playground")
        pipeline = load_model_pipeline()
        if pipeline is None:
            st.warning("`data/model/production_pipeline.pkl` is missing. Train the model first to enable predictions.")
        else:
            msno = st.text_input("Lookup an existing `msno`", value=analysis_df['msno'].iloc[0])
            prediction_frame = build_prediction_frame(master_df, msno)
            if prediction_frame is None:
                st.info("No row found for that user id. You can still score the first available user below.")
                fallback_msno = master_df['msno'].iloc[0]
                prediction_frame = build_prediction_frame(master_df, fallback_msno)
                msno = fallback_msno

            if prediction_frame is not None:
                # Ensure we do not display proxy sensitive information in the playground explicitly
                display_frame = prediction_frame.copy()
                st.dataframe(display_frame, use_container_width=True)
                if st.button("Score selected user"):
                    probability = float(pipeline.predict_proba(prediction_frame)[:, 1][0])
                    st.metric("Predicted churn probability", f"{probability:.2%}")
                    st.write("Predicted label:", "Churn" if probability >= 0.5 else "Stay")


if __name__ == "__main__":
    main()
