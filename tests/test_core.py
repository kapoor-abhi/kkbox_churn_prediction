import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from src.components.data_processor import KKBoxFeatureEngineering
from app import app

# --- 1. Test the Math (Processor) ---
def test_processor_division_by_zero():
    """Check if processor handles 0 transactions without crashing."""
    processor = KKBoxFeatureEngineering()
    
    # Create dummy data with 0 transactions
    df = pd.DataFrame({
        'total_payment': [100.0],
        'total_transactions': [0], # Danger!
        'bd': [28]
    })
    
    # Fit and Transform
    processor.fit(df)
    res = processor.transform(df)
    
    # Assertions
    # With epsilon, 100 / (0 + 1e-6) should be a large number, but NOT infinity/error
    assert 'avg_payment_value' in res.columns
    assert not np.isinf(res['avg_payment_value'].iloc[0])

def test_age_imputation():
    """Check if missing age is filled with learned median."""
    processor = KKBoxFeatureEngineering()
    
    # Train data (Median = 30)
    train_df = pd.DataFrame({'bd': [20, 30, 40]})
    processor.fit(train_df)
    
    # Test data (Missing age = 500, outlier)
    test_df = pd.DataFrame({'bd': [500]}) 
    res = processor.transform(test_df)
    
    # Should replace 500 with 30
    assert res['bd'].iloc[0] == 30.0

# --- 2. Test the API ---
client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    # Mock input data
    payload = {
        "msno": "test_user",
        "total_transactions": 5,
        "total_payment": 500,
        "total_cancel_count": 0,
        "promo_transaction_count": 0,
        "avg_plan_days": 30,
        "total_secs_played": 1000,
        "total_unique_songs": 10,
        "total_songs_played": 20,
        "total_songs_100_percent": 10,
        "active_days": 5
    }
    
    # Note: We expect 503 if model isn't loaded locally, 
    # but if you ran the pipeline, it should be 200.
    # We allow 503 to pass this test if model file is missing in test env.
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert "prediction" in data
    else:
        assert response.status_code == 503