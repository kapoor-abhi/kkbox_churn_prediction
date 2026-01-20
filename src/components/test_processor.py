import pandas as pd
import numpy as np
from src.components.data_processor import KKBoxFeatureEngineering

def test_feature_calculation():
    # 1. Create dummy data
    data = pd.DataFrame({
        'total_payment': [100.0],
        'total_transactions': [10],
        'total_cancel_count': [1],
        'bd': [30],
        'gender': ['male']
    })
    
    # 2. Run processor
    processor = KKBoxFeatureEngineering()
    processed = processor.transform(data)
    
    # 3. Assertions (The Verification)
    assert processed['avg_payment_value'].iloc[0] == 10.0
    assert processed['cancel_rate'].iloc[0] == 0.1
    assert 'age_group' in processed.columns
    assert processed['age_group'].iloc[0] == '26-35'

def test_outlier_handling():
    data = pd.DataFrame({'bd': [-500]}) # Junk age
    processor = KKBoxFeatureEngineering()
    processor.fit(data) # Set median
    processed = processor.transform(data)
    
    # Should be filled with default median (28)
    assert processed['bd'].iloc[0] == 28