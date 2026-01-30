import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class KKBoxFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Feature Engineering Transformer for KKBox Churn Prediction.
        - Handles missing values (Age imputation).
        - Creates derived ratio features (Interaction features).
        - Prepares data types for LightGBM.
        """
        # Constants for stability
        self.epsilon = 1e-6
        self.age_bins = [0, 18, 25, 35, 50, 80]
        self.age_labels = ['0-18', '19-25', '26-35', '36-50', '51-80']
        
        # State parameters (Learned during fit)
        self.median_age_ = 28.0 # Default fallback

    def fit(self, X, y=None):
        """
        Learn global statistics from the Training Data to ensure consistency in Production.
        """
        # Check if 'bd' (age) exists to calculate median
        if 'bd' in X.columns:
            # Filter valid ages (10-90) to calculate a realistic median
            valid_ages = X.loc[(X['bd'] >= 10) & (X['bd'] <= 90), 'bd']
            if not valid_ages.empty:
                self.median_age_ = float(valid_ages.median())
        
        return self

    def transform(self, X):
        """
        Apply feature engineering logic to generate the Derived Features.
        """
        # Create a copy to avoid SettingWithCopy warnings on the original DF
        df = X.copy()
        
        # ==========================================
        # 1. Demographics & Cleaning
        # ==========================================
        if 'bd' in df.columns:
            # Replace outliers with NaN, then fill with learned median
            df.loc[(df['bd'] < 10) | (df['bd'] > 90), 'bd'] = np.nan
            df['bd'] = df['bd'].fillna(self.median_age_)
            
            # Create Age Groups
            df['age_group'] = pd.cut(df['bd'], bins=self.age_bins, labels=self.age_labels, right=False)
            df['age_group'] = df['age_group'].astype(str).replace('nan', 'Unknown')

        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna('unknown').astype(str).str.lower()

        # ==========================================
        # 2. Transaction Ratios (Financial Behavior)
        # ==========================================
        # Average value per transaction
        if 'total_payment' in df.columns and 'total_transactions' in df.columns:
            df['avg_payment_value'] = df['total_payment'] / (df['total_transactions'] + self.epsilon)
        
        # Cancellation rate
        if 'total_cancel_count' in df.columns and 'total_transactions' in df.columns:
            df['cancel_rate'] = df['total_cancel_count'] / (df['total_transactions'] + self.epsilon)
            df['cancel_rate'] = df['cancel_rate'].clip(0, 1) # Probability cannot exceed 1
            
        # Promo usage ratio
        if 'promo_transaction_count' in df.columns and 'total_transactions' in df.columns:
            df['promo_ratio'] = df['promo_transaction_count'] / (df['total_transactions'] + self.epsilon)
            df['promo_ratio'] = df['promo_ratio'].clip(0, 1)

        # ==========================================
        # 3. Usage Ratios (User Engagement)
        # ==========================================
        # Avg seconds played per active day
        if 'total_secs_played' in df.columns and 'active_days' in df.columns:
            df['avg_secs_played_daily'] = df['total_secs_played'] / (df['active_days'] + self.epsilon)
            
        # Avg unique songs per active day (Discovery rate)
        if 'total_unique_songs' in df.columns and 'active_days' in df.columns:
            df['avg_unique_songs_daily'] = df['total_unique_songs'] / (df['active_days'] + self.epsilon)
            
        # Song Completion Rate (Do they finish songs?)
        if 'total_songs_100_percent' in df.columns and 'total_songs_played' in df.columns:
            df['completion_rate'] = df['total_songs_100_percent'] / (df['total_songs_played'] + self.epsilon)
            df['completion_rate'] = df['completion_rate'].clip(0, 1)
            
        # Uniqueness Rate (Diversity of listening)
        if 'total_unique_songs' in df.columns and 'total_songs_played' in df.columns:
            df['uniqueness_rate'] = df['total_unique_songs'] / (df['total_songs_played'] + self.epsilon)
            df['uniqueness_rate'] = df['uniqueness_rate'].clip(0, 1)

        # ==========================================
        # 4. Behavioral Trends (Change over time)
        # ==========================================
        # Ratio of listening time (Second Half vs First Half)
        if 'total_secs_second_half' in df.columns and 'total_secs_first_half' in df.columns:
            df['secs_trend_ratio'] = df['total_secs_second_half'] / (df['total_secs_first_half'] + self.epsilon)
            # Clip huge jumps (e.g., 0 secs -> 1000 secs = Inf ratio) to 10x
            df['secs_trend_ratio'] = df['secs_trend_ratio'].clip(upper=10).fillna(0)
            
        # Absolute change in active days
        if 'active_days_second_half' in df.columns and 'active_days_first_half' in df.columns:
            df['activity_trend_abs'] = df['active_days_second_half'] - df['active_days_first_half']

        # ==========================================
        # 5. Type Casting for LightGBM
        # ==========================================
        # We explicitly cast categorical columns to 'category' dtype.
        # LightGBM handles this natively and is robust to new categories in production.
        cat_cols = ['city', 'gender', 'registered_via', 'age_group', 'payment_method_id']
        
        for col in cat_cols:
            if col in df.columns:
                # Ensure it's treated as a category, not a number or object
                df[col] = df[col].astype(str).astype('category')

        return df