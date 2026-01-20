import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class KKBoxFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.epsilon = 1e-6
        self.age_bins = [0, 18, 25, 35, 50, 80]
        self.age_labels = ['0-18', '19-25', '26-35', '36-50', '51-80']
        self.median_age = 28

    def fit(self, X, y=None):
        if 'bd' in X.columns:
            valid_ages = X.loc[(X['bd'] > 10) & (X['bd'] < 90), 'bd']
            if not valid_ages.empty:
                self.median_age = float(valid_ages.median())
        return self

    def transform(self, X):
        df = X.copy()
        
        # --- 1. Demographics ---
        if 'bd' in df.columns:
            df.loc[(df['bd'] < 10) | (df['bd'] > 90), 'bd'] = np.nan
            df['bd'] = df['bd'].fillna(self.median_age)
            df['age_group'] = pd.cut(df['bd'], bins=self.age_bins, labels=self.age_labels, right=False)
            df['age_group'] = df['age_group'].cat.add_categories('Unknown').fillna('Unknown')

        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.lower().fillna('unknown')

        # --- 2. Transaction Ratios ---
        if 'total_payment' in df.columns and 'total_transactions' in df.columns:
            df['avg_payment_value'] = df['total_payment'] / (df['total_transactions'] + self.epsilon)
        
        if 'total_cancel_count' in df.columns and 'total_transactions' in df.columns:
            df['cancel_rate'] = df['total_cancel_count'] / (df['total_transactions'] + self.epsilon)
            
        if 'promo_transaction_count' in df.columns and 'total_transactions' in df.columns:
            df['promo_ratio'] = df['promo_transaction_count'] / (df['total_transactions'] + self.epsilon)

        # --- 3. Usage Ratios (User Logs) ---
        if 'total_secs_played' in df.columns and 'active_days' in df.columns:
            df['avg_secs_played_daily'] = df['total_secs_played'] / (df['active_days'] + self.epsilon)
            
        if 'total_unique_songs' in df.columns and 'active_days' in df.columns:
            df['avg_unique_songs_daily'] = df['total_unique_songs'] / (df['active_days'] + self.epsilon)
            
        if 'total_songs_100_percent' in df.columns and 'total_songs_played' in df.columns:
            df['completion_rate'] = df['total_songs_100_percent'] / (df['total_songs_played'] + self.epsilon)
            
        if 'total_unique_songs' in df.columns and 'total_songs_played' in df.columns:
            df['uniqueness_rate'] = df['total_unique_songs'] / (df['total_songs_played'] + self.epsilon)

        # --- 4. Trend Processing ---
        if 'total_secs_second_half' in df.columns and 'total_secs_first_half' in df.columns:
            df['secs_trend_ratio'] = df['total_secs_second_half'] / (df['total_secs_first_half'] + self.epsilon)
            df['secs_trend_ratio'] = df['secs_trend_ratio'].clip(upper=10).fillna(0)
            
        if 'active_days_second_half' in df.columns and 'active_days_first_half' in df.columns:
            df['activity_trend_abs'] = df['active_days_second_half'] - df['active_days_first_half']

        return df