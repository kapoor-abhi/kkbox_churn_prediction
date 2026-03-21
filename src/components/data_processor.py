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
        self.epsilon = 1e-6
        self.age_bins = [0, 18, 25, 35, 50, 101]
        self.age_labels = ['0-18', '19-25', '26-35', '36-50', '51-100']
        self.median_age_ = 28.0
        self.median_registration_age_ = 365.0

    def fit(self, X, y=None):
        """
        Learn global statistics from the Training Data to ensure consistency in Production.
        """
        if 'bd' in X.columns:
            valid_ages = pd.to_numeric(X['bd'], errors='coerce')
            valid_ages = valid_ages[(valid_ages > 0) & (valid_ages <= 100)]
            if not valid_ages.empty:
                self.median_age_ = float(valid_ages.median())

        if 'registration_age_days' in X.columns:
            reg_age = pd.to_numeric(X['registration_age_days'], errors='coerce')
            reg_age = reg_age[reg_age >= 0]
            if not reg_age.empty:
                self.median_registration_age_ = float(reg_age.median())

        return self

    def transform(self, X):
        """
        Apply feature engineering logic to generate the Derived Features.
        """
        df = X.copy()

        if 'bd' in df.columns:
            df['bd'] = pd.to_numeric(df['bd'], errors='coerce')
            df.loc[(df['bd'] <= 0) | (df['bd'] > 100), 'bd'] = np.nan
            df['bd'] = df['bd'].fillna(self.median_age_)
            df['age_group'] = pd.cut(df['bd'], bins=self.age_bins, labels=self.age_labels, right=False)
            df['age_group'] = df['age_group'].astype(str).replace('nan', 'Unknown')

        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna('unknown').astype(str).str.lower()

        if 'registration_age_days' in df.columns:
            df['registration_age_days'] = pd.to_numeric(df['registration_age_days'], errors='coerce')
            df['registration_age_days'] = df['registration_age_days'].fillna(self.median_registration_age_)

        if 'days_since_last_transaction' in df.columns:
            df['days_since_last_transaction'] = pd.to_numeric(
                df['days_since_last_transaction'],
                errors='coerce',
            ).fillna(-1)

        if 'total_payment' in df.columns and 'total_transactions' in df.columns:
            df['avg_payment_value'] = df.get('avg_payment_value', df['total_payment'] / (df['total_transactions'] + self.epsilon))

        if 'total_cancel_count' in df.columns and 'total_transactions' in df.columns:
            df['cancel_rate'] = df.get('cancel_rate', df['total_cancel_count'] / (df['total_transactions'] + self.epsilon))
            df['cancel_rate'] = df['cancel_rate'].clip(0, 1)

        if 'promo_transaction_count' in df.columns and 'total_transactions' in df.columns:
            df['promo_ratio'] = df.get('promo_ratio', df['promo_transaction_count'] / (df['total_transactions'] + self.epsilon))
            df['promo_ratio'] = df['promo_ratio'].clip(0, 1)

        if 'auto_renew_count' in df.columns and 'total_transactions' in df.columns:
            df['auto_renew_ratio'] = df.get(
                'auto_renew_ratio',
                df['auto_renew_count'] / (df['total_transactions'] + self.epsilon),
            )
            df['auto_renew_ratio'] = df['auto_renew_ratio'].clip(0, 1)

        if 'total_secs_played' in df.columns and 'active_days' in df.columns:
            df['avg_secs_played_daily'] = df.get(
                'avg_secs_played_daily',
                df['total_secs_played'] / (df['active_days'] + self.epsilon),
            )

        if 'total_unique_songs' in df.columns and 'active_days' in df.columns:
            df['avg_unique_songs_daily'] = df.get(
                'avg_unique_songs_daily',
                df['total_unique_songs'] / (df['active_days'] + self.epsilon),
            )

        if 'total_songs_100_percent' in df.columns and 'total_songs_played' in df.columns:
            df['completion_rate'] = df.get(
                'completion_rate',
                df['total_songs_100_percent'] / (df['total_songs_played'] + self.epsilon),
            )
            df['completion_rate'] = df['completion_rate'].clip(0, 1)

        if 'total_unique_songs' in df.columns and 'total_songs_played' in df.columns:
            df['uniqueness_rate'] = df.get(
                'uniqueness_rate',
                df['total_unique_songs'] / (df['total_songs_played'] + self.epsilon),
            )
            df['uniqueness_rate'] = df['uniqueness_rate'].clip(0, 1)

        if 'recent_30_secs' in df.columns and 'previous_30_secs' in df.columns:
            df['recent_listening_ratio'] = df.get(
                'recent_listening_ratio',
                df['recent_30_secs'] / (df['previous_30_secs'] + self.epsilon),
            )
            df['recent_listening_ratio'] = df['recent_listening_ratio'].clip(0, 30)

        if 'recent_30_active_days' in df.columns and 'previous_30_active_days' in df.columns:
            df['recent_activity_ratio'] = df.get(
                'recent_activity_ratio',
                df['recent_30_active_days'] / (df['previous_30_active_days'] + self.epsilon),
            )
            df['recent_activity_ratio'] = df['recent_activity_ratio'].clip(0, 30)
            df['activity_trend_abs'] = df['recent_30_active_days'] - df['previous_30_active_days']

        if 'days_until_membership_expire' in df.columns:
            df['days_until_membership_expire'] = pd.to_numeric(
                df['days_until_membership_expire'],
                errors='coerce',
            ).fillna(0)

        if 'has_transaction_history' in df.columns:
            df['has_transaction_history'] = pd.to_numeric(
                df['has_transaction_history'],
                errors='coerce',
            ).fillna(0).astype(int)

        if 'has_log_history' in df.columns:
            df['has_log_history'] = pd.to_numeric(
                df['has_log_history'],
                errors='coerce',
            ).fillna(0).astype(int)

        cat_cols = ['city', 'gender', 'registered_via', 'age_group']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).astype('category')

        return df
