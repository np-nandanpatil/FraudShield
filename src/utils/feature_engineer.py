import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for transaction data"""
    
    def __init__(self):
        # Define numerical and categorical columns
        self.numerical_cols = ['Amount']
        self.categorical_cols = [
            'Transaction_Type', 'Merchant_Type',
            'Device_ID', 'Location', 'Sender_ID', 'Receiver_ID'
        ]
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ]
        )
        
        # Initialize feature names
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame):
        """Fit the feature engineer on training data"""
        try:
            # Ensure all required columns exist
            for col in self.numerical_cols + self.categorical_cols:
                if col not in X.columns:
                    X[col] = None
            
            # Fill NaN values
            X[self.numerical_cols] = X[self.numerical_cols].fillna(0)
            X[self.categorical_cols] = X[self.categorical_cols].fillna('Unknown')
            
            # Fit the preprocessor
            self.preprocessor.fit(X)
            
            # Get feature names
            self._get_feature_names(X)
            
            logger.info("Feature engineer fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting feature engineer: {str(e)}")
            raise
            
    def _get_feature_names(self, X: pd.DataFrame):
        """Get feature names after one-hot encoding"""
        try:
            # Get numerical feature names
            num_features = self.numerical_cols
            
            # Get categorical feature names
            cat_features = []
            for col in self.categorical_cols:
                if col in X.columns:
                    unique_values = X[col].unique()
                    cat_features.extend([f"{col}_{val}" for val in unique_values])
            
            self.feature_names = num_features + cat_features
            
        except Exception as e:
            logger.error(f"Error getting feature names: {str(e)}")
            self.feature_names = None
            
    def process_transactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of transactions"""
        try:
            # Ensure all required columns exist
            for col in self.numerical_cols + self.categorical_cols:
                if col not in X.columns:
                    X[col] = None
            
            # Fill NaN values
            X[self.numerical_cols] = X[self.numerical_cols].fillna(0)
            X[self.categorical_cols] = X[self.categorical_cols].fillna('Unknown')
            
            # Transform data
            X_transformed = self.preprocessor.transform(X)
            
            # Convert to DataFrame
            if self.feature_names:
                X_df = pd.DataFrame(X_transformed, columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X_transformed)
            
            # Ensure no NaN values
            X_df = X_df.fillna(0)
            
            return X_df
            
        except Exception as e:
            logger.error(f"Error processing transactions: {str(e)}")
            raise
            
    def process_single_transaction(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Process a single transaction"""
        try:
            # Convert to DataFrame
            X = pd.DataFrame([transaction])
            
            # Process using batch method
            X_processed = self.process_transactions(X)
            
            # Convert to dictionary
            return X_processed.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error processing single transaction: {str(e)}")
            raise
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Alias for process_transactions to match scikit-learn interface"""
        return self.process_transactions(X)
        
    def save(self, path: str):
        """Save the feature engineer to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"Feature engineer saved to {path}")
        except Exception as e:
            logger.error(f"Error saving feature engineer: {str(e)}")
            raise
            
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load a feature engineer from disk"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Feature engineer file not found: {path}")
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading feature engineer: {str(e)}")
            raise 