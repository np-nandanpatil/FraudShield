import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any
from datetime import datetime, timedelta

# Add module path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering class with default values."""
        self.numerical_cols = ["Amount"]
        self.categorical_cols = ["Transaction_Type", "Merchant_Type"]
        self.scaler = StandardScaler()
        self.encoded_cols = [
            'Amount',
            'Transaction_Type_Payment',
            'Transaction_Type_Transfer',
            'Transaction_Type_Withdrawal',
            'Merchant_Type_Online',
            'Merchant_Type_POS',
            'Merchant_Type_Retail',
            'Merchant_Type_Unknown',
            'Hour',
            'Day_of_Week',
            'Is_Weekend',
            'Is_Night',
            'Time_Since_Last_Txn',
            'Txn_Count_Total',
            'Time_Diff_Hours',
            'Recent_Txn_Count',
            'Amount_Deviation',
            'Amount_Ratio',
            'Is_New_Device',
            'Is_Unusual_Location',
            'Is_Suspicious_Receiver',
            'Is_First_Time_Receiver'
        ]
        # Define required columns if not already defined
        if not hasattr(self, 'required_columns'):
            self.required_columns = [
                'Amount', 'Sender_ID', 'Receiver_ID', 'Device_ID', 'Location',
                'Transaction_Type', 'Merchant_Type', 'Timestamp', 'Is_Fraud'
            ]
        
    def preprocess(self, df, fit=False):
        """Preprocess raw transaction data."""
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Define required fields and their default values
        required_fields = {
            'Amount': 0.0,
            'Sender_ID': 'Unknown',
            'Receiver_ID': 'Unknown',
            'Device_ID': 'Unknown',
            'Location': '0.0, 0.0',
            'Transaction_Type': 'Unknown',
            'Merchant_Type': 'Unknown',
            'Timestamp': pd.Timestamp.now(),
            'Is_Fraud': 0
        }
        
        # Ensure all required columns exist with default values
        for field, default_value in required_fields.items():
            if field not in data.columns:
                data[field] = default_value
        
        # Convert Timestamp to datetime if it's not already
        if data["Timestamp"].dtype != 'datetime64[ns]':
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])
        
        # Handle missing values
        data.fillna({
            "Amount": 0.0,
            "Merchant_Type": "Unknown",
            "Device_ID": "Unknown",
            "Location": "0.0, 0.0",
            "Transaction_Type": "Unknown",
            "Sender_ID": "Unknown",
            "Receiver_ID": "Unknown"
        }, inplace=True)
        
        # Scale numerical features
        if fit:
            data[self.numerical_cols] = self.scaler.fit_transform(data[self.numerical_cols])
        else:
            data[self.numerical_cols] = self.scaler.transform(data[self.numerical_cols])
        
        # One-hot encode categorical variables
        encoded_data = pd.get_dummies(data, columns=self.categorical_cols)
        
        # Ensure all encoded columns exist
        for col in self.encoded_cols:
            if col not in encoded_data.columns:
                if col.startswith('Transaction_Type_') or col.startswith('Merchant_Type_'):
                    encoded_data[col] = 0
        
        return encoded_data
    
    def engineer_features(self, df):
        """
        Engineer features for fraud detection.
        
        Args:
            df (pd.DataFrame): Preprocessed transaction data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        # Ensure data is sorted by timestamp
        data = df.sort_values(by=["Sender_ID", "Timestamp"]).reset_index(drop=True)
        
        # Store original columns that we need to keep
        original_cols = ["Sender_ID", "Receiver_ID", "Device_ID", "Location", "Timestamp"]
        
        # Time-based features
        data["Hour"] = data["Timestamp"].dt.hour
        data["Day_of_Week"] = data["Timestamp"].dt.dayofweek
        data["Is_Weekend"] = data["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)
        data["Is_Night"] = data["Hour"].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)
        
        # Time since last transaction (in minutes) for the same sender
        data["Time_Since_Last_Txn"] = data.groupby("Sender_ID")["Timestamp"].diff().dt.total_seconds() / 60.0
        # Fill NaN values with a high value for first transactions
        data["Time_Since_Last_Txn"] = data["Time_Since_Last_Txn"].fillna(10000)
        
        # Transaction count features
        sender_txn_counts = data.groupby("Sender_ID").cumcount()
        data["Txn_Count_Total"] = sender_txn_counts + 1
        
        # For rapid transactions
        data["Prev_Timestamp"] = data.groupby("Sender_ID")["Timestamp"].shift(1)
        data["Time_Diff_Hours"] = (data["Timestamp"] - data["Prev_Timestamp"]).dt.total_seconds() / 3600
        data["Time_Diff_Hours"] = data["Time_Diff_Hours"].fillna(24)
        
        data["Is_Recent"] = (data["Time_Diff_Hours"] <= 1).astype(int)
        data["Recent_Txn_Count"] = data.groupby("Sender_ID")["Is_Recent"].cumsum()
        
        # Only use Txn_Count_Last_10_Min if it exists and Recent_Txn_Count is not already set
        if "Txn_Count_Last_10_Min" in data.columns and "Recent_Txn_Count" not in data.columns:
            data["Recent_Txn_Count"] = data["Txn_Count_Last_10_Min"]
        
        # Amount-based features
        data["Avg_Amount"] = data.groupby("Sender_ID")["Amount"].transform("mean")
        data["Amount_Deviation"] = data["Amount"] - data["Avg_Amount"]
        data["Amount_Ratio"] = data["Amount"] / data["Avg_Amount"].replace(0, 0.01)
        
        # Device-based features
        data["Prev_Device"] = data.groupby("Sender_ID")["Device_ID"].shift(1)
        data["Is_New_Device"] = (data["Device_ID"] != data["Prev_Device"]).astype(int)
        data["Is_New_Device"] = data["Is_New_Device"].fillna(0)
        
        # Location-based features
        data["Prev_Location"] = data.groupby("Sender_ID")["Location"].shift(1)
        
        def calculate_distance(loc1, loc2):
            try:
                if pd.isna(loc1) or pd.isna(loc2):
                    return float('inf')
                lat1, lon1 = map(float, loc1.split(','))
                lat2, lon2 = map(float, loc2.split(','))
                # Simple Euclidean distance - can be replaced with Haversine formula for more accuracy
                return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
            except:
                return float('inf')
        
        data["Location_Distance"] = data.apply(
            lambda x: calculate_distance(x["Location"], x["Prev_Location"]), axis=1
        )
        
        # Consider location unusual if distance is greater than 100km and time difference is less than 24 hours
        data["Time_Diff_Hours"] = data.groupby("Sender_ID")["Timestamp"].diff().dt.total_seconds() / 3600
        data["Is_Unusual_Location"] = ((data["Location_Distance"] > 100) & 
                                     (data["Time_Diff_Hours"] < 24)).astype(int)
        data["Is_Unusual_Location"] = data["Is_Unusual_Location"].fillna(0)
        
        # Receiver-based features
        data["Is_Suspicious_Receiver"] = data["Receiver_ID"].str.contains("Suspicious").astype(int)
        data["Sender_Receiver_Key"] = data["Sender_ID"] + "_" + data["Receiver_ID"]
        data["Receiver_Count"] = data.groupby("Sender_Receiver_Key").cumcount()
        data["Is_First_Time_Receiver"] = (data["Receiver_Count"] == 0).astype(int)
        
        # Drop temporary columns but keep original ones
        cols_to_drop = [
            "Prev_Timestamp", "Prev_Device", "Prev_Location", "Sender_Receiver_Key",
            "Receiver_Count", "Avg_Amount", "Is_Recent", "Txn_Count_Last_10_Min"
        ]
        model_data = data.drop(columns=cols_to_drop, errors='ignore')
        
        # Ensure Is_Fraud column exists for prediction
        if "Is_Fraud" not in model_data.columns:
            model_data["Is_Fraud"] = 0
            
        return model_data
    
    def process_data(self, df, fit=False):
        """
        Complete data processing pipeline.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            fit (bool): Whether to fit or transform
            
        Returns:
            tuple: (X, y) pair of features and target
        """
        # Preprocess data
        preprocessed_data = self.preprocess(df, fit=fit)
        
        # Engineer features
        engineered_data = self.engineer_features(preprocessed_data)
        
        # Drop raw columns that are not engineered features
        raw_columns = [
            'Device_ID', 'Location', 'Receiver_ID', 'Sender_ID', 'Timestamp',
            'Transaction_Type', 'Merchant_Type'
        ]
        engineered_data = engineered_data.drop(columns=raw_columns, errors='ignore')
        
        # Split into features and target
        if "Is_Fraud" in engineered_data.columns:
            y = engineered_data["Is_Fraud"]
            X = engineered_data.drop(columns=["Is_Fraud"])
        else:
            y = None
            X = engineered_data
            
        # Ensure all required features are present
        for col in self.encoded_cols:
            if col not in X.columns:
                X[col] = 0
                
        # Keep only the required features in the correct order
        X = X[self.encoded_cols]
            
        return X, y
    
    def process_single_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Process a single transaction for real-time prediction."""
        # Create a copy to avoid modifying the original
        transaction = transaction.copy()
        
        # Define required fields and their default values
        required_fields = {
            'Amount': 0.0,
            'Sender_ID': 'Unknown',
            'Receiver_ID': 'Unknown',
            'Device_ID': 'Unknown',
            'Location': '0.0, 0.0',
            'Transaction_Type': 'Unknown',
            'Merchant_Type': 'Unknown',
            'Timestamp': pd.Timestamp.now(),
            'Is_Fraud': 0
        }
        
        # Ensure all required fields are present with proper types
        for field, default_value in required_fields.items():
            if field not in transaction:
                transaction[field] = default_value
        
        # Ensure Timestamp is properly formatted
        if isinstance(transaction['Timestamp'], str):
            transaction['Timestamp'] = pd.to_datetime(transaction['Timestamp'])
        
        # Create DataFrame with single transaction
        df = pd.DataFrame([transaction])
        
        # Process through feature engineering pipeline
        X, _ = self.process_data(df, fit=False)
        
        # Ensure all required features are present
        for col in self.encoded_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Keep only the required features in the correct order
        X = X[self.encoded_cols]
        
        return X
    
    def save(self, output_path):
        """Save the feature engineer to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"Feature engineer saved to {output_path}")
    
    @classmethod
    def load(cls, input_path):
        """Load a feature engineer from disk."""
        return joblib.load(input_path) 