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
            'Amount_Log',
            'Transaction_Type_Card',
            'Transaction_Type_UPI',
            'Transaction_Type_Netbanking',
            'Merchant_Type_Online_Retail',
            'Merchant_Type_Food',
            'Merchant_Type_Travel',
            'Merchant_Type_Entertainment',
            'Merchant_Type_Utilities',
            'Merchant_Type_Unknown',
            'Hour',
            'Day_of_Week',
            'Is_Weekend',
            'Is_Night',
            'Hour_Sin',
            'Hour_Cos',
            'Day_Sin',
            'Day_Cos',
            'Time_Since_Last_Txn',
            'Txn_Count_Total',
            'Txn_Count_Last_1H',
            'Txn_Count_Last_6H',
            'Txn_Count_Last_24H',
            'Time_Diff_Hours',
            'Rapid_Txn_Count_Last_1H',
            'Amount_Deviation',
            'Amount_Ratio',
            'Is_Small_Amount',
            'Is_Large_Amount',
            'Amount_Rolling_Mean_10',
            'Amount_Rolling_Std_10',
            'Is_New_Device',
            'Device_Count',
            'Is_Unknown_Device',
            'Location_Distance_Km',
            'Is_Unusual_Location',
            'Location_Speed_Kmh',
            'Is_High_Speed',
            'Is_Suspicious_Receiver',
            'Is_Test_Receiver',
            'Is_First_Time_Receiver',
            'Receiver_Diversity',
            'Receiver_Frequency'
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
        Engineer features for fraud detection with enhanced fraud pattern detection.

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

        # Enhanced time-based features
        data["Time_Since_Last_Txn"] = data.groupby("Sender_ID")["Timestamp"].diff().dt.total_seconds() / 60.0
        data["Time_Since_Last_Txn"] = data["Time_Since_Last_Txn"].fillna(10000)

        # Transaction velocity features (multiple time windows)
        data["Prev_Timestamp"] = data.groupby("Sender_ID")["Timestamp"].shift(1)
        data["Time_Diff_Hours"] = (data["Timestamp"] - data["Prev_Timestamp"]).dt.total_seconds() / 3600
        data["Time_Diff_Hours"] = data["Time_Diff_Hours"].fillna(24)

        # Simplified transaction velocity features
        data["Txn_Count_Last_1H"] = data.groupby("Sender_ID")["Time_Diff_Hours"].transform(
            lambda x: (x <= 1).cumsum().groupby(x.index).last()
        ).fillna(0)

        data["Txn_Count_Last_6H"] = data.groupby("Sender_ID")["Time_Diff_Hours"].transform(
            lambda x: (x <= 6).cumsum().groupby(x.index).last()
        ).fillna(0)

        data["Txn_Count_Last_24H"] = data.groupby("Sender_ID")["Time_Diff_Hours"].transform(
            lambda x: (x <= 24).cumsum().groupby(x.index).last()
        ).fillna(0)

        # Rapid transaction detection (simplified)
        data["Is_Rapid_Txn"] = (data["Time_Diff_Hours"] <= 0.1).astype(int)
        data["Rapid_Txn_Count_Last_1H"] = data.groupby("Sender_ID")["Is_Rapid_Txn"].transform(
            lambda x: x.rolling(window=10, min_periods=1).sum()
        ).fillna(0)

        # Transaction count features
        sender_txn_counts = data.groupby("Sender_ID").cumcount()
        data["Txn_Count_Total"] = sender_txn_counts + 1

        # Amount-based features with enhanced analysis
        data["Avg_Amount"] = data.groupby("Sender_ID")["Amount"].transform("mean")
        data["Amount_Deviation"] = data["Amount"] - data["Avg_Amount"]
        data["Amount_Ratio"] = data["Amount"] / data["Avg_Amount"].replace(0, 0.01)

        # Amount pattern features
        data["Is_Small_Amount"] = (data["Amount"] <= 100).astype(int)
        data["Is_Large_Amount"] = (data["Amount"] >= 50000).astype(int)
        data["Amount_Log"] = np.log1p(data["Amount"])  # Log transform for scale invariance

        # Rolling statistics for amount patterns
        data["Amount_Rolling_Mean_10"] = data.groupby("Sender_ID")["Amount"].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        data["Amount_Rolling_Std_10"] = data.groupby("Sender_ID")["Amount"].transform(
            lambda x: x.rolling(10, min_periods=1).std()
        ).fillna(0)

        # Device-based features
        data["Prev_Device"] = data.groupby("Sender_ID")["Device_ID"].shift(1)
        data["Is_New_Device"] = (data["Device_ID"] != data["Prev_Device"]).astype(int)
        data["Is_New_Device"] = data["Is_New_Device"].fillna(0)

        # Device consistency features
        data["Device_Count"] = data.groupby("Sender_ID")["Device_ID"].transform("nunique")
        data["Is_Unknown_Device"] = data["Device_ID"].str.contains("Unknown").astype(int)

        # Enhanced location-based features
        data["Prev_Location"] = data.groupby("Sender_ID")["Location"].shift(1)

        def haversine_distance(loc1, loc2):
            """Calculate haversine distance between two lat/lon points"""
            try:
                if pd.isna(loc1) or pd.isna(loc2):
                    return float('inf')
                lat1, lon1 = map(float, loc1.split(','))
                lat2, lon2 = map(float, loc2.split(','))

                # Haversine formula
                R = 6371  # Earth's radius in kilometers
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return R * c
            except:
                return float('inf')

        data["Location_Distance_Km"] = data.apply(
            lambda x: haversine_distance(x["Location"], x["Prev_Location"]), axis=1
        )

        # Replace infinity with reasonable defaults
        data["Location_Distance_Km"] = data["Location_Distance_Km"].replace([np.inf, -np.inf], 10000)

        # Location anomaly detection
        data["Is_Unusual_Location"] = ((data["Location_Distance_Km"] > 500) &
                                     (data["Time_Diff_Hours"] < 24)).astype(int)
        data["Is_Unusual_Location"] = data["Is_Unusual_Location"].fillna(0)

        # Location velocity (speed between transactions)
        data["Location_Speed_Kmh"] = data["Location_Distance_Km"] / data["Time_Diff_Hours"].replace(0, 0.01)
        data["Location_Speed_Kmh"] = data["Location_Speed_Kmh"].replace([np.inf, -np.inf], 0)
        data["Is_High_Speed"] = (data["Location_Speed_Kmh"] > 1000).astype(int)  # Impossible speed

        # Receiver-based features
        data["Is_Suspicious_Receiver"] = data["Receiver_ID"].str.contains("Suspicious").astype(int)
        data["Is_Test_Receiver"] = data["Receiver_ID"].str.contains("Test").astype(int)

        # Receiver pattern analysis
        data["Sender_Receiver_Key"] = data["Sender_ID"] + "_" + data["Receiver_ID"]
        data["Receiver_Count"] = data.groupby("Sender_Receiver_Key").cumcount()
        data["Is_First_Time_Receiver"] = (data["Receiver_Count"] == 0).astype(int)

        # Simplified receiver diversity features
        data["Receiver_Diversity"] = data.groupby("Sender_ID")["Receiver_ID"].transform("nunique")
        data["Receiver_Frequency"] = data.groupby(["Sender_ID", "Receiver_ID"]).cumcount() + 1

        # Additional features removed to focus on core functionality

        # Behavioral pattern features
        data["Hour_Sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
        data["Hour_Cos"] = np.cos(2 * np.pi * data["Hour"] / 24)
        data["Day_Sin"] = np.sin(2 * np.pi * data["Day_of_Week"] / 7)
        data["Day_Cos"] = np.cos(2 * np.pi * data["Day_of_Week"] / 7)

        # Drop temporary columns but keep original ones
        cols_to_drop = [
            "Prev_Timestamp", "Prev_Device", "Prev_Location", "Sender_Receiver_Key",
            "Receiver_Count", "Avg_Amount", "Is_Rapid_Txn", "Prev_Txn_Type",
            "Txn_Count_Last_10_Min"  # Legacy column
        ]
        model_data = data.drop(columns=cols_to_drop, errors='ignore')

        # Replace any remaining infinity values with 0
        model_data = model_data.replace([np.inf, -np.inf], 0)

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