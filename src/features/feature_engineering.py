import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering class with default values."""
        self.numerical_cols = ["Amount"]
        self.categorical_cols = ["Transaction_Type", "Merchant_Type"]
        self.scaler = StandardScaler()
        self.encoded_cols = []
        
    def preprocess(self, df, fit=False):
        """
        Preprocess raw transaction data.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            fit (bool): Whether to fit or transform with the scaler
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Drop the Transaction_ID column if it exists
        if "Transaction_ID" in data.columns:
            data = data.drop(columns=["Transaction_ID"])
        
        # Convert Timestamp to datetime if it's not already
        if "Timestamp" in data.columns and data["Timestamp"].dtype != 'datetime64[ns]':
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])
        
        # Handle missing values
        data.fillna({
            "Amount": 0,
            "Merchant_Type": "Unknown",
            "Device_ID": "Unknown",
            "Location": "0.0, 0.0"
        }, inplace=True)
        
        # Scale numerical features
        if fit:
            data[self.numerical_cols] = self.scaler.fit_transform(data[self.numerical_cols])
        else:
            data[self.numerical_cols] = self.scaler.transform(data[self.numerical_cols])
        
        # One-hot encode categorical variables
        encoded_data = pd.get_dummies(data, columns=self.categorical_cols, drop_first=True)
        
        # Store encoded column names for future use
        if fit:
            self.encoded_cols = encoded_data.columns.tolist()
        
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
        
        # Time-based features
        data["Hour"] = data["Timestamp"].dt.hour
        data["Day_of_Week"] = data["Timestamp"].dt.dayofweek
        data["Is_Weekend"] = data["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)
        data["Is_Night"] = data["Hour"].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)
        
        # Time since last transaction (in minutes) for the same sender
        data["Time_Since_Last_Txn"] = data.groupby("Sender_ID")["Timestamp"].diff().dt.total_seconds() / 60.0
        # Fill NaN values with a high value for first transactions
        data["Time_Since_Last_Txn"] = data["Time_Since_Last_Txn"].fillna(10000)
        
        # Transaction count features (simplified to avoid rolling window issues)
        # Count of transactions by the same sender
        sender_txn_counts = data.groupby("Sender_ID").cumcount()
        data["Txn_Count_Total"] = sender_txn_counts + 1  # Add 1 to start from 1 instead of 0
        
        # For rapid transactions, count transactions in the last hour for each sender
        # Group by sender and calculate time differences between consecutive transactions
        data["Prev_Timestamp"] = data.groupby("Sender_ID")["Timestamp"].shift(1)
        data["Time_Diff_Hours"] = (data["Timestamp"] - data["Prev_Timestamp"]).dt.total_seconds() / 3600
        data["Time_Diff_Hours"] = data["Time_Diff_Hours"].fillna(24)  # Assume 24 hours for first transaction
        
        # Count recent transactions (in last hour - simplification of sliding window)
        data["Is_Recent"] = (data["Time_Diff_Hours"] <= 1).astype(int)
        data["Recent_Txn_Count"] = data.groupby("Sender_ID")["Is_Recent"].cumsum()
        
        # Amount-based features
        # Calculate average amount per sender
        data["Avg_Amount"] = data.groupby("Sender_ID")["Amount"].transform("mean")
        
        # Amount deviation from sender's average (high deviation indicates potential fraud)
        data["Amount_Deviation"] = data["Amount"] - data["Avg_Amount"]
        
        # Ratio of current amount to sender's average (high ratio indicates potential fraud)
        data["Amount_Ratio"] = data["Amount"] / data["Avg_Amount"].replace(0, 0.01)  # Avoid division by zero
        
        # Device-based features
        # Flag for new device (1 if device changed from last transaction)
        data["Prev_Device"] = data.groupby("Sender_ID")["Device_ID"].shift(1)
        data["Is_New_Device"] = (data["Device_ID"] != data["Prev_Device"]).astype(int)
        data["Is_New_Device"] = data["Is_New_Device"].fillna(0)
        
        # Location-based features
        # Flag for unusual location (simplified: 1 if location changed significantly)
        data["Prev_Location"] = data.groupby("Sender_ID")["Location"].shift(1)
        data["Is_Unusual_Location"] = (data["Location"] != data["Prev_Location"]).astype(int)
        data["Is_Unusual_Location"] = data["Is_Unusual_Location"].fillna(0)
        
        # Receiver-based features
        # Flag for suspicious receiver (contains 'Suspicious' in the name)
        data["Is_Suspicious_Receiver"] = data["Receiver_ID"].str.contains("Suspicious").astype(int)
        
        # First time transaction with this receiver
        data["Sender_Receiver_Key"] = data["Sender_ID"] + "_" + data["Receiver_ID"]
        data["Receiver_Count"] = data.groupby("Sender_Receiver_Key").cumcount()
        data["Is_First_Time_Receiver"] = (data["Receiver_Count"] == 0).astype(int)
        
        # Drop temporary and unnecessary columns
        cols_to_drop = [
            "Timestamp", "Sender_ID", "Receiver_ID", "Device_ID", "Location", 
            "Prev_Timestamp", "Prev_Device", "Prev_Location", "Sender_Receiver_Key",
            "Receiver_Count", "Avg_Amount", "Is_Recent"
        ]
        model_data = data.drop(columns=cols_to_drop, errors='ignore')
        
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
        
        # Split into features and target
        if "Is_Fraud" in engineered_data.columns:
            y = engineered_data["Is_Fraud"]
            X = engineered_data.drop(columns=["Is_Fraud"])
        else:
            y = None
            X = engineered_data
            
        return X, y
    
    def save(self, output_path):
        """Save the feature engineer to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"Feature engineer saved to {output_path}")
    
    @classmethod
    def load(cls, input_path):
        """Load a feature engineer from disk."""
        return joblib.load(input_path)
        
def process_single_transaction(transaction, feature_engineer):
    """
    Process a single transaction dictionary.
    
    Args:
        transaction (dict): Single transaction data
        feature_engineer (FeatureEngineer): Fitted feature engineer
        
    Returns:
        pd.DataFrame: Processed features for prediction
    """
    # Convert to DataFrame if dictionary
    if isinstance(transaction, dict):
        txn_df = pd.DataFrame([transaction])
    else:
        txn_df = pd.DataFrame([transaction.to_dict()])
    
    # Process data
    X, _ = feature_engineer.process_data(txn_df, fit=False)
    
    # Ensure all columns from training are present
    for col in feature_engineer.encoded_cols:
        if col not in X.columns and col != "Is_Fraud":
            X[col] = 0
    
    # Keep only the columns used during training
    X = X[feature_engineer.encoded_cols]
    
    return X 