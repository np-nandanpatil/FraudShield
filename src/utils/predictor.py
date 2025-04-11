import pandas as pd
import numpy as np
import joblib
import time
import os
import json
from datetime import datetime, timedelta
import threading
from collections import defaultdict
from typing import Dict, List, Any
import logging

class RealTimePredictor:
    """Real-time fraud prediction system"""
    
    def __init__(self, model_path, feature_engineer_path, threshold=0.3):
        """
        Initialize the real-time predictor.
        
        Args:
            model_path (str): Path to the trained model
            feature_engineer_path (str): Path to the saved feature engineer
            threshold (float): Decision threshold for fraud classification
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_engineer_path):
            raise FileNotFoundError(f"Feature engineer file not found: {feature_engineer_path}")
            
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        self.threshold = threshold
        self.predictions_history = []
        self.user_history = defaultdict(list)
        self.history_lock = threading.Lock()
        
    def update_history(self, sender: str, timestamp: datetime) -> int:
        """
        Update transaction history for a user.
        
        Args:
            sender (str): User ID
            timestamp (datetime): Transaction timestamp
            
        Returns:
            int: Number of transactions in last 10 minutes
        """
        with self.history_lock:
            # Remove transactions older than 10 minutes
            cutoff_time = timestamp - timedelta(minutes=10)
            self.user_history[sender] = [
                t for t in self.user_history[sender] 
                if t > cutoff_time
            ]
            # Add new transaction
            self.user_history[sender].append(timestamp)
            return len(self.user_history[sender])
    
    def _infer_fraud_type(self, transaction: Dict[str, Any], fraud_prob: float) -> str:
        """
        Infer the type of fraud based on transaction features.
        
        Args:
            transaction (dict): Transaction data
            fraud_prob (float): Fraud probability
            
        Returns:
            str: Inferred fraud type
        """
        # Feature weights for different fraud types
        fraud_weights = {
            'Phishing Link': {
                'Amount': 0.3,
                'Merchant_Type': 0.2,
                'Receiver_ID': 0.3,
                'Transaction_Type': 0.2
            },
            'QR Code Scam': {
                'Amount': 0.3,
                'Merchant_Type': 0.4,
                'Transaction_Type': 0.3
            },
            'SIM Swap': {
                'Device_ID': 0.4,
                'Location': 0.3,
                'Transaction_Type': 0.3
            },
            'Fake UPI App': {
                'Device_ID': 0.4,
                'Merchant_Type': 0.3,
                'Transaction_Type': 0.3
            },
            'Small Testing': {
                'Amount': 0.4,
                'Txn_Count_Last_10_Min': 0.4,
                'Transaction_Type': 0.2
            },
            'Card Skimming': {
                'Amount': 0.4,
                'Merchant_Type': 0.3,
                'Device_ID': 0.3
            },
            'Data Breach Reuse': {
                'Amount': 0.3,
                'Merchant_Type': 0.3,
                'Device_ID': 0.4
            },
            'Unusual Location': {
                'Location': 0.5,
                'Amount': 0.3,
                'Transaction_Type': 0.2
            },
            'CNP Fraud': {
                'Amount': 0.4,
                'Merchant_Type': 0.3,
                'Device_ID': 0.3
            }
        }
        
        # Calculate scores for each fraud type
        fraud_scores = {}
        for fraud_type, weights in fraud_weights.items():
            score = 0
            for feature, weight in weights.items():
                if feature == 'Amount':
                    # Higher score for unusual amounts
                    amount = float(transaction.get('Amount', 0))
                    if amount > 5000:  # Very high amount
                        score += weight
                    elif amount < 10:  # Very small amount
                        score += weight * 0.8
                elif feature == 'Merchant_Type':
                    # Higher score for unknown merchants
                    if transaction.get('Merchant_Type') == 'Unknown':
                        score += weight
                elif feature == 'Device_ID':
                    # Higher score for new/unknown devices
                    device_id = transaction.get('Device_ID', '')
                    if 'New_Device' in device_id or 'Unknown_Device' in device_id:
                        score += weight
                elif feature == 'Location':
                    # Higher score for unusual locations
                    if transaction.get('Is_Unusual_Location', False):
                        score += weight
                elif feature == 'Txn_Count_Last_10_Min':
                    # Higher score for frequent transactions
                    if transaction.get('Txn_Count_Last_10_Min', 0) > 5:
                        score += weight
                elif feature == 'Transaction_Type':
                    # Base score for transaction type
                    score += weight * 0.5
                elif feature == 'Receiver_ID':
                    # Higher score for suspicious receivers
                    if 'Suspicious' in transaction.get('Receiver_ID', ''):
                        score += weight
            
            fraud_scores[fraud_type] = score
        
        # Return the fraud type with highest score
        if fraud_scores:
            return max(fraud_scores.items(), key=lambda x: x[1])[0]
        return "Unknown Fraud Type"
    
    def _process_transaction(self, txn_df):
        """
        Process a transaction DataFrame through feature engineering.
        
        Args:
            txn_df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Processed features
        """
        try:
            # Ensure we have a DataFrame
            if not isinstance(txn_df, pd.DataFrame):
                txn_df = pd.DataFrame([txn_df])
            
            # Convert DataFrame to dictionary for feature engineering
            transaction = txn_df.iloc[0].to_dict()
            
            # Process through feature engineering pipeline
            X = self.feature_engineer.process_single_transaction(transaction)
            
            # Verify feature names match model expectations
            missing_features = set(self.feature_engineer.encoded_cols) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Ensure features are in the correct order
            X = X[self.feature_engineer.encoded_cols]
            
            return X
            
        except Exception as e:
            logging.error(f"Error processing transaction: {str(e)}")
            # Return a DataFrame with default values
            return pd.DataFrame([[0] * len(self.feature_engineer.encoded_cols)],
                              columns=self.feature_engineer.encoded_cols)
    
    def predict_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict whether a transaction is fraudulent.
        
        Args:
            transaction (dict): Transaction data
            
        Returns:
            dict: Prediction result
        """
        start_time = time.time()
        
        try:
            # Create a copy to avoid modifying the original
            transaction_copy = transaction.copy()
            
            # Ensure Timestamp is properly formatted
            if isinstance(transaction_copy.get('Timestamp'), str):
                transaction_copy['Timestamp'] = pd.to_datetime(transaction_copy['Timestamp'])
            elif transaction_copy.get('Timestamp') is None:
                transaction_copy['Timestamp'] = datetime.now()
            
            # Update transaction history
            txn_count = self.update_history(
                transaction_copy.get('Sender_ID', 'Unknown'),
                transaction_copy['Timestamp']
            )
            transaction_copy['Recent_Txn_Count'] = txn_count
            
            # Process transaction data
            X = self._process_transaction(transaction_copy)
            
            # Make prediction
            fraud_prob = self.model.predict_proba(X)[0][1]
            is_fraud = fraud_prob >= self.threshold
            
            # Format result
            result = {
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_prob),
                'fraud_type': self._infer_fraud_type(transaction_copy, fraud_prob) if is_fraud else None,
                'processing_time': time.time() - start_time
            }
            
            # Add to history
            self.predictions_history.append({
                'timestamp': datetime.now().isoformat(),
                'transaction': transaction_copy,
                'prediction': result
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing transaction: {str(e)}")
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'fraud_type': None,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def save_predictions(self, output_path: str):
        """Save prediction history to a JSON file."""
        # Convert predictions to JSON-serializable format
        json_predictions = []
        for pred in self.predictions_history:
            json_pred = {
                'timestamp': pred['timestamp'],
                'transaction': {
                    k: v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v
                    for k, v in pred['transaction'].items()
                },
                'prediction': pred['prediction']
            }
            json_predictions.append(json_pred)
            
        with open(output_path, 'w') as f:
            json.dump(json_predictions, f, indent=2)
    
    def get_fraud_type_summary(self) -> Dict[str, int]:
        """Get summary of detected fraud types."""
        fraud_types = {}
        for pred in self.predictions_history:
            if pred.get('is_fraud'):
                fraud_type = pred.get('fraud_type', 'Unknown')
                fraud_types[fraud_type] = fraud_types.get(fraud_type, 0) + 1
        return fraud_types 