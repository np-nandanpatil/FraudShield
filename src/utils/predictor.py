import pandas as pd
import numpy as np
import joblib
import time
import os
import json
from datetime import datetime

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
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        self.threshold = threshold
        self.predictions_history = []
        
    def predict_transaction(self, transaction):
        """
        Predict whether a transaction is fraudulent.
        
        Args:
            transaction (dict): Transaction data
            
        Returns:
            dict: Prediction result
        """
        start_time = time.time()
        
        # Create a copy to avoid modifying the original
        transaction_copy = transaction.copy()
        
        # Ensure Timestamp is properly formatted
        if isinstance(transaction_copy.get('Timestamp'), str):
            transaction_copy['Timestamp'] = pd.to_datetime(transaction_copy['Timestamp'])
        elif transaction_copy.get('Timestamp') is None:
            transaction_copy['Timestamp'] = datetime.now()
            
        # Convert to DataFrame
        txn_df = pd.DataFrame([transaction_copy])
        
        # Process transaction data
        X = self._process_transaction(txn_df)
        
        # Make prediction
        _, fraud_prob = self.model.predict(X)
        is_fraud = fraud_prob[0] >= self.threshold
        
        # Format result
        result = {
            'Transaction_ID': transaction.get('Transaction_ID', 'Unknown'),
            'Timestamp': transaction_copy.get('Timestamp').strftime('%Y-%m-%d %H:%M:%S') if hasattr(transaction_copy.get('Timestamp'), 'strftime') else str(transaction_copy.get('Timestamp')),
            'Amount': transaction.get('Amount', 0),
            'Sender_ID': transaction.get('Sender_ID', 'Unknown'),
            'Transaction_Type': transaction.get('Transaction_Type', 'Unknown'),
            'Is_Fraud': bool(is_fraud),
            'Fraud_Probability': float(fraud_prob[0]),
            'Fraud_Type': self._detect_fraud_type(transaction, fraud_prob[0]) if is_fraud else "Not Fraud",
            'Prediction_Time_ms': (time.time() - start_time) * 1000
        }
        
        # Store prediction in history
        self.predictions_history.append(result)
        
        return result
    
    def _process_transaction(self, txn_df):
        """Process transaction data for prediction."""
        # Apply feature engineering
        from src.features.feature_engineering import process_single_transaction
        # Convert DataFrame to dictionary if needed
        if isinstance(txn_df, pd.DataFrame):
            transaction = txn_df.iloc[0].to_dict()
        else:
            transaction = txn_df
        X = process_single_transaction(transaction, self.feature_engineer)
        return X
    
    def _detect_fraud_type(self, transaction, fraud_prob):
        """
        Detect the specific type of fraud.
        
        This is a simplified detection - in a real system, you would likely
        use a more sophisticated approach or a dedicated model.
        """
        if fraud_prob < self.threshold:
            return "Not Fraud"
            
        # Extract transaction details
        txn_type = transaction.get('Transaction_Type', '')
        amount = transaction.get('Amount', 0)
        merchant_type = transaction.get('Merchant_Type', '')
        receiver_id = transaction.get('Receiver_ID', '')
        device_id = transaction.get('Device_ID', '')
        location = transaction.get('Location', '')
        
        # Simplified fraud type detection based on transaction characteristics
        if txn_type == "UPI":
            if "Suspicious" in receiver_id:
                return "Phishing Link"
            elif merchant_type == "Unknown":
                return "QR Code Scam"
            elif "New_Device" in device_id or "Unknown_Device" in device_id:
                if "-34.6037, -58.3816" in location:  # Buenos Aires
                    return "SIM Swap Attack"
                else:
                    return "Fake UPI App"
            elif amount <= 10:
                return "Small Testing Transaction"
        else:  # Card
            if merchant_type == "Retail_Physical" and amount > 3000:
                return "Card Skimming"
            elif merchant_type == "Online_Retail":
                if amount > 5000:
                    return "CNP Fraud"
                else:
                    return "Data Breach Reuse"
            elif "35.6762, 139.6503" in location:  # Tokyo
                return "Unusual Location/Activity"
        
        return "Unknown Fraud"
    
    def get_recent_predictions(self, n=10):
        """Get the most recent predictions."""
        return self.predictions_history[-n:] if self.predictions_history else []
        
    def save_predictions(self, output_path):
        """Save prediction history to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.predictions_history, f, indent=2)
        print(f"Predictions saved to {output_path}")
        
    def load_predictions(self, input_path):
        """Load prediction history from JSON file."""
        with open(input_path, 'r') as f:
            self.predictions_history = json.load(f)
        print(f"Loaded {len(self.predictions_history)} predictions from {input_path}")
        
    def get_fraud_type_summary(self):
        """Get a summary of detected fraud types."""
        fraud_predictions = [p for p in self.predictions_history if p['Is_Fraud']]
        fraud_types = [p['Fraud_Type'] for p in fraud_predictions]
        
        if not fraud_types:
            return {"No frauds detected": 0}
            
        # Count occurrences of each fraud type
        fraud_counts = {}
        for fraud_type in fraud_types:
            if fraud_type in fraud_counts:
                fraud_counts[fraud_type] += 1
            else:
                fraud_counts[fraud_type] = 1
                
        return fraud_counts 