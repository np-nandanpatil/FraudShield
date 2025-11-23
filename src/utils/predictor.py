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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RealTimePredictor:
    """Real-time fraud prediction system with continuous learning"""

    def __init__(self, model_path, feature_engineer_path, threshold=0.3):
        """
        Initialize the real-time predictor.

        Args:
            model_path (str): Path to the trained model
            feature_engineer_path (str): Path to the saved feature engineer
            threshold (float): Decision threshold for fraud classification
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(feature_engineer_path):
                raise FileNotFoundError(
                    f"Feature engineer file not found: {feature_engineer_path}"
                )

            self.model = joblib.load(model_path)
            self.feature_engineer = joblib.load(feature_engineer_path)
            self.threshold = threshold
            self.predictions_history = []
            self.user_history = defaultdict(list)
            self.history_lock = threading.Lock()
            self.predictions_lock = threading.Lock()

            # Continuous learning parameters
            self.model_path = model_path
            self.feature_engineer_path = feature_engineer_path
            self.training_data = []
            self.training_data_lock = threading.Lock()
            self.last_retrain_time = datetime.now()
            self.retrain_interval = timedelta(hours=24)  # Retrain every 24 hours
            self.min_samples_for_retraining = 50  # Minimum number of samples needed to retrain

            # Start continuous learning thread
            self.continuous_learning_enabled = True
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
            self.learning_thread.start()

            logger.info("RealTimePredictor initialized successfully with continuous learning")

        except Exception as e:
            logger.error(f"Error initializing RealTimePredictor: {str(e)}")
            raise

    def _continuous_learning_loop(self):
        """Background thread for continuous model retraining"""
        logger.info("Starting continuous learning thread")
        while self.continuous_learning_enabled:
            try:
                time.sleep(60)  # Check every minute
                
                now = datetime.now()
                time_since_last_retrain = now - self.last_retrain_time
                
                # Check if it's time to retrain and we have enough data
                with self.training_data_lock:
                    sufficient_data = len(self.training_data) >= self.min_samples_for_retraining
                
                if time_since_last_retrain >= self.retrain_interval and sufficient_data:
                    logger.info(f"Starting model retraining with {len(self.training_data)} samples")
                    self._retrain_model()
                    self.last_retrain_time = now
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {str(e)}")
    
    def _retrain_model(self):
        """Retrain the model with new data"""
        try:
            # Create a copy of current training data
            with self.training_data_lock:
                training_data = self.training_data.copy()
                
            if not training_data:
                logger.warning("No training data available for retraining")
                return
            
            # Create DataFrame from collected data
            df = pd.DataFrame(training_data)
            
            # Check if we have both positive and negative examples
            if df['Is_Fraud'].nunique() < 2:
                logger.warning("Training data doesn't have both fraud and non-fraud examples")
                return
            
            # Process features using the feature engineer
            X = self.feature_engineer.process_transactions(df)
            y = df['Is_Fraud'].astype(int)
            
            # Retrain the model (partial fit)
            if hasattr(self.model, 'partial_fit'):
                # For models that support incremental learning
                self.model.partial_fit(X, y)
            else:
                # Load existing training data and combine with new data
                try:
                    # Try to load the original training data
                    original_data_path = 'data/original_training_data.csv'
                    if os.path.exists(original_data_path):
                        original_df = pd.read_csv(original_data_path)
                        combined_df = pd.concat([original_df, df], ignore_index=True)
                    else:
                        combined_df = df
                        
                    # Process combined data
                    combined_X = self.feature_engineer.process_transactions(combined_df)
                    combined_y = combined_df['Is_Fraud'].astype(int)
                    
                    # Retrain model from scratch
                    self.model.fit(combined_X, combined_y)
                except Exception as e:
                    logger.error(f"Error loading original training data: {str(e)}")
                    # Fallback: just use new data
                    self.model.fit(X, y)
            
            # Save updated model
            temp_model_path = f"{self.model_path}.temp"
            joblib.dump(self.model, temp_model_path)
            
            # Atomically replace the old model file
            os.replace(temp_model_path, self.model_path)
            
            logger.info(f"Model successfully retrained and saved to {self.model_path}")
            
            # Clear training data after successful retraining
            with self.training_data_lock:
                # Keep some recent data for next retraining
                recent_count = min(100, len(self.training_data) // 5)
                self.training_data = self.training_data[-recent_count:]
                
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
    
    def add_to_training_data(self, transaction, is_fraud):
        """Add a transaction to the training data for continuous learning
        
        Args:
            transaction (dict): Transaction data
            is_fraud (bool): Whether the transaction is fraudulent
        """
        try:
            # Create a copy to avoid modifying the original
            data_point = transaction.copy()
            
            # Ensure necessary fields
            data_point['Is_Fraud'] = bool(is_fraud)
            if 'Timestamp' not in data_point or not isinstance(data_point['Timestamp'], datetime):
                data_point['Timestamp'] = datetime.now()
                
            # Add to training data with thread safety
            with self.training_data_lock:
                self.training_data.append(data_point)
                
                # Limit size to prevent memory issues
                if len(self.training_data) > 10000:
                    self.training_data = self.training_data[-5000:]
                    
            logger.debug(f"Added transaction to training data, new size: {len(self.training_data)}")
            
        except Exception as e:
            logger.error(f"Error adding to training data: {str(e)}")
    
    def feedback(self, transaction_id, actual_fraud, feedback_source="user"):
        """Process feedback on a prediction, used for continuous learning
        
        Args:
            transaction_id (str): Transaction ID
            actual_fraud (bool): Whether the transaction was actually fraudulent
            feedback_source (str): Source of the feedback (user/system/manual)
        
        Returns:
            bool: Whether the feedback was successfully processed
        """
        try:
            logger.info(f"Received feedback for transaction {transaction_id}: is_fraud={actual_fraud}, source={feedback_source}")
            
            # Find the transaction in prediction history
            transaction_data = None
            prediction = None
            
            with self.predictions_lock:
                for item in self.predictions_history:
                    if str(item.get('transaction_id', '')) == str(transaction_id):
                        transaction_data = item.get('transaction', None)
                        prediction = item.get('is_fraud', None)
                        logger.info(f"Found transaction {transaction_id} in prediction history")
                        break
            
            if not transaction_data:
                logger.warning(f"Transaction {transaction_id} not found in prediction history")
                
                # If we can't find in predictions_history, create a minimal transaction object
                # This handles cases where the server restarted or predictions were cleared
                transaction_data = {
                    'Transaction_ID': transaction_id,
                    'Timestamp': datetime.now(),
                    'Amount': 0,  # We don't know the amount
                    'Transaction_Type': 'Unknown',
                    'Is_Fraud': actual_fraud,
                    'Fraud_Type': 'Admin Flagged' if actual_fraud else '-'
                }
                
                logger.info(f"Created minimal transaction data for feedback on {transaction_id}")
                
            # Check if prediction was correct
            if prediction is not None and prediction != actual_fraud:
                logger.info(f"Incorrect prediction for transaction {transaction_id}. "
                           f"Predicted: {prediction}, Actual: {actual_fraud}")
                
            # Add to training data
            try:
                self.add_to_training_data(transaction_data, actual_fraud)
                logger.info(f"Added transaction {transaction_id} to training data")
            except Exception as e:
                logger.error(f"Error adding to training data: {str(e)}")
            
            # Log feedback success
            logger.info(f"Feedback successfully processed for transaction {transaction_id}: "
                       f"actual_fraud={actual_fraud}, source={feedback_source}")
            
            # Always return true - we don't want to fail feedback
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback for {transaction_id}: {str(e)}")
            # Still return True - we want the API to succeed even if there was an error
            return True
            
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

            # Clean up old users (no transactions in last 24 hours)
            old_cutoff = timestamp - timedelta(hours=24)
            self.user_history = {
                user: times
                for user, times in self.user_history.items()
                if any(t > old_cutoff for t in times)
            }

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
        # Get transaction type
        transaction_type = transaction.get("Transaction_Type", "Unknown")

        # Define which fraud types apply to which transaction types
        transaction_type_mapping = {
            "UPI": [
                "Phishing Link",
                "QR Code Scam",
                "SIM Swap",
                "Fake UPI App",
                "Small Testing",
                "Unusual Location",
            ],
            "Card": [
                "Card Skimming",
                "Data Breach Reuse",
                "CNP Fraud",
                "Unusual Location",
                "Small Testing",
            ],
        }

        # Default to all fraud types if transaction type is unknown
        applicable_fraud_types = transaction_type_mapping.get(
            transaction_type,
            list(transaction_type_mapping["UPI"] + transaction_type_mapping["Card"]),
        )

        # Feature weights for different fraud types
        fraud_weights = {
            "Phishing Link": {
                "Amount": 0.3,
                "Merchant_Type": 0.2,
                "Receiver_ID": 0.3,
                "Transaction_Type": 0.2,
            },
            "QR Code Scam": {
                "Amount": 0.3,
                "Merchant_Type": 0.4,
                "Transaction_Type": 0.3,
            },
            "SIM Swap": {"Device_ID": 0.4, "Location": 0.3, "Transaction_Type": 0.3},
            "Fake UPI App": {
                "Device_ID": 0.4,
                "Merchant_Type": 0.3,
                "Transaction_Type": 0.3,
            },
            "Small Testing": {
                "Amount": 0.4,
                "Txn_Count_Last_10_Min": 0.4,
                "Transaction_Type": 0.2,
            },
            "Card Skimming": {"Amount": 0.4, "Merchant_Type": 0.3, "Device_ID": 0.3},
            "Data Breach Reuse": {
                "Amount": 0.3,
                "Merchant_Type": 0.3,
                "Device_ID": 0.4,
            },
            "Unusual Location": {
                "Location": 0.5,
                "Amount": 0.3,
                "Transaction_Type": 0.2,
            },
            "CNP Fraud": {"Amount": 0.4, "Merchant_Type": 0.3, "Device_ID": 0.3},
        }

        # Calculate scores for each applicable fraud type
        fraud_scores = {}

        # Only consider fraud types applicable to this transaction type
        for fraud_type in applicable_fraud_types:
            if fraud_type not in fraud_weights:
                continue

            weights = fraud_weights[fraud_type]
            score = 0
            for feature, weight in weights.items():
                if feature == "Amount":
                    # Higher score for unusual amounts
                    amount = float(transaction.get("Amount", 0))
                    if amount > 5000:  # Very high amount
                        score += weight
                    elif amount < 10:  # Very small amount
                        score += weight * 0.8
                elif feature == "Merchant_Type":
                    # Higher score for unknown merchants
                    if transaction.get("Merchant_Type") == "Unknown":
                        score += weight
                elif feature == "Device_ID":
                    # Higher score for new/unknown devices
                    device_id = transaction.get("Device_ID", "")
                    if "New_Device" in device_id or "Unknown_Device" in device_id:
                        score += weight
                elif feature == "Location":
                    # Higher score for unusual locations
                    if transaction.get("Is_Unusual_Location", False):
                        score += weight
                elif feature == "Txn_Count_Last_10_Min":
                    # Higher score for frequent transactions
                    if transaction.get("Txn_Count_Last_10_Min", 0) > 5:
                        score += weight
                elif feature == "Transaction_Type":
                    # Base score for transaction type
                    score += weight * 0.5
                elif feature == "Receiver_ID":
                    # Higher score for suspicious receivers
                    if "Suspicious" in transaction.get("Receiver_ID", ""):
                        score += weight

            fraud_scores[fraud_type] = score

        # Return the fraud type with highest score
        if fraud_scores:
            return max(fraud_scores.items(), key=lambda x: x[1])[0]

        # If no applicable fraud type has a score, return a generic fraud type based on transaction type
        if transaction_type == "Card":
            return "Suspicious Card Transaction"
        elif transaction_type == "UPI":
            return "Suspicious UPI Transaction"
        else:
            return "Unknown Fraud Type"

    def _process_transaction(self, txn_df):
        """Process a single transaction for prediction"""
        try:
            # Convert to DataFrame if not already
            if not isinstance(txn_df, pd.DataFrame):
                txn_df = pd.DataFrame([txn_df])
            
            # Process features using the feature engineer
            # Support multiple FeatureEngineer interfaces for backward compatibility
            if hasattr(self.feature_engineer, 'process_transactions'):
                # New interface in src/utils/feature_engineer.py
                X = self.feature_engineer.process_transactions(txn_df)
            elif hasattr(self.feature_engineer, 'process_single_transaction'):
                # Older interface in src/features/feature_engineering.py
                # Expects a dict and returns a processed DataFrame
                X = self.feature_engineer.process_single_transaction(txn_df.iloc[0].to_dict())
            elif hasattr(self.feature_engineer, 'process_data'):
                # Fallback to generic process_data(df, fit=False) -> (X, y)
                processed = self.feature_engineer.process_data(txn_df, fit=False)
                # process_data may return (X, y) or just X
                if isinstance(processed, tuple) and len(processed) >= 1:
                    X = processed[0]
                else:
                    X = processed
            else:
                raise AttributeError("Loaded FeatureEngineer has no compatible processing method")
            
            # Ensure no NaN values remain
            if X.isna().any().any():
                X = X.fillna(0)
            
            return X
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            raise

    def predict_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud for a single transaction"""
        start_time = time.time()
        
        try:
            # Convert transaction to DataFrame
            txn_df = pd.DataFrame([transaction])
            
            # Process transaction
            X = self._process_transaction(txn_df)
            
            # Predict
            fraud_prob = self.model.predict_proba(X)[0][1]
            is_fraud = fraud_prob >= self.threshold
            
            # Infer fraud type
            fraud_type = self._infer_fraud_type(transaction, fraud_prob)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Create result
            result = {
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_prob),
                'fraud_type': fraud_type,
                'processing_time': processing_time
            }
            
            # Add to prediction history
            with self.predictions_lock:
                self.predictions_history.append({
                    'transaction_id': transaction.get('Transaction_ID', 'unknown'),
                    'transaction': transaction,
                    'prediction': result
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting transaction: {str(e)}")
            # Return default result with processing time
            processing_time = (time.time() - start_time) * 1000
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'fraud_type': 'Error',
                'processing_time': processing_time
            }

    def save_predictions(self, output_path: str):
        """Save prediction history to a JSON file."""
        # Convert predictions to JSON-serializable format
        json_predictions = []
        for pred in self.predictions_history:
            # Prepare transaction data for JSON serialization
            try:
                transaction_data = {}
                for k, v in pred.get("transaction", {}).items():
                    if isinstance(v, (datetime, pd.Timestamp)):
                        transaction_data[k] = v.isoformat()
                    elif isinstance(v, (int, float, str, bool, type(None))):
                        transaction_data[k] = v
                    else:
                        # Convert other types to string to ensure serializability
                        transaction_data[k] = str(v)
                
                # Build JSON-compatible prediction record
                json_pred = {
                    "transaction_id": pred.get("transaction_id", "Unknown"),
                    "timestamp": pred.get("timestamp").isoformat() if isinstance(pred.get("timestamp"), (datetime, pd.Timestamp)) else str(pred.get("timestamp")),
                    "is_fraud": pred.get("is_fraud", False),
                    "fraud_probability": float(pred.get("fraud_probability", 0)),
                    "fraud_type": pred.get("fraud_type", ""),
                    "transaction": transaction_data
                }
                json_predictions.append(json_pred)
            except Exception as e:
                logger.error(f"Error serializing prediction: {str(e)}")
                continue

        try:
            with open(output_path, "w") as f:
                json.dump(json_predictions, f, indent=2)
            logger.info(f"Successfully saved {len(json_predictions)} predictions to {output_path}")
        except Exception as e:
            logger.error(f"Error saving predictions to {output_path}: {str(e)}")

    def get_fraud_type_summary(self) -> Dict[str, int]:
        """Get summary of detected fraud types."""
        fraud_types = {}
        for pred in self.predictions_history:
            if pred.get("is_fraud"):
                fraud_type = pred.get("fraud_type", "Unknown")
                fraud_types[fraud_type] = fraud_types.get(fraud_type, 0) + 1
        return fraud_types
