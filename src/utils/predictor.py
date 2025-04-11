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
            logger.error(f"Error processing transaction: {str(e)}")
            # Return a DataFrame with default values
            return pd.DataFrame(
                [[0] * len(self.feature_engineer.encoded_cols)],
                columns=self.feature_engineer.encoded_cols,
            )

    def predict_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud for a transaction.

        Args:
            transaction (dict): Transaction data

        Returns:
            dict: Prediction results
        """
        try:
            # Basic input validation
            if not transaction:
                logger.error("Empty transaction data received")
                return {
                    "is_fraud": True,
                    "fraud_probability": 0.95,
                    "fraud_type": "Empty Transaction Data",
                }
            
            # Validate transaction has required fields
            required_fields = ["Transaction_Type", "Amount"]
            missing_fields = [field for field in required_fields if field not in transaction]
            
            if missing_fields:
                logger.error(f"Missing required transaction fields: {missing_fields}")
                return {
                    "is_fraud": True,
                    "fraud_probability": 0.9,
                    "fraud_type": f"Missing Required Fields: {', '.join(missing_fields)}",
                }

            # Make a copy to avoid modifying the original
            transaction = transaction.copy()
            
            # Log the transaction details for debugging (mask sensitive data)
            sanitized_transaction = transaction.copy()
            if "Sender_ID" in sanitized_transaction and sanitized_transaction["Transaction_Type"] == "Card":
                sanitized_transaction["Sender_ID"] = "****" + str(sanitized_transaction["Sender_ID"])[-4:]
            logger.info(f"Processing transaction: {json.dumps(sanitized_transaction, default=str)}")

            # Handle UPI transactions with special formatting
            if transaction["Transaction_Type"] == "UPI":
                try:
                    # Process sender ID for UPI transactions
                    sender_id = transaction.get("Sender_ID", "")
                    if sender_id:
                        # Validate UPI ID format (username@provider)
                        if "@" in sender_id:
                            parts = sender_id.split("@")
                            if len(parts) == 2:
                                username, provider = parts
                                
                                # Fix common provider issues
                                if not provider or len(provider) < 2:
                                    # Default to a standard provider if missing
                                    provider = "okaxis"
                                elif "." in provider:
                                    # Handle multiple dots in provider (e.g., "user@123.oksbi")
                                    # Take the last part as the actual provider
                                    provider_parts = provider.split(".")
                                    provider = provider_parts[-1]
                                
                                # Reconstruct valid UPI ID
                                transaction["Sender_ID"] = f"{username}@{provider}"
                            else:
                                # Too many @ symbols, use first part with default provider
                                transaction["Sender_ID"] = f"{parts[0]}@okaxis"
                        else:
                            # No @ symbol, append default provider
                            transaction["Sender_ID"] = f"{sender_id}@okaxis"
                            
                        logger.info(f"Processed UPI ID: {sender_id} -> {transaction['Sender_ID']}")
                except Exception as e:
                    logger.error(f"Error processing UPI Sender_ID: {sender_id} - {str(e)}")
                    # Use a default value if cannot process
                    transaction["Sender_ID"] = "unknown@upi"
                
                # Also handle Receiver_ID for UPI transactions
                try:
                    receiver_id = transaction.get("Receiver_ID", "")
                    if receiver_id and "@" in receiver_id:
                        parts = receiver_id.split("@")
                        if len(parts) == 2:
                            username, provider = parts
                            
                            # Fix common provider issues
                            if not provider or len(provider) < 2:
                                provider = "upi"
                            elif "." in provider:
                                provider_parts = provider.split(".")
                                provider = provider_parts[-1]
                            
                            # Reconstruct valid UPI ID
                            transaction["Receiver_ID"] = f"{username}@{provider}"
                except Exception as e:
                    logger.error(f"Error processing UPI Receiver_ID: {receiver_id} - {str(e)}")
                    # Don't override Receiver_ID if it fails processing

            # Also handle special processing for Card transactions
            if transaction["Transaction_Type"] == "Card":
                try:
                    # Validate and sanitize card numbers
                    card_number = transaction.get("Sender_ID", "")
                    
                    # If it looks like a masked card number (has * characters)
                    if card_number and "*" in card_number:
                        # Ensure proper format, but keep the masking
                        digits_only = ''.join(c for c in card_number if c.isdigit() or c == '*')
                        if len(digits_only) > 8:  # Minimum reasonable length for masked card
                            transaction["Sender_ID"] = digits_only
                except Exception as e:
                    logger.error(f"Error processing Card number: {str(e)}")

            # Convert timestamp string to datetime if needed
            if "Timestamp" in transaction and isinstance(transaction["Timestamp"], str):
                try:
                    transaction["Timestamp"] = datetime.fromisoformat(
                        transaction["Timestamp"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    # If timestamp can't be parsed, use current time
                    transaction["Timestamp"] = datetime.now()
            elif "Timestamp" not in transaction:
                transaction["Timestamp"] = datetime.now()

            # Add transaction count feature
            sender = transaction.get("Sender_ID", "Unknown")
            transaction["Txn_Count_Last_10_Min"] = self.update_history(
                sender, transaction["Timestamp"]
            )

            # Convert to DataFrame
            txn_df = pd.DataFrame([transaction])

            # Convert amount to float if needed
            if "Amount" in txn_df.columns and not pd.api.types.is_numeric_dtype(
                txn_df["Amount"]
            ):
                txn_df["Amount"] = pd.to_numeric(txn_df["Amount"], errors="coerce")

            # Prepare features
            X = self._process_transaction(txn_df)

            # Make prediction
            fraud_probability = self.model.predict_proba(X)[0, 1]
            is_fraud = fraud_probability > self.threshold

            # Infer fraud type if predicted as fraud
            fraud_type = ""
            if is_fraud:
                fraud_type = self._infer_fraud_type(transaction, fraud_probability)

            # Store transaction for continuous learning
            stored_transaction = {
                "transaction_id": transaction.get("Transaction_ID", "Unknown"),
                "timestamp": transaction["Timestamp"],
                "is_fraud": is_fraud,
                "fraud_probability": fraud_probability,
                "fraud_type": fraud_type if is_fraud else "",
                "transaction": transaction
            }
            
            # Log prediction
            with self.predictions_lock:
                self.predictions_history.append(stored_transaction)
                
                # Keep only last 1000 predictions to prevent memory issues
                if len(self.predictions_history) > 1000:
                    self.predictions_history = self.predictions_history[-1000:]

            # Auto-add to training data for obvious cases (very high/low probability)
            # This serves as "system feedback" for clear cases
            if fraud_probability > 0.9 or fraud_probability < 0.1:
                self.add_to_training_data(transaction, is_fraud)

            # Return results
            return {
                "is_fraud": is_fraud,
                "fraud_probability": fraud_probability,
                "fraud_type": fraud_type if is_fraud else "",
            }

        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            return {
                "is_fraud": True,  # Fail safe - treat as fraud if we can't process
                "fraud_probability": 0.9,
                "fraud_type": "Processing Error",
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
