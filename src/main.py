import os
import sys
import pandas as pd
import time
import json
from datetime import datetime, timezone

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_generator import TransactionDataGenerator
from src.utils.predictor import RealTimePredictor

def generate_data_if_needed(data_path="data/fraud_transactions.csv", n_transactions=10000):
    """Generate synthetic data if it doesn't exist."""
    if not os.path.exists(data_path):
        print(f"Generating synthetic data at {data_path}...")
        generator = TransactionDataGenerator(n_transactions=n_transactions)
        generator.generate_data()
        generator.save_data(data_path)
    else:
        print(f"Using existing data at {data_path}")

def train_model_if_needed(data_path="data/fraud_transactions.csv",
                         model_path="models/fraud_detection_model.pkl",
                         feature_engineer_path="models/feature_engineer.pkl"):
    """Train the model if it doesn't exist."""
    if not os.path.exists(model_path) or not os.path.exists(feature_engineer_path):
        print("Training new model...")
        from src.models.train_model import train_model
        train_model(
            data_path=data_path,
            model_output_path=model_path,
            feature_engineer_output_path=feature_engineer_path,
            output_dir="models/evaluation"
        )
    else:
        print(f"Using existing model at {model_path}")

def generate_test_scenarios():
    """Generate test scenarios for different fraud types."""
    # Use a string timestamp format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    scenarios = [
        # UPI Frauds
        {
            "Transaction_ID": "Test_Phishing",
            "Timestamp": current_time,
            "Amount": 6500,  # Very high amount
            "Sender_ID": "User_1",
            "Receiver_ID": "Suspicious_001",
            "Transaction_Type": "UPI",
            "Merchant_Type": "Unknown",
            "Device_ID": "Device_1",
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        {
            "Transaction_ID": "Test_QR_Scam",
            "Timestamp": current_time,
            "Amount": 2500,  # Medium-high amount
            "Sender_ID": "User_1",
            "Receiver_ID": "Merchant_123",
            "Transaction_Type": "UPI",
            "Merchant_Type": "Unknown",
            "Device_ID": "Device_1",
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        {
            "Transaction_ID": "Test_SIM_Swap",
            "Timestamp": current_time,
            "Amount": 4000,  # High amount
            "Sender_ID": "User_1",
            "Receiver_ID": "Merchant_456",
            "Transaction_Type": "UPI",
            "Merchant_Type": "Unknown",  # Changed to Unknown
            "Device_ID": "New_Device_123",
            "Location": "-34.6037, -58.3816"  # Buenos Aires
        },
        {
            "Transaction_ID": "Test_Fake_UPI_App",
            "Timestamp": current_time,
            "Amount": 3500,  # Medium-high amount
            "Sender_ID": "User_1",
            "Receiver_ID": "Merchant_789",
            "Transaction_Type": "UPI",
            "Merchant_Type": "Unknown",  # Changed to Unknown
            "Device_ID": "Unknown_Device_456",
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        {
            "Transaction_ID": "Test_Small_Testing",
            "Timestamp": current_time,
            "Amount": 5,  # Very small amount
            "Sender_ID": "User_1",
            "Receiver_ID": "Merchant_101",
            "Transaction_Type": "UPI",
            "Merchant_Type": "Unknown",
            "Device_ID": "Device_1",
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        
        # Card Frauds
        {
            "Transaction_ID": "Test_Card_Skimming",
            "Timestamp": current_time,
            "Amount": 8500,  # Very high amount
            "Sender_ID": "User_2",
            "Receiver_ID": "Merchant_202",
            "Transaction_Type": "Card",
            "Merchant_Type": "Retail_Physical",
            "Device_ID": "Unknown_Device_789",  # Unknown device
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        {
            "Transaction_ID": "Test_Data_Breach",
            "Timestamp": current_time,
            "Amount": 4500,  # High amount
            "Sender_ID": "User_2",
            "Receiver_ID": "Merchant_303",
            "Transaction_Type": "Card",
            "Merchant_Type": "Online_Retail",
            "Device_ID": "Unknown_Device_101",  # Unknown device
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        {
            "Transaction_ID": "Test_Unusual_Location",
            "Timestamp": current_time,
            "Amount": 6000,  # High amount
            "Sender_ID": "User_2",
            "Receiver_ID": "Merchant_404",
            "Transaction_Type": "Card",
            "Merchant_Type": "Online_Retail",
            "Device_ID": "Device_2",
            "Location": "35.6762, 139.6503"  # Tokyo
        },
        {
            "Transaction_ID": "Test_CNP_Fraud",
            "Timestamp": current_time,
            "Amount": 9500,  # Very high amount
            "Sender_ID": "User_2",
            "Receiver_ID": "Merchant_505",
            "Transaction_Type": "Card",
            "Merchant_Type": "Online_Retail",
            "Device_ID": "Unknown_Device_202",  # Unknown device
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        
        # Legitimate Transactions
        {
            "Transaction_ID": "Test_Legitimate_UPI",
            "Timestamp": current_time,
            "Amount": 250,  # Normal amount
            "Sender_ID": "User_3",
            "Receiver_ID": "Merchant_606",
            "Transaction_Type": "UPI",
            "Merchant_Type": "Food",
            "Device_ID": "Device_3",
            "Location": "12.9716, 77.5946"  # Bangalore
        },
        {
            "Transaction_ID": "Test_Legitimate_Card",
            "Timestamp": current_time,
            "Amount": 1200,  # Normal amount
            "Sender_ID": "User_3",
            "Receiver_ID": "Merchant_707",
            "Transaction_Type": "Card",
            "Merchant_Type": "Entertainment",
            "Device_ID": "Device_3",
            "Location": "12.9716, 77.5946"  # Bangalore
        }
    ]
    
    return scenarios

def simulate_real_time_detection(model_path="models/fraud_detection_model.pkl",
                                feature_engineer_path="models/feature_engineer.pkl"):
    """Simulate real-time fraud detection."""
    print("Initializing real-time fraud detection system...")
    predictor = RealTimePredictor(model_path, feature_engineer_path)
    
    # Generate test scenarios
    scenarios = generate_test_scenarios()
    
    print("\n=== Real-Time Fraud Detection Demo ===")
    print(f"Processing {len(scenarios)} transactions...\n")
    
    # Process each transaction
    for i, transaction in enumerate(scenarios):
        print(f"Transaction {i+1}/{len(scenarios)}: {transaction['Transaction_ID']}")
        
        # Simulate network latency
        time.sleep(0.5)
        
        # Predict
        result = predictor.predict_transaction(transaction)
        
        # Display result
        status = "FRAUD DETECTED ❌" if result['is_fraud'] else "Legitimate ✓"
        print(f"  Amount: {transaction['Amount']:.2f} | {transaction['Transaction_Type']} | {status}")
        
        if result['is_fraud']:
            print(f"  Fraud Type: {result['fraud_type']}")
            print(f"  Probability: {result['fraud_probability']:.4f}")
        
        print(f"  Processing Time: {result['processing_time']:.2f} ms")
        print()
    
    # Save predictions
    predictor.save_predictions("data/predictions.json")
    
    # Display summary
    print("\n=== Fraud Detection Summary ===")
    total_txns = len(scenarios)
    fraud_txns = sum(1 for p in predictor.predictions_history if p['prediction']['is_fraud'])
    print(f"Total Transactions: {total_txns}")
    print(f"Fraudulent Transactions: {fraud_txns} ({fraud_txns/total_txns*100:.1f}%)")
    print("Fraud Types Detected:")
    
    # Count fraud types
    fraud_types = {}
    for p in predictor.predictions_history:
        if p['prediction']['is_fraud'] and p['prediction']['fraud_type']:
            fraud_type = p['prediction']['fraud_type']
            fraud_types[fraud_type] = fraud_types.get(fraud_type, 0) + 1
    
    # Display fraud type summary
    for fraud_type, count in fraud_types.items():
        print(f"  - {fraud_type}: {count}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Generate data if needed
    generate_data_if_needed()
    
    # Train model if needed
    train_model_if_needed()
    
    # Run real-time detection demo
    simulate_real_time_detection() 