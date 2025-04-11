import os
import sys
import json
import time
import random
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predictor import RealTimePredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_transaction(sender_id, base_time, transaction_type=None):
    """Generate a realistic transaction with dynamic patterns"""
    transaction = {
        'Sender_ID': sender_id,
        'Receiver_ID': f'REC_{random.randint(1000, 9999)}',
        'Timestamp': base_time + timedelta(minutes=random.randint(0, 30)),
        'Amount': float(random.uniform(10, 1000)),  # Ensure Amount is float
        'Location': f'{random.uniform(-90, 90):.6f}, {random.uniform(-180, 180):.6f}',
        'Device_ID': f'DEV_{random.randint(1000, 9999)}',
        'Transaction_Type': transaction_type or random.choice(['Transfer', 'Payment', 'Withdrawal']),
        'Merchant_Type': random.choice(['Retail', 'Online', 'ATM', 'POS']),
        'Is_Fraud': 0
    }
    
    # Add suspicious characteristics with 30% probability
    if random.random() < 0.3:
        transaction.update({
            'Amount': float(random.uniform(5000, 10000)),  # Ensure Amount is float
            'Location': '0.0, 0.0',  # Unusual location
            'Device_ID': 'DEV_Unknown',  # Unknown device
            'Merchant_Type': 'Unknown',  # Unknown merchant
            'Is_Fraud': 1
        })
    
    return transaction

def run_demo():
    """Run the fraud detection demo"""
    logger.info("Initializing fraud detection demo...")
    
    try:
        # Initialize predictor with model paths
        model_path = os.path.join('models', 'fraud_detection_model.pkl')
        feature_engineer_path = os.path.join('models', 'feature_engineer.pkl')
        
        # Verify model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_engineer_path):
            raise FileNotFoundError(f"Feature engineer file not found: {feature_engineer_path}")
        
        predictor = RealTimePredictor(model_path, feature_engineer_path)
        logger.info("Successfully initialized predictor")
        
        # Generate normal transactions
        base_time = datetime.now()
        transactions = []
        for i in range(10):
            transactions.append(generate_transaction(f'USER_{i}', base_time))
        
        # Add specific test cases
        test_cases = [
            {
                'Sender_ID': 'USER_TEST',
                'Receiver_ID': 'REC_SUSPICIOUS',
                'Timestamp': base_time + timedelta(minutes=5),
                'Amount': 5.0,  # Ensure Amount is float
                'Location': '0.0, 0.0',
                'Device_ID': 'DEV_Unknown',
                'Transaction_Type': 'Transfer',
                'Merchant_Type': 'Unknown',
                'Is_Fraud': 1
            },
            {
                'Sender_ID': 'USER_TEST',
                'Receiver_ID': 'REC_NORMAL',
                'Timestamp': base_time + timedelta(minutes=10),
                'Amount': 8000.0,  # Ensure Amount is float
                'Location': '19.0760, 72.8777',  # Mumbai coordinates
                'Device_ID': 'DEV_9999',
                'Transaction_Type': 'Payment',
                'Merchant_Type': 'Online',
                'Is_Fraud': 1
            }
        ]
        transactions.extend(test_cases)
        
        # Process transactions and collect predictions
        results = []
        for txn in transactions:
            try:
                prediction = predictor.predict_transaction(txn)
                results.append({
                    'Sender_ID': txn['Sender_ID'],
                    'Amount': float(txn['Amount']),  # Ensure Amount is float
                    'Is_Fraud': txn['Is_Fraud'],
                    'Predicted_Fraud': int(prediction.get('is_fraud', False)),
                    'Confidence': float(prediction.get('fraud_probability', 0.0)),
                    'Fraud_Type': prediction.get('fraud_type'),
                    'Processing_Time': float(prediction.get('processing_time', 0.0))
                })
                logger.info(f"Transaction from {txn['Sender_ID']}: "
                          f"Amount={txn['Amount']}, "
                          f"Fraud={prediction.get('is_fraud', False)}, "
                          f"Confidence={prediction.get('fraud_probability', 0.0):.2f}")
            except Exception as e:
                logger.error(f"Error processing transaction: {str(e)}")
                results.append({
                    'Sender_ID': txn['Sender_ID'],
                    'Amount': float(txn['Amount']),
                    'Is_Fraud': txn['Is_Fraud'],
                    'Error': str(e)
                })
        
        # Save results
        with open('demo_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        total_txns = len(results)
        detected_frauds = sum(1 for r in results if r.get('Predicted_Fraud', False))
        actual_frauds = sum(1 for r in results if r['Is_Fraud'])
        errors = sum(1 for r in results if 'Error' in r)
        
        print("\nDemo Summary:")
        print(f"Total Transactions: {total_txns}")
        print(f"Actual Fraudulent: {actual_frauds}")
        print(f"Detected Fraudulent: {detected_frauds}")
        print(f"Processing Errors: {errors}")
        if actual_frauds > 0:
            print(f"Detection Rate: {(detected_frauds/actual_frauds*100):.1f}%")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_demo() 