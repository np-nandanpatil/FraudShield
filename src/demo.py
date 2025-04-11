import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.03):
    """Generate synthetic transaction data with fraud patterns."""
    np.random.seed(42)
    
    # Create legitimate transactions
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    legitimate_data = {
        'amount': np.random.uniform(10, 1000, n_legitimate),
        'is_new_device': np.random.choice([0, 1], n_legitimate, p=[0.9, 0.1]),
        'is_unusual_location': np.random.choice([0, 1], n_legitimate, p=[0.95, 0.05]),
        'is_first_time_receiver': np.random.choice([0, 1], n_legitimate, p=[0.7, 0.3]),
        'is_suspicious_receiver': np.zeros(n_legitimate),
        'transaction_frequency': np.random.uniform(0.1, 10, n_legitimate),
        'time_since_last_txn': np.random.uniform(60, 10000, n_legitimate),
        'is_weekend': np.random.choice([0, 1], n_legitimate, p=[0.7, 0.3]),
        'is_night': np.random.choice([0, 1], n_legitimate, p=[0.8, 0.2]),
        'transaction_type_upi': np.random.choice([0, 1], n_legitimate, p=[0.5, 0.5]),
        'transaction_type_card': np.random.choice([0, 1], n_legitimate, p=[0.5, 0.5]),
        'is_fraud': np.zeros(n_legitimate),
        'fraud_type': ['Legitimate'] * n_legitimate
    }
    
    # Create different types of fraud
    
    # 1. Phishing Links (UPI)
    n_phishing = n_fraud // 9
    phishing_data = {
        'amount': np.random.uniform(1000, 5000, n_phishing),
        'is_new_device': np.random.choice([0, 1], n_phishing, p=[0.8, 0.2]),
        'is_unusual_location': np.random.choice([0, 1], n_phishing, p=[0.7, 0.3]),
        'is_first_time_receiver': np.ones(n_phishing),
        'is_suspicious_receiver': np.ones(n_phishing),
        'transaction_frequency': np.random.uniform(0.1, 2, n_phishing),
        'time_since_last_txn': np.random.uniform(100, 5000, n_phishing),
        'is_weekend': np.random.choice([0, 1], n_phishing),
        'is_night': np.random.choice([0, 1], n_phishing),
        'transaction_type_upi': np.ones(n_phishing),
        'transaction_type_card': np.zeros(n_phishing),
        'is_fraud': np.ones(n_phishing),
        'fraud_type': ['Phishing Link'] * n_phishing
    }
    
    # 2. QR Code Scams (UPI)
    n_qr = n_fraud // 9
    qr_data = {
        'amount': np.random.uniform(500, 2000, n_qr),
        'is_new_device': np.random.choice([0, 1], n_qr, p=[0.9, 0.1]),
        'is_unusual_location': np.random.choice([0, 1], n_qr, p=[0.9, 0.1]),
        'is_first_time_receiver': np.ones(n_qr),
        'is_suspicious_receiver': np.zeros(n_qr),
        'transaction_frequency': np.random.uniform(0.1, 2, n_qr),
        'time_since_last_txn': np.random.uniform(100, 5000, n_qr),
        'is_weekend': np.random.choice([0, 1], n_qr),
        'is_night': np.random.choice([0, 1], n_qr),
        'transaction_type_upi': np.ones(n_qr),
        'transaction_type_card': np.zeros(n_qr),
        'is_fraud': np.ones(n_qr),
        'fraud_type': ['QR Code Scam'] * n_qr
    }
    
    # 3. SIM Swap Attacks (UPI)
    n_sim = n_fraud // 9
    sim_data = {
        'amount': np.random.uniform(1000, 3000, n_sim),
        'is_new_device': np.ones(n_sim),
        'is_unusual_location': np.ones(n_sim),
        'is_first_time_receiver': np.random.choice([0, 1], n_sim),
        'is_suspicious_receiver': np.zeros(n_sim),
        'transaction_frequency': np.random.uniform(5, 20, n_sim),
        'time_since_last_txn': np.random.uniform(1, 60, n_sim),
        'is_weekend': np.random.choice([0, 1], n_sim),
        'is_night': np.random.choice([0, 1], n_sim),
        'transaction_type_upi': np.ones(n_sim),
        'transaction_type_card': np.zeros(n_sim),
        'is_fraud': np.ones(n_sim),
        'fraud_type': ['SIM Swap Attack'] * n_sim
    }
    
    # 4. Fake UPI Apps
    n_fake_app = n_fraud // 9
    fake_app_data = {
        'amount': np.random.uniform(1000, 3000, n_fake_app),
        'is_new_device': np.ones(n_fake_app),
        'is_unusual_location': np.random.choice([0, 1], n_fake_app),
        'is_first_time_receiver': np.random.choice([0, 1], n_fake_app),
        'is_suspicious_receiver': np.zeros(n_fake_app),
        'transaction_frequency': np.random.uniform(0.1, 5, n_fake_app),
        'time_since_last_txn': np.random.uniform(1, 1000, n_fake_app),
        'is_weekend': np.random.choice([0, 1], n_fake_app),
        'is_night': np.ones(n_fake_app),  # Often happens at night
        'transaction_type_upi': np.ones(n_fake_app),
        'transaction_type_card': np.zeros(n_fake_app),
        'is_fraud': np.ones(n_fake_app),
        'fraud_type': ['Fake UPI App'] * n_fake_app
    }
    
    # 5. Small Testing Transactions (UPI)
    n_small = n_fraud // 9
    small_data = {
        'amount': np.random.uniform(1, 10, n_small),
        'is_new_device': np.random.choice([0, 1], n_small),
        'is_unusual_location': np.random.choice([0, 1], n_small),
        'is_first_time_receiver': np.random.choice([0, 1], n_small),
        'is_suspicious_receiver': np.zeros(n_small),
        'transaction_frequency': np.random.uniform(10, 20, n_small),  # Many transactions
        'time_since_last_txn': np.random.uniform(1, 10, n_small),     # Short intervals
        'is_weekend': np.random.choice([0, 1], n_small),
        'is_night': np.random.choice([0, 1], n_small),
        'transaction_type_upi': np.ones(n_small),
        'transaction_type_card': np.zeros(n_small),
        'is_fraud': np.ones(n_small),
        'fraud_type': ['Small Testing Transaction'] * n_small
    }
    
    # 6. Card Skimming
    n_skimming = n_fraud // 9
    skimming_data = {
        'amount': np.random.uniform(3000, 8000, n_skimming),
        'is_new_device': np.zeros(n_skimming),
        'is_unusual_location': np.random.choice([0, 1], n_skimming),
        'is_first_time_receiver': np.ones(n_skimming),
        'is_suspicious_receiver': np.zeros(n_skimming),
        'transaction_frequency': np.random.uniform(0.1, 2, n_skimming),
        'time_since_last_txn': np.random.uniform(1000, 10000, n_skimming),  # Long gaps
        'is_weekend': np.random.choice([0, 1], n_skimming),
        'is_night': np.random.choice([0, 1], n_skimming),
        'transaction_type_upi': np.zeros(n_skimming),
        'transaction_type_card': np.ones(n_skimming),
        'is_fraud': np.ones(n_skimming),
        'fraud_type': ['Card Skimming'] * n_skimming
    }
    
    # 7. Data Breach Reuse
    n_breach = n_fraud // 9
    breach_data = {
        'amount': np.random.uniform(1000, 4000, n_breach),
        'is_new_device': np.zeros(n_breach),
        'is_unusual_location': np.zeros(n_breach),
        'is_first_time_receiver': np.ones(n_breach),
        'is_suspicious_receiver': np.zeros(n_breach),
        'transaction_frequency': np.random.uniform(0.1, 2, n_breach),
        'time_since_last_txn': np.random.uniform(100, 5000, n_breach),
        'is_weekend': np.random.choice([0, 1], n_breach),
        'is_night': np.random.choice([0, 1], n_breach),
        'transaction_type_upi': np.zeros(n_breach),
        'transaction_type_card': np.ones(n_breach),
        'is_fraud': np.ones(n_breach),
        'fraud_type': ['Data Breach Reuse'] * n_breach
    }
    
    # 8. Unusual Location/Activity
    n_unusual = n_fraud // 9
    unusual_data = {
        'amount': np.random.uniform(1000, 5000, n_unusual),
        'is_new_device': np.random.choice([0, 1], n_unusual),
        'is_unusual_location': np.ones(n_unusual),
        'is_first_time_receiver': np.random.choice([0, 1], n_unusual),
        'is_suspicious_receiver': np.zeros(n_unusual),
        'transaction_frequency': np.random.uniform(0.1, 5, n_unusual),
        'time_since_last_txn': np.random.uniform(100, 5000, n_unusual),
        'is_weekend': np.random.choice([0, 1], n_unusual),
        'is_night': np.random.choice([0, 1], n_unusual),
        'transaction_type_upi': np.zeros(n_unusual),
        'transaction_type_card': np.ones(n_unusual),
        'is_fraud': np.ones(n_unusual),
        'fraud_type': ['Unusual Location/Activity'] * n_unusual
    }
    
    # 9. CNP Fraud
    n_cnp = n_fraud - (n_phishing + n_qr + n_sim + n_fake_app + n_small + n_skimming + n_breach + n_unusual)
    cnp_data = {
        'amount': np.random.uniform(5000, 9000, n_cnp),
        'is_new_device': np.random.choice([0, 1], n_cnp),
        'is_unusual_location': np.random.choice([0, 1], n_cnp),
        'is_first_time_receiver': np.ones(n_cnp),
        'is_suspicious_receiver': np.zeros(n_cnp),
        'transaction_frequency': np.random.uniform(0.1, 2, n_cnp),
        'time_since_last_txn': np.random.uniform(100, 5000, n_cnp),
        'is_weekend': np.random.choice([0, 1], n_cnp),
        'is_night': np.random.choice([0, 1], n_cnp, p=[0.3, 0.7]),  # More likely at night
        'transaction_type_upi': np.zeros(n_cnp),
        'transaction_type_card': np.ones(n_cnp),
        'is_fraud': np.ones(n_cnp),
        'fraud_type': ['CNP Fraud'] * n_cnp
    }
    
    # Combine all data
    all_data = {}
    for key in legitimate_data:
        all_data[key] = np.concatenate([
            legitimate_data[key],
            phishing_data[key],
            qr_data[key],
            sim_data[key],
            fake_app_data[key],
            small_data[key],
            skimming_data[key],
            breach_data[key],
            unusual_data[key],
            cnp_data[key]
        ])
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def train_model(data):
    """Train a Random Forest model for fraud detection."""
    # Prepare features and target
    X = data.drop(columns=["is_fraud", "fraud_type"])
    y = data["is_fraud"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["amount", "transaction_frequency", "time_since_last_txn"]
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy: {test_acc:.4f}")
    
    return model, scaler, X_test, y_test

def predict_transaction(model, scaler, transaction, threshold=0.5):
    """
    Predict whether a transaction is fraudulent.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        transaction: Dictionary of transaction features
        threshold: Decision threshold for fraud
        
    Returns:
        Dictionary of prediction results
    """
    # Create a DataFrame from the transaction
    txn_df = pd.DataFrame([transaction])
    
    # Scale numerical features
    numerical_cols = ["amount", "transaction_frequency", "time_since_last_txn"]
    txn_df[numerical_cols] = scaler.transform(txn_df[numerical_cols])
    
    # Make prediction
    start_time = time.time()
    fraud_prob = model.predict_proba(txn_df)[0, 1]
    is_fraud = fraud_prob >= threshold
    prediction_time = (time.time() - start_time) * 1000  # in milliseconds
    
    # Determine fraud type
    if is_fraud:
        # Simple rules for fraud type detection
        if transaction.get('is_suspicious_receiver', 0) == 1 and transaction.get('transaction_type_upi', 0) == 1:
            fraud_type = "Phishing Link"
        elif transaction.get('transaction_type_upi', 0) == 1 and transaction.get('amount') <= 10 and transaction.get('transaction_frequency') > 5:
            fraud_type = "Small Testing Transaction"
        elif transaction.get('transaction_type_upi', 0) == 1 and transaction.get('is_new_device', 0) == 1 and transaction.get('is_unusual_location', 0) == 1:
            fraud_type = "SIM Swap Attack"
        elif transaction.get('transaction_type_upi', 0) == 1 and transaction.get('is_new_device', 0) == 1:
            fraud_type = "Fake UPI App"
        elif transaction.get('transaction_type_card', 0) == 1 and transaction.get('amount') > 3000 and transaction.get('time_since_last_txn') > 1000:
            fraud_type = "Card Skimming"
        elif transaction.get('transaction_type_card', 0) == 1 and transaction.get('is_first_time_receiver', 0) == 1:
            fraud_type = "Data Breach Reuse"
        elif transaction.get('transaction_type_card', 0) == 1 and transaction.get('is_unusual_location', 0) == 1:
            fraud_type = "Unusual Location/Activity"
        elif transaction.get('transaction_type_card', 0) == 1 and transaction.get('amount') > 5000:
            fraud_type = "CNP Fraud"
        else:
            fraud_type = "Unknown Fraud"
    else:
        fraud_type = "Not Fraud"
    
    # Return prediction result
    result = {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(fraud_prob),
        "fraud_type": fraud_type,
        "prediction_time_ms": prediction_time
    }
    
    return result

def generate_test_scenarios():
    """Generate test scenarios for different fraud types."""
    scenarios = [
        # UPI Frauds
        {
            "name": "Phishing Link (UPI)",
            "amount": 4500,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 1,
            "is_suspicious_receiver": 1,
            "transaction_frequency": 1,
            "time_since_last_txn": 3000,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_type_upi": 1,
            "transaction_type_card": 0
        },
        {
            "name": "QR Code Scam (UPI)",
            "amount": 1500,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 1,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 1,
            "time_since_last_txn": 2000,
            "is_weekend": 1,
            "is_night": 0,
            "transaction_type_upi": 1,
            "transaction_type_card": 0
        },
        {
            "name": "SIM Swap Attack (UPI)",
            "amount": 3000,
            "is_new_device": 1,
            "is_unusual_location": 1,
            "is_first_time_receiver": 0,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 15,
            "time_since_last_txn": 5,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_type_upi": 1,
            "transaction_type_card": 0
        },
        {
            "name": "Fake UPI App",
            "amount": 2500,
            "is_new_device": 1,
            "is_unusual_location": 0,
            "is_first_time_receiver": 1,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 2,
            "time_since_last_txn": 500,
            "is_weekend": 0,
            "is_night": 1,
            "transaction_type_upi": 1,
            "transaction_type_card": 0
        },
        {
            "name": "Small Testing Transaction (UPI)",
            "amount": 5,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 0,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 15,
            "time_since_last_txn": 3,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_type_upi": 1,
            "transaction_type_card": 0
        },
        
        # Card Frauds
        {
            "name": "Card Skimming",
            "amount": 7500,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 1,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 1,
            "time_since_last_txn": 8000,
            "is_weekend": 1,
            "is_night": 0,
            "transaction_type_upi": 0,
            "transaction_type_card": 1
        },
        {
            "name": "Data Breach Reuse",
            "amount": 3500,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 1,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 1,
            "time_since_last_txn": 2000,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_type_upi": 0,
            "transaction_type_card": 1
        },
        {
            "name": "Unusual Location/Activity",
            "amount": 4000,
            "is_new_device": 0,
            "is_unusual_location": 1,
            "is_first_time_receiver": 0,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 2,
            "time_since_last_txn": 1500,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_type_upi": 0,
            "transaction_type_card": 1
        },
        {
            "name": "CNP Fraud",
            "amount": 8500,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 1,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 1,
            "time_since_last_txn": 3000,
            "is_weekend": 0,
            "is_night": 1,
            "transaction_type_upi": 0,
            "transaction_type_card": 1
        },
        
        # Legitimate Transactions
        {
            "name": "Legitimate UPI Transaction",
            "amount": 250,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 0,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 3,
            "time_since_last_txn": 1000,
            "is_weekend": 1,
            "is_night": 0,
            "transaction_type_upi": 1,
            "transaction_type_card": 0
        },
        {
            "name": "Legitimate Card Transaction",
            "amount": 1200,
            "is_new_device": 0,
            "is_unusual_location": 0,
            "is_first_time_receiver": 0,
            "is_suspicious_receiver": 0,
            "transaction_frequency": 2,
            "time_since_last_txn": 2000,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_type_upi": 0,
            "transaction_type_card": 1
        }
    ]
    
    return scenarios

def run_demo():
    """Run the fraud detection demo."""
    print("=== Real-Time Fraud Detection System Demo ===\n")
    
    # Generate synthetic data
    print("1. Generating synthetic transaction data...")
    data = generate_synthetic_data(n_samples=10000, fraud_ratio=0.03)
    print(f"   Generated {len(data)} transactions ({data['is_fraud'].sum()} fraudulent)\n")
    
    # Train model
    print("2. Training the fraud detection model...")
    model, scaler, X_test, y_test = train_model(data)
    print("   Model training complete\n")
    
    # Test model on scenarios
    print("3. Testing on fraud scenarios:")
    scenarios = generate_test_scenarios()
    
    results = []
    for i, scenario in enumerate(scenarios):
        name = scenario.pop("name")
        result = predict_transaction(model, scaler, scenario)
        
        # Add name and display
        result["name"] = name
        results.append(result)
        
        status = "FRAUD DETECTED ❌" if result["is_fraud"] else "Legitimate ✓"
        print(f"   {i+1}. {name}: {status}")
        
        if result["is_fraud"]:
            print(f"      Fraud Type: {result['fraud_type']}")
            print(f"      Probability: {result['fraud_probability']:.4f}")
        
        print(f"      Processing Time: {result['prediction_time_ms']:.2f} ms\n")
    
    # Display summary
    print("=== Fraud Detection Summary ===")
    fraud_count = sum(1 for r in results if r["is_fraud"])
    print(f"Total Scenarios: {len(scenarios)}")
    print(f"Fraud Detected: {fraud_count}/{len(scenarios)}")
    
    # Count by fraud type
    fraud_types = {}
    for r in results:
        if r["is_fraud"]:
            fraud_type = r["fraud_type"]
            fraud_types[fraud_type] = fraud_types.get(fraud_type, 0) + 1
    
    print("\nFraud Types Detected:")
    for fraud_type, count in fraud_types.items():
        print(f"  - {fraud_type}: {count}")

if __name__ == "__main__":
    run_demo() 