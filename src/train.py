import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from features.feature_engineering import FeatureEngineer

def generate_training_data(n_samples=10000):
    """Generate synthetic training data"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Generate base transaction
        transaction = {
            'Sender_ID': f'USER_{np.random.randint(1, 1000)}',
            'Receiver_ID': f'REC_{np.random.randint(1, 1000)}',
            'Timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30)),
            'Amount': np.random.uniform(10, 1000),
            'Location': f'{np.random.uniform(-90, 90):.6f}, {np.random.uniform(-180, 180):.6f}',
            'Device_ID': f'DEV_{np.random.randint(1000, 9999)}',
            'Transaction_Type': np.random.choice(['Transfer', 'Payment', 'Withdrawal']),
            'Merchant_Type': np.random.choice(['Retail', 'Online', 'ATM', 'POS']),
            'Is_Fraud': 0
        }
        
        # Add fraud characteristics with 20% probability
        if np.random.random() < 0.2:
            transaction.update({
                'Amount': np.random.uniform(5000, 10000),  # Unusually high amount
                'Location': '0.0, 0.0',  # Unusual location
                'Device_ID': 'DEV_Unknown',  # Unknown device
                'Merchant_Type': 'Unknown',  # Unknown merchant
                'Is_Fraud': 1
            })
            
        data.append(transaction)
    
    return pd.DataFrame(data)

def train_model():
    """Train the fraud detection model"""
    print("Generating training data...")
    train_data = generate_training_data()
    
    print("Initializing feature engineering pipeline...")
    feature_engineer = FeatureEngineer()
    
    print("Processing training data...")
    X, y = feature_engineer.process_data(train_data, fit=True)
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    print("Saving model and feature engineer...")
    os.makedirs('models', exist_ok=True)
    feature_engineer.save('models/feature_engineer.pkl')
    joblib.dump(model, 'models/fraud_detection_model.pkl')
    print("Training complete!")

if __name__ == "__main__":
    train_model() 