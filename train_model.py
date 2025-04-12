import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.feature_engineering import FeatureEngineer

def train_model():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    data_path = 'data/fraud_transactions.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Initialize feature engineer
    print("Initializing feature engineering...")
    feature_engineer = FeatureEngineer()
    
    # Process data
    print("Processing data...")
    X, y = feature_engineer.process_data(df, fit=True)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Save model and feature engineer
    print("Saving model and feature engineer...")
    joblib.dump(model, 'models/fraud_detection_model.pkl')
    feature_engineer.save('models/feature_engineer.pkl')
    
    # Print performance metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
if __name__ == '__main__':
    train_model() 