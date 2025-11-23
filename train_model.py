import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.feature_engineering import FeatureEngineer
from src.utils.predictor import RealTimePredictor

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
    
    # Evaluate different thresholds
    from sklearn.metrics import classification_report, confusion_matrix

    # Get predictions for test set
    test_probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\nModel Performance:")
    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nThreshold Analysis:")
    print("Threshold | Accuracy | Precision | Recall | F1-Score")
    print("-" * 55)

    best_f1 = 0
    best_threshold = 0.3

    for threshold in thresholds:
        y_pred = (test_probabilities >= threshold).astype(int)
        accuracy = (y_pred == y_test).mean()
        precision = (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
        recall = (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{threshold:>9} | {accuracy:>8.4f} | {precision:>9.4f} | {recall:>6.4f} | {f1:>9.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.4f})")

    # Update the predictor with the best threshold
    predictor = RealTimePredictor('models/fraud_detection_model.pkl', 'models/feature_engineer.pkl', threshold=best_threshold)
    print(f"Updated predictor threshold to: {best_threshold}")
    
if __name__ == '__main__':
    train_model() 