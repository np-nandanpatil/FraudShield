import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.feature_engineering import FeatureEngineer
from src.models.fraud_model import FraudDetectionModel

def train_model(data_path, model_output_path, feature_engineer_output_path, output_dir=None, 
                model_type="random_forest", use_smote=True, test_size=0.2, random_state=42):
    """
    Train a fraud detection model and save it.
    
    Args:
        data_path (str): Path to the raw data file
        model_output_path (str): Path to save the trained model
        feature_engineer_output_path (str): Path to save the feature engineer
        output_dir (str): Directory to save evaluation results
        model_type (str): Type of model to use (random_forest or xgboost)
        use_smote (bool): Whether to use SMOTE for handling class imbalance
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} transactions")
    
    # Create feature engineer
    print("Creating and fitting feature engineer...")
    feature_engineer = FeatureEngineer()
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["Is_Fraud"])
    print(f"Training data: {len(train_data)} rows, Test data: {len(test_data)} rows")
    
    # Process data
    print("Processing training data...")
    X_train, y_train = feature_engineer.process_data(train_data, fit=True)
    print("Processing test data...")
    X_test, y_test = feature_engineer.process_data(test_data, fit=False)
    
    print(f"Training data shape: {X_train.shape}, Fraud ratio: {y_train.mean():.2%}")
    print(f"Test data shape: {X_test.shape}, Fraud ratio: {y_test.mean():.2%}")
    
    # Train model
    print(f"Training {model_type} model...")
    model = FraudDetectionModel(model_type=model_type, use_smote=use_smote)
    model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test, threshold=0.5, output_dir=output_dir)
    
    # Save model and feature engineer
    print(f"Saving model to {model_output_path}...")
    model.save(model_output_path)
    
    print(f"Saving feature engineer to {feature_engineer_output_path}...")
    feature_engineer.save(feature_engineer_output_path)
    
    return model, feature_engineer, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a fraud detection model")
    parser.add_argument("--data_path", type=str, default="data/fraud_transactions.csv", help="Path to the raw data file")
    parser.add_argument("--model_output_path", type=str, default="models/fraud_detection_model.pkl", help="Path to save the trained model")
    parser.add_argument("--feature_engineer_output_path", type=str, default="models/feature_engineer.pkl", help="Path to save the feature engineer")
    parser.add_argument("--output_dir", type=str, default="models/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "xgboost"], help="Type of model to use")
    parser.add_argument("--use_smote", type=bool, default=True, help="Whether to use SMOTE for handling class imbalance")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output_path,
        feature_engineer_output_path=args.feature_engineer_output_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        use_smote=args.use_smote,
        test_size=args.test_size,
        random_state=args.random_state
    ) 