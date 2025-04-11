import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import time

class FraudDetectionModel:
    def __init__(self, model_type="random_forest", use_smote=True):
        """
        Initialize the fraud detection model.
        
        Args:
            model_type (str): Type of model to use ("random_forest" or "xgboost")
            use_smote (bool): Whether to use SMOTE for handling class imbalance
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.model = None
        self.feature_importances = None
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,  # For imbalanced data
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the fraud detection model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            self: Trained model
        """
        start_time = time.time()
        
        # Apply SMOTE for handling class imbalance if enabled
        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
            print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")
            
            # Train on resampled data
            self.model.fit(X_resampled, y_resampled)
        else:
            # Train on original data
            self.model.fit(X_train, y_train)
        
        # Calculate feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        training_time = time.time() - start_time
        print(f"Model trained in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """
        Predict fraud probability for new data.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        start_time = time.time()
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        prediction_time = time.time() - start_time
        avg_time_per_record = prediction_time / len(X) if len(X) > 0 else 0
        print(f"Prediction made in {prediction_time:.4f} seconds (avg {avg_time_per_record*1000:.2f} ms per record)")
        
        return y_pred, y_prob
    
    def evaluate(self, X_test, y_test, threshold=0.5, output_dir=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            threshold (float): Decision threshold for fraud
            output_dir (str): Directory to save evaluation plots
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Make predictions
        _, y_prob = self.predict(X_test)
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Display results
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Classification Report (threshold={threshold}):")
        print(classification_report(y_test, y_pred))
        
        # Generate plots if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Legitimate', 'Fraud'], 
                        yticklabels=['Legitimate', 'Fraud'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            
            # Feature Importance
            if self.feature_importances is not None:
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=self.feature_importances.head(15))
                plt.title('Top 15 Feature Importances')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                
            # ROC Curve
            plt.figure(figsize=(8, 6))
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            
            # Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        
        # Calculate metrics for each fraud type
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'roc_auc': roc_auc
        }
        
        return metrics
    
    def save(self, output_path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"Model saved to {output_path}")
    
    @classmethod
    def load(cls, input_path):
        """Load a trained model from disk."""
        return joblib.load(input_path)
        
    def get_feature_importance(self, top_n=None):
        """Get the most important features."""
        if self.feature_importances is None:
            return None
            
        if top_n:
            return self.feature_importances.head(top_n)
        return self.feature_importances 