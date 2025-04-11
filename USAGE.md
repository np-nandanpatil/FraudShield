# Using the Real-Time Fraud Detection System

This guide explains how to set up and run the real-time fraud detection system.

## Prerequisites

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/synthhack.git
   cd synthhack
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Alternatively, install the package in development mode:
   ```
   pip install -e .
   ```

## Running the Demo

### Quick Demo

For a simplified demonstration that showcases all fraud types without any complex dependencies:

```
python src/demo.py
```

This will:
1. Generate synthetic transaction data
2. Train a Random Forest model
3. Test the model on various fraud scenarios
4. Display the results

### Full System (Advanced)

To run the complete system with all features:

```
python src/main.py
```

This will:
1. Generate more complex synthetic transaction data
2. Train a model with advanced feature engineering
3. Run a real-time detection simulation with test scenarios

## Components

### 1. Data Generation

The system can generate synthetic data that mimics real-world transaction patterns:

- UPI Frauds: Phishing Links, QR Code Scams, SIM Swap Attacks, Fake UPI Apps, Small Testing Transactions
- Card Frauds: Card Skimming, Data Breach Reuse, Unusual Location/Activity, CNP Fraud

### 2. Feature Engineering

The system extracts relevant features from transaction data:

- Time-based features (hour, day of week, weekend/night flags)
- Transaction frequency and timing
- Device and location changes
- Amount-based features
- Receiver information

### 3. Model Training

The system uses Random Forest (default) or XGBoost models with options:

- SMOTE for handling class imbalance
- Hyperparameter optimization
- Feature importance analysis

### 4. Real-Time Detection

For real-time transaction processing:

```python
from src.utils.predictor import RealTimePredictor

# Initialize the predictor
predictor = RealTimePredictor(
    model_path="models/fraud_detection_model.pkl",
    feature_engineer_path="models/feature_engineer.pkl"
)

# Process a transaction
transaction = {
    "Transaction_ID": "Txn_12345",
    "Timestamp": "2023-10-15 14:30:00",
    "Amount": 5000,
    "Sender_ID": "User_123",
    "Receiver_ID": "Merchant_456",
    "Transaction_Type": "UPI",
    "Merchant_Type": "Online_Retail",
    "Device_ID": "Device_789",
    "Location": "12.9716, 77.5946"
}

result = predictor.predict_transaction(transaction)

# Take action based on result
if result["Is_Fraud"]:
    print(f"FRAUD DETECTED: {result['Fraud_Type']}")
    # Block the transaction
else:
    print("Transaction is legitimate")
    # Allow the transaction
```

## Simplified API (src/demo.py)

For a simpler approach without complex data processing:

```python
from src.demo import generate_synthetic_data, train_model, predict_transaction

# Generate data and train model
data = generate_synthetic_data(n_samples=10000, fraud_ratio=0.03)
model, scaler, _, _ = train_model(data)

# Process a transaction
transaction = {
    "amount": 5000,
    "is_new_device": 1,
    "is_unusual_location": 0,
    "is_first_time_receiver": 1,
    "is_suspicious_receiver": 0,
    "transaction_frequency": 2,
    "time_since_last_txn": 3000,
    "is_weekend": 0,
    "is_night": 0,
    "transaction_type_upi": 1,
    "transaction_type_card": 0
}

result = predict_transaction(model, scaler, transaction)
print(f"Fraud detected: {result['is_fraud']}")
print(f"Fraud type: {result['fraud_type']}")
```

## Hackathon Demo Tips

For a successful hackathon demo:

1. **Use the simplified demo**: Start with `demo.py` to showcase the core functionality
2. **Explain fraud patterns**: Highlight how the system can detect different fraud types
3. **Demonstrate real-time ability**: Show the low prediction times (typically <10ms per transaction)
4. **Emphasize UPI and card coverage**: Show how the system handles all specified fraud types

## Evaluation Results

After training, check the `models/evaluation` directory for:
- Confusion matrix plot
- ROC curve
- Precision-Recall curve
- Feature importance plot 