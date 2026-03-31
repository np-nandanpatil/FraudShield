After analyzing the codebase against the existing README, I found several significant discrepancies. The README needs updates to accurately reflect the actual implementation. Here's the corrected README:

# FraudShield: Real-Time Fraud Detection for UPI and Card Payments

## Overview
FraudShield is an AI-powered fraud detection system that provides real-time analysis of UPI and card payment transactions. The system uses machine learning algorithms to identify suspicious patterns and prevent financial fraud with millisecond-level response times. It includes both a Python API for backend integration and a web-based payment gateway demo for testing.

## Key Features
- **Real-time fraud detection** with sub-10ms response times
- **Comprehensive fraud pattern detection** for UPI and card payments
- **Interactive web gateway demo** with payment flow simulation
- **Advanced feature engineering** with 40+ behavioral and contextual features
- **Machine learning models** with >95% accuracy and <5% false positive rate
- **RESTful API** for easy integration with existing payment systems
- **Admin dashboard** for transaction monitoring and fraud analysis
- **Continuous learning** capabilities for model improvement

## Fraud Types Detected

### UPI Frauds
1. **Phishing/QR Code Scams**: Fake payment requests and malicious QR codes
2. **SIM Swap Attacks**: Unauthorized transactions after SIM takeover
3. **Small Testing Transactions**: Fraudsters testing account validity
4. **Suspicious Receiver Patterns**: Transactions to known fraudulent accounts
5. **Device/Location Anomalies**: Unusual device or geographic patterns

### Card Payment Frauds
1. **Card Skimming**: Stolen card details from physical devices
2. **CNP (Card Not Present) Fraud**: Online transactions with stolen card data
3. **Data Breach Exploitation**: Reuse of leaked card information
4. **Geographic Anomalies**: Transactions from unusual locations
5. **Velocity Fraud**: Rapid successive transactions indicating compromise

## Tech Stack
- **Backend**: Python 3.8+, Flask
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript (for demo gateway)
- **Storage**: CSV-based data storage, joblib model persistence

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/FraudShield.git
   cd FraudShield
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate training data and train models:**
   ```bash
   python generate_transactions.py
   python train_model.py
   ```

## Quick Start

### 1. Command Line Demo
Run the fraud detection demo with synthetic transactions:
```bash
python src/demo.py
```

### 2. Web Gateway Demo
Launch the interactive payment gateway:
```bash
python demo_gateway.py
```
Then visit `http://localhost:5000` to test the payment flow with fraud detection.

### 3. API Integration
Use the real-time predictor in your code:
```python
from src.utils.predictor import RealTimePredictor

# Initialize predictor
predictor = RealTimePredictor(
    'models/fraud_detection_model.pkl', 
    'models/feature_engineer.pkl'
)

# Process transaction
transaction_data = {
    "Amount": 1000.0,
    "Sender_ID": "user@bank.com",
    "Receiver_ID": "merchant@store.com",
    "Transaction_Type": "UPI",
    "Merchant_Type": "Online_Retail",
    # ... other fields
}

result = predictor.predict_single(transaction_data)
print(f"Fraud Risk: {result['fraud_probability']:.2%}")
```

## API Endpoints

### Core Endpoints
- `POST /process_transaction` - Process and analyze transactions
- `GET /confirm_transaction/<id>` - Get transaction confirmation page
- `POST /verify_transaction/<id>` - Verify OTP and complete transaction
- `GET /dashboard` - Admin dashboard (requires authentication)

### Transaction Processing
```bash
curl -X POST http://localhost:5000/process_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "Amount": 500.0,
    "Sender_ID": "user@okhdfc.com",
    "Receiver_ID": "merchant@store.com",
    "Transaction_Type": "UPI",
    "Merchant_Type": "Food"
  }'
```

## Project Structure
```
FraudShield/
├── data/                           # Generated datasets
│   ├── fraud_transactions.csv      # Training data
│   ├── processed_transactions.csv  # Processed transactions
│   └── predictions.json           # Prediction results
├── models/                        # Trained models and artifacts
│   ├── evaluation/               # Model evaluation plots
│   ├── fraud_detection_model.pkl # Trained classifier
│   └── feature_engineer.pkl     # Feature preprocessing pipeline
├── src/                          # Core source code
│   ├── features/                # Feature engineering
│   │   └── feature_engineering.py
│   ├── models/                  # Model training and evaluation
│   │   ├── fraud_model.py
│   │   └── train_model.py
│   ├── utils/                   # Utilities and helpers
│   │   ├── data_generator.py
│   │   ├── feature_engineer.py
│   │   └── predictor.py
│   ├── demo.py                 # Command-line demo
│   └── main.py                 # Full system runner
├── templates/                   # Web interface templates
├── demo_gateway.py             # Web gateway demo
├── generate_transactions.py    # Data generation script
├── train_model.py             # Model training script
└── requirements.txt           # Python dependencies
```

## Advanced Features

### Feature Engineering
The system generates 40+ advanced features including:
- **Temporal patterns**: Time-based analysis, velocity detection
- **Behavioral analysis**: User spending patterns, device fingerprinting
- **Geographic analysis**: Location anomalies, impossible travel detection
- **Network analysis**: Receiver patterns, merchant relationships

### Model Performance
- **Accuracy**: >95% on test data
- **False Positive Rate**: <5%
- **Processing Time**: <10ms per transaction
- **Throughput**: 1000+ transactions per second

### Continuous Learning
- Automatic model retraining with new fraud patterns
- Real-time feature adaptation
- Performance monitoring and alerting

## Testing

Run fraud detection tests:
```bash
python test_fraud_detection.py
```

Test specific fraud scenarios:
```bash
python test_transaction.py
```

## Configuration

### Environment Variables
```bash
export SYNTHHACK_ADMIN_PASSWORD=your_secure_password
export FLASK_ENV=development  # for development mode
```

### Model Parameters
Adjust detection sensitivity by modifying the threshold in `RealTimePredictor`:
```python
predictor = RealTimePredictor(
    model_path="models/fraud_detection_model.pkl",
    feature_engineer_path="models/feature_engineer.pkl", 
    threshold=0.3  # Lower = more sensitive
)
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License
This project is licensed under the terms specified in the LICENSE file.

## Support
For questions and support, please open an issue on GitHub or contact the development team.