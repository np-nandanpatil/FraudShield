# SynthHack: Real-Time Fraud Detection for UPI and Card Payments

## Overview
An AI-based system to detect and block financial fraud in real-time for UPI and card payment transactions. The system identifies suspicious activities associated with specific fraud types and flags or blocks them instantly to prevent financial loss.

## Key Features
- Real-time fraud detection with millisecond-level responses
- Coverage for multiple fraud types in UPI and card payments
- Machine learning-based classification with high accuracy
- Simplified API for easy integration

## Fraud Types Detected

### UPI Frauds
1. **Phishing Links**: Victims click fake payment requests expecting to receive money but end up paying the scammer
2. **QR Code Scams**: Scammers provide QR codes that appear to credit money but debit the victim's account instead
3. **SIM Swap Attacks**: Hackers gain control of a victim's SIM card to make unauthorized transactions
4. **Fake UPI Apps**: Counterfeit apps mimic legitimate UPI platforms to steal credentials
5. **Small Testing Transactions**: Fraudsters perform small transactions to test if an account is active

### Card Payment Frauds
1. **Card Skimming**: Card details stolen via physical devices at ATMs or POS terminals
2. **Data Breach Reuse**: Card numbers leaked from one site are used on others
3. **Unusual Location/Activity**: Sudden transactions in a new country or with high amounts
4. **CNP (Card Not Present) Fraud**: Online purchases using stolen card details

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the simplified demo:
   ```
   python src/demo.py
   ```

This demo will:
- Generate synthetic transaction data with patterns for all fraud types
- Train a machine learning model to detect the frauds
- Test the model on various fraud scenarios
- Display detection results and timing metrics

## Advanced Usage

For more advanced usage with comprehensive feature engineering:
```
python src/main.py
```

See [USAGE.md](USAGE.md) for detailed instructions and API documentation.

## Project Structure
```
├── data/               # Data files (synthetic transactions)
├── models/             # Saved trained models and scalers
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── features/       # Feature engineering code
│   ├── models/         # Model training and evaluation
│   ├── utils/          # Utility functions
│   ├── main.py         # Full system implementation
│   └── demo.py         # Simplified demo implementation
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Performance Metrics
- Detection rate: >95% across all fraud types
- False positive rate: <5%
- Processing time: <10ms per transaction
- Real-time capability: Can process thousands of transactions per second