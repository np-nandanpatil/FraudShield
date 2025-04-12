import os
import csv
from datetime import datetime, timedelta
import random
import uuid

def generate_synthetic_transactions(num_transactions=1000):
    """Generate synthetic transaction data with realistic patterns"""
    
    # Transaction types and their probabilities
    transaction_types = {
        'UPI': 0.6,
        'Card': 0.3,
        'Netbanking': 0.1
    }
    
    # Merchant types
    merchant_types = ['Online', 'Retail', 'Travel', 'Food', 'Entertainment', 'Utilities']
    
    # Fraud patterns
    fraud_types = {
        'QR Code Scam': 0.3,
        'Small Testing': 0.25,
        'Phishing': 0.2,
        'SIM Swap': 0.15,
        'Data Breach': 0.1
    }
    
    # Generate transactions
    transactions = []
    start_date = datetime.now() - timedelta(days=30)
    
    for _ in range(num_transactions):
        # Generate timestamp within last 30 days
        timestamp = start_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Determine transaction type
        txn_type = random.choices(
            list(transaction_types.keys()),
            weights=list(transaction_types.values())
        )[0]
        
        # Generate amount (with realistic patterns)
        if random.random() < 0.8:  # 80% normal transactions
            amount = random.randint(100, 10000)  # Regular amounts
        else:
            amount = random.randint(10000, 100000)  # Large transactions
            
        # Determine if fraudulent (about 15% fraud rate)
        is_fraud = random.random() < 0.15
        
        # For fraudulent transactions, adjust patterns
        if is_fraud:
            if random.random() < 0.3:  # Small testing transactions
                amount = random.randint(1, 100)
            fraud_type = random.choices(
                list(fraud_types.keys()),
                weights=list(fraud_types.values())
            )[0]
        else:
            fraud_type = '-'
            
        # Generate transaction ID
        txn_id = f"{txn_type}_{int(timestamp.timestamp())}"
        
        # Generate sender and receiver IDs
        if txn_type == 'UPI':
            sender_id = f"user{random.randint(1000,9999)}@upi"
            receiver_id = f"merchant{random.randint(100,999)}@upi"
        elif txn_type == 'Card':
            sender_id = f"{random.randint(100000,999999)}******{random.randint(1000,9999)}"
            receiver_id = f"MERCHANT_{random.randint(1000,9999)}"
        else:
            sender_id = f"ACC_{random.randint(10000,99999)}"
            receiver_id = f"MERCHANT_{random.randint(1000,9999)}"
            
        # Generate device ID and location
        device_id = f"Device_{random.randint(1000,9999)}"
        location = f"{random.uniform(8.0, 37.0):.4f}, {random.uniform(68.0, 97.0):.4f}"
        
        # Calculate fraud probability
        if is_fraud:
            fraud_prob = random.uniform(0.7, 0.95)
        else:
            fraud_prob = random.uniform(0.01, 0.3)
            
        transactions.append([
            txn_id,
            timestamp.isoformat(),
            amount,
            sender_id,
            receiver_id,
            txn_type,
            random.choice(merchant_types),
            device_id,
            location,
            is_fraud,
            fraud_type,
            fraud_prob,
            'COMPLETED'
        ])
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x[1])
    
    # Write to CSV
    os.makedirs('data', exist_ok=True)
    with open('data/processed_transactions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Transaction_ID',
            'Timestamp',
            'Amount',
            'Sender_ID',
            'Receiver_ID',
            'Transaction_Type',
            'Merchant_Type',
            'Device_ID',
            'Location',
            'Is_Fraud',
            'Fraud_Type',
            'Fraud_Probability',
            'Status'
        ])
        writer.writerows(transactions)
    
    print(f"Generated {num_transactions} synthetic transactions")

if __name__ == '__main__':
    generate_synthetic_transactions(1000)  # Generate 1000 transactions 