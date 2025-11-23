import os
import csv
from datetime import datetime, timedelta
import random
import uuid

class FraudPatternGenerator:
    """Generate realistic fraud patterns for synthetic transaction data"""

    def __init__(self):
        # Base locations (major Indian cities)
        self.base_locations = {
            'Delhi': (28.6139, 77.2090),
            'Mumbai': (19.0760, 72.8777),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714)
        }

        # Fraud patterns with their characteristics
        self.fraud_patterns = {
            'QR_Code_Scam': {
                'description': 'Fake QR codes leading to fraudulent transfers',
                'amount_range': (100, 5000),
                'receiver_pattern': 'Suspicious_*',
                'location_anomaly': True,
                'device_anomaly': False,
                'rapid_transactions': False
            },
            'Small_Testing': {
                'description': 'Small amount testing before large fraud',
                'amount_range': (1, 100),
                'receiver_pattern': 'Test_*',
                'location_anomaly': False,
                'device_anomaly': False,
                'rapid_transactions': True
            },
            'SIM_Swap': {
                'description': 'Device/location change indicating SIM swap',
                'amount_range': (500, 50000),
                'receiver_pattern': 'Merchant_*',
                'location_anomaly': True,
                'device_anomaly': True,
                'rapid_transactions': True
            },
            'Data_Breach': {
                'description': 'Stolen card credentials used',
                'amount_range': (1000, 100000),
                'receiver_pattern': 'Merchant_*',
                'location_anomaly': True,
                'device_anomaly': True,
                'rapid_transactions': False
            },
            'Phishing_Scam': {
                'description': 'Credentials obtained via phishing',
                'amount_range': (500, 25000),
                'receiver_pattern': 'Suspicious_*',
                'location_anomaly': True,
                'device_anomaly': False,
                'rapid_transactions': False
            },
            'Card_Testing': {
                'description': 'Multiple small transactions to test stolen card',
                'amount_range': (1, 50),
                'receiver_pattern': 'Test_*',
                'location_anomaly': False,
                'device_anomaly': False,
                'rapid_transactions': True
            },
            'Unusual_Location': {
                'description': 'Transaction from unusual geographic location',
                'amount_range': (100, 50000),
                'receiver_pattern': 'Merchant_*',
                'location_anomaly': True,
                'device_anomaly': False,
                'rapid_transactions': False
            }
        }

    def generate_fraudulent_transaction(self, txn_type, base_user_data):
        """Generate a fraudulent transaction with realistic patterns"""
        fraud_type = random.choice(list(self.fraud_patterns.keys()))
        pattern = self.fraud_patterns[fraud_type]

        # Generate amount based on fraud pattern
        amount = random.randint(*pattern['amount_range'])

        # Generate receiver based on pattern
        if 'Suspicious' in pattern['receiver_pattern']:
            receiver_id = f"Suspicious_{random.randint(1, 100)}@{txn_type.lower()}"
        elif 'Test' in pattern['receiver_pattern']:
            receiver_id = f"Test_Merchant_{random.randint(1, 50)}@{txn_type.lower()}"
        else:
            receiver_id = f"Merchant_{random.randint(100, 999)}@{txn_type.lower()}"

        # Handle location anomaly
        if pattern['location_anomaly']:
            # Choose a different city from user's base location
            available_cities = [city for city in self.base_locations.keys()
                              if city != base_user_data['base_city']]
            fraud_city = random.choice(available_cities)
            location = f"{self.base_locations[fraud_city][0]:.4f}, {self.base_locations[fraud_city][1]:.4f}"
        else:
            location = base_user_data['base_location']

        # Handle device anomaly
        if pattern['device_anomaly']:
            device_id = f"Unknown_Device_{random.randint(1, 100)}"
        else:
            device_id = base_user_data['device_id']

        return {
            'amount': amount,
            'receiver_id': receiver_id,
            'location': location,
            'device_id': device_id,
            'fraud_type': fraud_type.replace('_', ' ')
        }

    def generate_legitimate_transaction(self, txn_type, base_user_data, recent_transactions):
        """Generate a legitimate transaction"""
        # Normal amount distribution
        if random.random() < 0.7:
            amount = random.randint(100, 5000)  # Regular shopping
        elif random.random() < 0.9:
            amount = random.randint(5000, 25000)  # Larger purchases
        else:
            amount = random.randint(25000, 100000)  # Big purchases

        # Normal merchant selection
        merchant_types = ['Online_Retail', 'Food', 'Travel', 'Entertainment', 'Utilities']
        merchant_type = random.choice(merchant_types)
        receiver_id = f"Merchant_{random.randint(100, 999)}@{txn_type.lower()}"

        # Slight location variation (normal travel)
        base_lat, base_lon = base_user_data['base_location'].split(', ')
        lat_offset = random.uniform(-0.01, 0.01)  # Small variation
        lon_offset = random.uniform(-0.01, 0.01)
        location = f"{float(base_lat) + lat_offset:.4f}, {float(base_lon) + lon_offset:.4f}"

        return {
            'amount': amount,
            'receiver_id': receiver_id,
            'location': location,
            'device_id': base_user_data['device_id'],
            'merchant_type': merchant_type
        }

def generate_synthetic_transactions(num_transactions=10000):
    """Generate synthetic transaction data with realistic fraud patterns"""

    fraud_generator = FraudPatternGenerator()

    # Transaction types and their probabilities
    transaction_types = {
        'UPI': 0.65,
        'Card': 0.30,
        'Netbanking': 0.05
    }

    # Generate user profiles for consistency
    num_users = 1000
    users = {}
    for i in range(num_users):
        base_city = random.choice(list(fraud_generator.base_locations.keys()))
        base_location = f"{fraud_generator.base_locations[base_city][0]:.4f}, {fraud_generator.base_locations[base_city][1]:.4f}"
        users[i] = {
            'user_id': f"User_{i}",
            'base_city': base_city,
            'base_location': base_location,
            'device_id': f"Device_{random.randint(1000, 9999)}",
            'upi_id': f"user{random.randint(1000, 9999)}@upi",
            'card_id': f"{random.randint(100000, 999999)}******{random.randint(1000, 9999)}"
        }

    # Generate transactions
    transactions = []
    start_date = datetime.now() - timedelta(days=90)  # 3 months of data

    # Track recent transactions per user for velocity features
    user_recent_txns = {user_id: [] for user_id in users.keys()}

    for i in range(num_transactions):
        # Select random user
        user_id = random.choice(list(users.keys()))
        user_data = users[user_id]

        # Generate timestamp
        timestamp = start_date + timedelta(
            days=random.randint(0, 89),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )

        # Determine transaction type
        txn_type = random.choices(
            list(transaction_types.keys()),
            weights=list(transaction_types.values())
        )[0]

        # Determine if fraudulent (8% fraud rate - more realistic)
        is_fraud = random.random() < 0.08

        if is_fraud:
            # Generate fraudulent transaction
            fraud_data = fraud_generator.generate_fraudulent_transaction(txn_type, user_data)
            amount = fraud_data['amount']
            receiver_id = fraud_data['receiver_id']
            location = fraud_data['location']
            device_id = fraud_data['device_id']
            fraud_type = fraud_data['fraud_type']
            merchant_type = 'Unknown'  # Fraudulent merchants are unknown
            # Add some noise - not all fraud has high probability
            fraud_prob = random.uniform(0.4, 0.95) if random.random() < 0.8 else random.uniform(0.1, 0.4)
        else:
            # Generate legitimate transaction
            legit_data = fraud_generator.generate_legitimate_transaction(
                txn_type, user_data, user_recent_txns[user_id]
            )
            amount = legit_data['amount']
            receiver_id = legit_data['receiver_id']
            location = legit_data['location']
            device_id = legit_data['device_id']
            merchant_type = legit_data['merchant_type']
            fraud_type = '-'
            # Add some noise - some legitimate transactions might look suspicious
            fraud_prob = random.uniform(0.01, 0.35) if random.random() < 0.9 else random.uniform(0.35, 0.6)

        # Set sender ID based on transaction type
        if txn_type == 'UPI':
            sender_id = user_data['upi_id']
        elif txn_type == 'Card':
            sender_id = user_data['card_id']
        else:
            sender_id = f"ACC_{random.randint(10000, 99999)}"

        # Generate transaction ID
        txn_id = f"{txn_type}_{int(timestamp.timestamp())}_{i}"

        # Track transaction for velocity features
        user_recent_txns[user_id].append({
            'timestamp': timestamp,
            'amount': amount,
            'location': location
        })

        # Keep only recent transactions (last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        user_recent_txns[user_id] = [
            txn for txn in user_recent_txns[user_id]
            if txn['timestamp'] > cutoff_time
        ]

        transactions.append([
            txn_id,
            timestamp.isoformat(),
            amount,
            sender_id,
            receiver_id,
            txn_type,
            merchant_type,
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
    with open('data/fraud_transactions.csv', 'w', newline='') as f:
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

    print(f"Generated {num_transactions} synthetic transactions with realistic fraud patterns")
    fraud_count = sum(1 for txn in transactions if txn[9])  # Is_Fraud column
    print(f"Fraud rate: {fraud_count/num_transactions:.1%}")

if __name__ == '__main__':
    generate_synthetic_transactions(1000)  # Generate 1000 transactions 