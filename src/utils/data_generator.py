import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

class TransactionDataGenerator:
    def __init__(self, n_transactions=10000, fraud_ratio=0.03):
        """
        Initialize the transaction data generator.
        
        Args:
            n_transactions (int): Number of transactions to generate
            fraud_ratio (float): Ratio of fraudulent transactions
        """
        self.n_transactions = n_transactions
        self.fraud_ratio = fraud_ratio
        self.fake = Faker()
        self.data = None
        
    def generate_data(self):
        """Generate synthetic transaction data with embedded fraud patterns."""
        # Generate base transaction data
        data = {
            "Transaction_ID": [f"Txn_{i:05d}" for i in range(self.n_transactions)],
            "Timestamp": [self.fake.date_time_this_year() for _ in range(self.n_transactions)],
            "Amount": [round(random.uniform(1, 10000), 2) for _ in range(self.n_transactions)],
            "Sender_ID": [f"User_{random.randint(1, 1000)}" for _ in range(self.n_transactions)],
            "Receiver_ID": [f"Merchant_{random.randint(1, 500)}" for _ in range(self.n_transactions)],
            "Transaction_Type": [random.choice(["UPI", "Card"]) for _ in range(self.n_transactions)],
            "Merchant_Type": [random.choice(["Online_Retail", "Food", "Travel", "Entertainment", "Utilities", "Unknown"]) for _ in range(self.n_transactions)],
            "Device_ID": [f"Device_{random.randint(1, 2000)}" for _ in range(self.n_transactions)],
            "Location": [f"{self.fake.latitude()}, {self.fake.longitude()}" for _ in range(self.n_transactions)],
            "Is_Fraud": [0] * self.n_transactions  # Initialize as legitimate
        }
        
        df = pd.DataFrame(data)
        
        # Sort by timestamp
        df = df.sort_values(by="Timestamp").reset_index(drop=True)
        
        # Inject fraud patterns
        fraud_indices = random.sample(range(self.n_transactions), int(self.n_transactions * self.fraud_ratio))
        df.loc[fraud_indices, "Is_Fraud"] = 1
        
        # Split fraud indices into different fraud types
        n_fraud_types = 9  # Total number of fraud types
        fraud_type_indices = np.array_split(fraud_indices, n_fraud_types)
        
        # 1. Phishing Links: High amounts to unknown receivers
        phishing_indices = fraud_type_indices[0]
        df.loc[phishing_indices, "Amount"] = [random.uniform(1000, 5000) for _ in range(len(phishing_indices))]
        df.loc[phishing_indices, "Receiver_ID"] = [f"Suspicious_{i}" for i in range(len(phishing_indices))]
        df.loc[phishing_indices, "Transaction_Type"] = "UPI"
        
        # 2. QR Code Scams: Debits to unknown merchants
        qr_indices = fraud_type_indices[1]
        df.loc[qr_indices, "Merchant_Type"] = "Unknown"
        df.loc[qr_indices, "Amount"] = [random.uniform(500, 2000) for _ in range(len(qr_indices))]
        df.loc[qr_indices, "Transaction_Type"] = "UPI"
        
        # 3. SIM Swap Attacks: New device, unusual location
        sim_swap_indices = fraud_type_indices[2]
        df.loc[sim_swap_indices, "Device_ID"] = [f"New_Device_{i}" for i in range(len(sim_swap_indices))]
        df.loc[sim_swap_indices, "Location"] = ["-34.6037, -58.3816" for _ in range(len(sim_swap_indices))]  # Example: Buenos Aires
        df.loc[sim_swap_indices, "Transaction_Type"] = "UPI"
        
        # 4. Fake UPI Apps: Irregular device and timing
        fake_app_indices = fraud_type_indices[3]
        df.loc[fake_app_indices, "Device_ID"] = [f"Unknown_Device_{i}" for i in range(len(fake_app_indices))]
        df.loc[fake_app_indices, "Timestamp"] = [self.fake.date_time_between(start_date="-1h", end_date="now") for _ in range(len(fake_app_indices))]
        df.loc[fake_app_indices, "Transaction_Type"] = "UPI"
        
        # 5. Small Testing Transactions: Multiple small amounts
        test_txn_indices = fraud_type_indices[4]
        df.loc[test_txn_indices, "Amount"] = [random.uniform(1, 10) for _ in range(len(test_txn_indices))]
        df.loc[test_txn_indices, "Transaction_Type"] = "UPI"
        
        # 6. Card Skimming: High-value transactions at physical merchants
        skimming_indices = fraud_type_indices[5]
        df.loc[skimming_indices, "Amount"] = [random.uniform(3000, 8000) for _ in range(len(skimming_indices))]
        df.loc[skimming_indices, "Merchant_Type"] = "Retail_Physical"
        df.loc[skimming_indices, "Transaction_Type"] = "Card"
        
        # 7. Data Breach Reuse: Online transactions with stolen card details
        data_breach_indices = fraud_type_indices[6]
        df.loc[data_breach_indices, "Merchant_Type"] = "Online_Retail"
        df.loc[data_breach_indices, "Transaction_Type"] = "Card"
        df.loc[data_breach_indices, "Amount"] = [random.uniform(1000, 4000) for _ in range(len(data_breach_indices))]
        
        # 8. Unusual Location/Activity: Transactions from new locations
        unusual_loc_indices = fraud_type_indices[7]
        df.loc[unusual_loc_indices, "Location"] = ["35.6762, 139.6503" for _ in range(len(unusual_loc_indices))]  # Example: Tokyo
        df.loc[unusual_loc_indices, "Transaction_Type"] = "Card"
        
        # 9. CNP Fraud: Online purchases with high amounts
        cnp_indices = fraud_type_indices[8]
        df.loc[cnp_indices, "Merchant_Type"] = "Online_Retail"
        df.loc[cnp_indices, "Transaction_Type"] = "Card"
        df.loc[cnp_indices, "Amount"] = [random.uniform(5000, 9000) for _ in range(len(cnp_indices))]
        
        self.data = df
        return df
    
    def save_data(self, output_path):
        """Save the generated data to a CSV file."""
        if self.data is None:
            self.generate_data()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        print(f"Synthetic dataset with fraud patterns saved as '{output_path}'.")
        
        # Print stats
        fraud_count = self.data["Is_Fraud"].sum()
        print(f"Total transactions: {len(self.data)}")
        print(f"Fraudulent transactions: {fraud_count} ({fraud_count/len(self.data)*100:.2f}%)")
        print(f"UPI transactions: {(self.data['Transaction_Type'] == 'UPI').sum()}")
        print(f"Card transactions: {(self.data['Transaction_Type'] == 'Card').sum()}")

if __name__ == "__main__":
    # Generate and save data
    generator = TransactionDataGenerator(n_transactions=10000, fraud_ratio=0.03)
    df = generator.generate_data()
    generator.save_data("../../data/fraud_transactions.csv") 