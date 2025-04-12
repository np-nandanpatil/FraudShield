import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def combine_transactions():
    # Read the fraud transactions dataset
    fraud_df = pd.read_csv('data/fraud_transactions.csv')
    
    # Convert Is_Fraud to int (1 or 0)
    fraud_df['Is_Fraud'] = fraud_df['Is_Fraud'].astype(int)
    
    # Map Is_Fraud to Status
    fraud_df['Status'] = fraud_df['Is_Fraud'].map({1: 'Fraud', 0: 'Success'})
    
    # Add Fraud_Type
    fraud_types = ['Account Takeover', 'Identity Theft', 'Merchant Fraud', 'Phishing']
    fraud_df['Fraud_Type'] = fraud_df.apply(
        lambda row: random.choice(fraud_types) if row['Is_Fraud'] == 1 else 'None',
        axis=1
    )
    
    # Add Fraud_Probability
    fraud_df['Fraud_Probability'] = fraud_df.apply(
        lambda row: round(0.7 + 0.3 * np.random.random(), 4) if row['Is_Fraud'] == 1 
        else round(0.2 * np.random.random(), 4),
        axis=1
    )
    
    # Convert timestamp to datetime if it's not already
    fraud_df['Timestamp'] = pd.to_datetime(fraud_df['Timestamp'])
    
    # Get a mix of fraud and non-fraud transactions
    fraud_transactions = fraud_df[fraud_df['Is_Fraud'] == 1].sort_values('Timestamp', ascending=False).head(100)
    non_fraud_transactions = fraud_df[fraud_df['Is_Fraud'] == 0].sort_values('Timestamp', ascending=False).head(400)
    
    # Combine and shuffle the transactions
    combined_df = pd.concat([fraud_transactions, non_fraud_transactions])
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Ensure Is_Fraud is int
    combined_df['Is_Fraud'] = combined_df['Is_Fraud'].astype(int)
    
    # Save to processed_transactions.csv
    combined_df.to_csv('data/processed_transactions.csv', index=False)
    print(f"Processed {len(combined_df)} transactions and saved to data/processed_transactions.csv")
    print(f"Fraud transactions: {len(fraud_transactions)}")
    print(f"Non-fraud transactions: {len(non_fraud_transactions)}")

if __name__ == "__main__":
    combine_transactions() 