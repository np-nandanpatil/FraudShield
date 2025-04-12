import requests
import json

# Create a test transaction that requires confirmation
def create_test_transaction():
    url = "http://localhost:5000/process_transaction"
    
    # Sample transaction data matching the screenshot
    transaction_data = {
        "Amount": 9.10,
        "Transaction_Type": "UPI",
        "Merchant_Type": "Online_Retail",
        "Sender_ID": "rpr092004@okaxis",  # UPI ID from screenshot
        "Receiver_ID": "SynthHack Demo Store",
        "Device_ID": "Regular Device",
        "Location": "Bangalore",
        "Order_ID": "ORD-474420"
    }
    
    # Send the request
    response = requests.post(url, json=transaction_data)
    
    # Print the response
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    
    # If successful, get the transaction ID and redirect to confirmation page
    if response.status_code == 200:
        try:
            data = response.json()
            if "transaction_id" in data:
                print(f"\nTransaction created successfully!")
                print(f"Transaction ID: {data['transaction_id']}")
                print(f"Please visit: http://localhost:5000/confirm_transaction/{data['transaction_id']}")
        except:
            print("Could not parse JSON response")

if __name__ == "__main__":
    create_test_transaction() 