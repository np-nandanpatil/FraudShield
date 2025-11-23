import requests
import json

# Test data that should trigger fraud detection
test_data = {
    "Transaction_ID": "TEST_FRAUD_001",
    "Timestamp": "2025-11-23T23:45:00.000Z",
    "Amount": 5000.00,  # High amount
    "Sender_ID": "suspicious@user.com",  # Suspicious sender
    "Receiver_ID": "Merchant_999",
    "Transaction_Type": "UPI",
    "Merchant_Type": "Unknown",  # Unknown merchant
    "Device_ID": "Unknown_Device_123",  # Unknown device
    "Location": "12.9716, 77.5946"
}

url = "http://127.0.0.1:5000/process_transaction"

try:
    response = requests.post(url, json=test_data, headers={'Content-Type': 'application/json'})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")