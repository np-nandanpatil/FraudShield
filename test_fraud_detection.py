#!/usr/bin/env python3
"""
Test script to validate the improved fraud detection system
"""

import requests
import json
import time

def test_fraud_detection():
    """Test the fraud detection with various transaction scenarios"""

    base_url = "http://127.0.0.1:5000"

    # Test cases with different fraud scenarios
    test_cases = [
        {
            "name": "Legitimate UPI transaction",
            "data": {
                "Transaction_ID": "TEST_LEGIT_UPI_001",
                "Amount": 500.0,
                "Sender_ID": "user1234@okhdfcbank.com",  # Valid UPI format
                "Receiver_ID": "merchant456@okaxis.com",
                "Transaction_Type": "UPI",
                "Merchant_Type": "Food",
                "Device_ID": "Device_1234",
                "Location": "12.9716, 77.5946",  # Bangalore
                "Timestamp": "2025-01-20T10:30:00"
            },
            "expected_fraud": False
        },
        {
            "name": "Suspicious UPI transaction (QR scam)",
            "data": {
                "Transaction_ID": "TEST_FRAUD_UPI_001",
                "Amount": 2000.0,
                "Sender_ID": "user5678@oksbi.com",  # Valid UPI format
                "Receiver_ID": "suspicious999@paytm.com",  # Suspicious receiver
                "Transaction_Type": "UPI",
                "Merchant_Type": "Unknown",
                "Device_ID": "Device_5678",
                "Location": "28.6139, 77.2090",  # Delhi (different from usual)
                "Timestamp": "2025-01-20T10:35:00"
            },
            "expected_fraud": True
        },
        {
            "name": "Small testing transaction (Card)",
            "data": {
                "Transaction_ID": "TEST_FRAUD_CARD_001",
                "Amount": 10.0,
                "card_number": "4532015112830366",  # Valid test card number
                "cvv": "123",
                "expiry": "12/26",
                "Receiver_ID": "Test_Merchant_001",
                "Transaction_Type": "Card",
                "Merchant_Type": "Unknown",
                "Device_ID": "Device_9999",
                "Location": "12.9716, 77.5946",
                "Timestamp": "2025-01-20T10:40:00"
            },
            "expected_fraud": True
        },
        {
            "name": "Legitimate Card transaction",
            "data": {
                "Transaction_ID": "TEST_LEGIT_CARD_001",
                "Amount": 2500.0,
                "card_number": "4111111111111111",  # Valid test card number
                "cvv": "456",
                "expiry": "08/27",
                "Receiver_ID": "Merchant_789",
                "Transaction_Type": "Card",
                "Merchant_Type": "Online_Retail",
                "Device_ID": "Device_1111",
                "Location": "12.9716, 77.5946",
                "Timestamp": "2025-01-20T10:45:00"
            },
            "expected_fraud": False
        },
        {
            "name": "SIM Swap fraud (device change + location)",
            "data": {
                "Transaction_ID": "TEST_FRAUD_SIM_001",
                "Amount": 15000.0,
                "Sender_ID": "user9999@okicici.com",  # Valid UPI format
                "Receiver_ID": "merchant123@paytm.com",
                "Transaction_Type": "UPI",
                "Merchant_Type": "Travel",
                "Device_ID": "Unknown_Device_456",  # Unknown device
                "Location": "19.0760, 72.8777",  # Mumbai (different city)
                "Timestamp": "2025-01-20T10:50:00"
            },
            "expected_fraud": True
        }
    ]

    print("Testing Improved Fraud Detection System")
    print("=" * 50)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 40)

        try:
            # Make request to /process_transaction endpoint
            response = requests.post(
                f"{base_url}/process_transaction",
                json=test_case['data'],
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                is_fraud_detected = result.get('status') == 'blocked'
                fraud_probability = result.get('confidence', 0)
                fraud_type = result.get('reason', 'None')

                print(f"Status: {result.get('status', 'Unknown')}")
                print(f"Fraud Detected: {is_fraud_detected}")
                print(f"Fraud Probability: {fraud_probability:.4f}")
                print(f"Fraud Type: {fraud_type}")
                print(f"Expected Fraud: {test_case['expected_fraud']}")

                # Check if detection matches expectation
                correct = is_fraud_detected == test_case['expected_fraud']
                print(f"Correct Detection: {'YES' if correct else 'NO'}")

                results.append({
                    'test': test_case['name'],
                    'expected': test_case['expected_fraud'],
                    'detected': is_fraud_detected,
                    'probability': fraud_probability,
                    'correct': correct
                })

            else:
                print(f"Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        time.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    correct_detections = sum(1 for r in results if r['correct'])
    total_tests = len(results)

    print(f"Total Tests: {total_tests}")
    print(f"Correct Detections: {correct_detections}")
    print(f"Accuracy: {correct_detections/total_tests*100:.1f}%")

    # Show fraud detection stats
    fraud_cases = [r for r in results if r['expected']]
    legit_cases = [r for r in results if not r['expected']]

    if fraud_cases:
        fraud_detected = sum(1 for r in fraud_cases if r['detected'])
        print(f"Fraud Cases Detected: {fraud_detected}/{len(fraud_cases)} ({fraud_detected/len(fraud_cases)*100:.1f}%)")

    if legit_cases:
        legit_correct = sum(1 for r in legit_cases if not r['detected'])
        print(f"Legitimate Cases Correct: {legit_correct}/{len(legit_cases)} ({legit_correct/len(legit_cases)*100:.1f}%)")

    print("\nTest completed!")

if __name__ == "__main__":
    test_fraud_detection()