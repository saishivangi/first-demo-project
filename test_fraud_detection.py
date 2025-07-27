#!/usr/bin/env python3
"""
Test script for UPI Fraud Detection System
This script demonstrates the API functionality and tests different transaction scenarios.
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:5000"

def test_api_prediction(transaction_data, description):
    """Test the fraud detection API with given transaction data"""
    print(f"\n🔍 Testing: {description}")
    print("=" * 60)
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict", json=transaction_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: Success")
            print(f"🎯 Prediction: {'🚨 FRAUD' if result['prediction'] == 1 else '✅ LEGITIMATE'}")
            print(f"📊 Probability: {result['probability']:.2%}")
            print(f"🔐 Confidence: {result['explanation']['confidence']}")
            
            if result['explanation']['risk_factors']:
                print(f"⚠️  Risk Factors:")
                for factor in result['explanation']['risk_factors']:
                    print(f"   • {factor}")
            else:
                print(f"✅ No significant risk factors detected")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to the server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main test function"""
    print("🛡️  UPI Fraud Detection System - API Test")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "data": {
                "sender_upi": "john.doe@paytm",
                "receiver_upi": "grocery.store@googlepay",
                "amount": 500.0,
                "transaction_mode": "UPI_PIN",
                "transaction_time": "14:30",
                "day_of_week": "Tuesday",
                "transaction_frequency": 5,
                "account_age_days": 365
            },
            "description": "Normal grocery transaction - Should be LEGITIMATE"
        },
        {
            "data": {
                "sender_upi": "suspicious.user@phonepe",
                "receiver_upi": "unknown.account@ybl",
                "amount": 85000.0,
                "transaction_mode": "QR_CODE",
                "transaction_time": "23:45",
                "day_of_week": "Sunday",
                "transaction_frequency": 25,
                "account_age_days": 15
            },
            "description": "High-risk transaction - Should be FRAUD"
        },
        {
            "data": {
                "sender_upi": "regular.customer@googlepay",
                "receiver_upi": "shop.owner@paytm",
                "amount": 2500.0,
                "transaction_mode": "BIOMETRIC",
                "transaction_time": "16:15",
                "day_of_week": "Friday",
                "transaction_frequency": 8,
                "account_age_days": 180
            },
            "description": "Regular shopping transaction - Should be LEGITIMATE"
        },
        {
            "data": {
                "sender_upi": "newbie@okaxis",
                "receiver_upi": "newbie@okaxis",
                "amount": 99999.0,
                "transaction_mode": "BANK_TRANSFER",
                "transaction_time": "02:30",
                "day_of_week": "Saturday",
                "transaction_frequency": 50,
                "account_age_days": 5
            },
            "description": "Same sender/receiver with high amount - Should be FRAUD"
        },
        {
            "data": {
                "sender_upi": "office.employee@googlepay",
                "receiver_upi": "company.canteen@phonepe",
                "amount": 150.0,
                "transaction_mode": "QR_CODE",
                "transaction_time": "12:30",
                "day_of_week": "Wednesday",
                "transaction_frequency": 3,
                "account_age_days": 450
            },
            "description": "Office lunch payment - Should be LEGITIMATE"
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        test_api_prediction(test_case["data"], f"Test {i}: {test_case['description']}")
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📌 To manually test the web interface:")
    print(f"   Open your browser and go to: {BASE_URL}")
    print("\n📊 Key features to test:")
    print("   • Dashboard: Overview and features")
    print("   • Fraud Detection: Interactive form with real-time analysis")
    print("   • Train Model: ML model training with different dataset sizes")
    print("   • Analytics: Comprehensive fraud statistics and charts")

if __name__ == "__main__":
    main()