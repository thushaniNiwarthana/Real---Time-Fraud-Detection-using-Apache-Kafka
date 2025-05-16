from kafka import KafkaProducer
import json
import random
import time
import pandas as pd
import joblib
from features import FEATURE_ORDER, CATEGORICAL_FEATURES
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load('Model Reports/fraud_detection_model.pkl')
scaler = joblib.load('Model Reports/scaler.pkl')

producer = KafkaProducer(
    bootstrap_servers = 'localhost:9092',
    value_serializer = lambda v: json.dumps(v).encode('utf-8')
)

transaction_counter = 0
fraud_toggle = True  # Global flag to alternate between fraud and non-fraud transactions

def preprocess_transaction(transaction):
    #Preprocess transaction to match model input format
    transaction_time = time.localtime(transaction['Time'])
    
    # Create dictionary with all features in correct order
    processed = {
        'Bank_Branch': transaction['Bank_Branch'],
        'Account_Type': transaction['Account_Type'],
        'Transaction_Amount': transaction['Transaction_Amount'],
        'Transaction_Type': transaction['Transaction_Type'],
        'Merchant_Category': transaction['Merchant_Category'],
        'Account_Balance': transaction['Account_Balance'],
        'Transaction_Location': transaction['Transaction_Location'],
        'Device_Type': transaction['Device_Type'],
        'Transaction_Currency': transaction['Transaction_Currency'],
        'year': transaction_time.tm_year,
        'month': transaction_time.tm_mon,
        'day': transaction_time.tm_mday,
        'hour': transaction_time.tm_hour,
        'minute': transaction_time.tm_min,
        'second': transaction_time.tm_sec
    }
    
    # Encode categorical features 
    label_encoder = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        processed[col] = label_encoder.fit_transform([processed[col]])[0]
    
    return processed

def generate_transaction():
    global transaction_counter, fraud_toggle 
    transaction_counter += 1 
    
    # Generate base transaction with all features 
    transaction = {
        "Time": int(time.time()),
        "Customer_ID": f"CUST{random.randint(10000, 99999)}",
        "Bank_Branch": random.choice(["Los Angeles", "San Francisco", "New York", "Chicago", "Miami"]),
        "Account_Type": random.choice(["Savings", "Business", "Checking"]),
        "Transaction_ID": f"TXN{random.randint(100000, 999999)}",
        "Transaction_Date": time.strftime("%Y-%m-%d"),
        "Transaction_Time": time.strftime("%H:%M:%S"),
        "Transaction_Amount": round(random.uniform(10, 10000), 2),
        "Merchant_ID": f"MERC{random.randint(1000, 9999)}",
        "Transaction_Type": random.choice(["POS", "Bank Transfer", "ATM Withdrawal", "Online"]),
        "Merchant_Category": random.choice(["Electronics", "Dining", "Grocery", "Clothing", "Services"]),
        "Account_Balance": round(random.uniform(0, 100000), 2),
        "Transaction_Device": f"DEV{random.randint(100, 999)}",
        "Transaction_Location": f"LOC{random.randint(1, 50)}",
        "Device_Type": random.choice(["Mobile", "Desktop", "ATM", "POS Terminal"]),
        "Transaction_Currency": random.choice(["USD", "EUR", "GBP"]),
        "Customer_Contact": f"+1{random.randint(2000000000, 9999999999)}",
        "Transaction_Description": random.choice(["Purchase", "Withdrawal", "Transfer", "Payment"]),
        "Customer_Email": f"user{random.randint(1000, 9999)}@example.com"
    }
    
    try:
        # Alternate between fraud and non-fraud transactions
        if fraud_toggle:
            transaction["Transaction_Amount"] = round(random.uniform(5000, 10000), 2)
            transaction["Is_Fraud"] = 1
            # Add suspicious patterns for fraud 
            if random.random() > 0.7: # 30% chance to add suspicious patterns 
                transaction["Transaction_Location"] = "LOC99"  # Suspicious location  
                transaction["Transaction_Time"] = "03:00:05"  # Unusual time
        else:
            transaction["Transaction_Amount"] = round(random.uniform(10, 3000), 2) 
            transaction["Is_Fraud"] = 0 

        # Toggle fraud flag for the next transaction
        fraud_toggle = not fraud_toggle
        
        # Create formatted log entry
        log_entry = f"""\

====================================================================
New Transaction Sent:
--------------------------------------------------------------------
Transaction ID: {transaction['Transaction_ID']}
Customer ID: {transaction['Customer_ID']}
Amount: ${transaction['Transaction_Amount']:.2f}
Type: {transaction['Transaction_Type']}
Merchant: {transaction['Merchant_Category']}
Location: {transaction['Transaction_Location']}
Device: {transaction['Device_Type']}
Date: {transaction['Transaction_Date']}
Time: {transaction['Transaction_Time']}
Fraud Status: {'FRAUD' if transaction['Is_Fraud'] == 1 else 'VALID'}
===================================================================="""
        
        # Write to log file
        with open("producer.log", "a") as log_file:
            log_file.write(log_entry + "\n")
        
        # Print to terminal
        print(log_entry) 
        
        return transaction
    
    except Exception as e:
        print(f"Error generating transaction: {str(e)}") 
        return None

if __name__ == "__main__":
    print("Starting transaction producer...") 
    print(f"Using feature order: {FEATURE_ORDER}")
    
    while True:
        transaction = generate_transaction()
        if transaction:
            producer.send('fraud_topic', transaction)
        time.sleep(2)