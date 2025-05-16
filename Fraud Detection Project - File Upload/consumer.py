from flask import Flask, render_template
import threading
import time
from datetime import datetime
from kafka import KafkaConsumer
import json
import pandas as pd
import psycopg2
import joblib
from features import FEATURE_ORDER, CATEGORICAL_FEATURES
from sklearn.preprocessing import LabelEncoder
import traceback

# Flask app
app = Flask(__name__)

# Global variables to store transactions and counters
transactions = []
fraud_count = 0
non_fraud_count = 0

KAFKA_CONFIG = {
    'bootstrap_servers': 'localhost:9092',
    'group_id': 'fraud_detection_consumer'
}

#Establish a connection to the PostgreSQL database.
def get_db_connection():
    return psycopg2.connect(
        dbname = "fraud_detection_system",
        user = "postgres",
        password = "0808", 
        host = "localhost",
        port = "9876"
    )
#Safely retrieve a value from the transaction dictionary.
def get_transaction_value(transaction, key, default = None): 
    return transaction.get(key, default) 

def preprocess_transaction(transaction):
    #Preprocess transaction to match model input format.
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
    for col in ['Transaction_Type', 'Merchant_Category', 'Transaction_Location', 'Device_Type', 'Account_Type', 'Bank_Branch', 'Transaction_Currency']:
        processed[col] = label_encoder.fit_transform([processed[col]])[0]
    
    return processed

# Function to process a transaction and make predictions and return results
def process_transaction(transaction, model, scaler):
    processed = preprocess_transaction(transaction)
    features = pd.DataFrame([processed], columns=FEATURE_ORDER)
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])
    return prediction, probability, processed

# Function to format transaction details for display.
def display_transaction(transaction, prediction, probability):
    return {
        "transaction_id": transaction.get('Transaction_ID', 'N/A'),
        "customer_id": transaction.get('Customer_ID', 'N/A'),
        "amount": f"${transaction.get('Transaction_Amount', 0):.2f}",
        "type": transaction.get('Transaction_Type', 'N/A'),
        "merchant": transaction.get('Merchant_Category', 'N/A'),
        "location": transaction.get('Transaction_Location', 'N/A'),
        "device": transaction.get('Device_Type', 'N/A'),
        "date": transaction.get('Transaction_Date', 'N/A'),
        "time": transaction.get('Transaction_Time', 'N/A'),
        "fraud_status": "FRAUD" if prediction == 1 else "VALID",
        "fraud_probability": f"{probability:.4f}"
    }

# Function to consume transactions from Kafka and process them.
def consume_transactions():
    global fraud_count, non_fraud_count

    MODEL_PATH = "Model Reports/fraud_detection_model.pkl"
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('Model Reports/scaler.pkl')
    
    print("Connecting to Kafka...")
    consumer = KafkaConsumer(
        'fraud_topic',
        value_deserializer = lambda m: json.loads(m.decode('utf-8')),
        bootstrap_servers = 'localhost:9092',
        group_id = 'fraud_detection_consumer'
    )
    print("Connected to Kafka. Waiting for messages...")
    
    try:    
        # Loop to consume messages from Kafka
        for message in consumer:
            print("Message received:", message.value)
            transaction = message.value 
            
            # Preprocess the transaction
            processed = preprocess_transaction(transaction)
            features = pd.DataFrame([processed], columns=FEATURE_ORDER)
            features_scaled = scaler.transform(features)
            
            # Predict fraud and probability
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0][1])
            
            # Update counters
            if prediction == 1:
                fraud_count += 1
            else:
                non_fraud_count += 1
            
            # Log transaction
            log_entry = f"""\

====================================================================
New Transaction Received:
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
Fraud Prediction: {'FRAUD' if prediction == 1 else 'VALID'}
Fraud Probability: {probability:.4f}
===================================================================="""
            
            # Write to log file
            with open("consumer.log", "a") as log_file:
                log_file.write(log_entry + "\n")
            
            print(log_entry)

    except Exception as e:
        print(f"Error consuming transactions: {str(e)}\n{traceback.format_exc()}")
    finally:
        consumer.close()

# Flask route to display transactions
@app.route("/")
def index():
    return render_template("dashboard.py", transactions = transactions)

# Start the Flask app in a separate thread
def start_flask():
    app.run(host = "0.0.0.0", port = 5000, debug = False)

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target = start_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start consuming transactions
    consume_transactions()