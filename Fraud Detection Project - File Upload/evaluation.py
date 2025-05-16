import os
import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, precision_recall_curve
from datetime import datetime
from dash import dcc, html
from dash.dependencies import Input, Output
import dash

app = dash.Dash(__name__)


@app.callback(
    [Output("metrics-table", "children"), Output("fraud-patterns-table", "children")],
    Input("interval-evaluation", "n_intervals")
)

# Update the evaluation tables
def update_evaluation_tables(n):
    metrics, fraud_patterns = calculate_evaluation_metrics()
    
    # Create metrics table
    metrics_table = html.Table(
        # Table header
        [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))] +
        # Table body
        [html.Tbody([
            html.Tr([html.Td(metric["Metric"]), html.Td(metric["Value"])]) for metric in metrics
        ])],
        className = "metrics-table"
    )
    
    # Create fraud patterns table
    fraud_patterns_table = html.Table(
        # Table header
        [html.Thead(html.Tr([html.Th("Fraud Pattern"), html.Th("Value")]))] +
        # Table body
        [html.Tbody([
            html.Tr([html.Td(pattern["Metric"]), html.Td(pattern["Value"])]) for pattern in fraud_patterns
        ])],
        className = "fraud-patterns-table"
    )
    
    return metrics_table, fraud_patterns_table

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Evaluation Metrics", className = "page-title"),
    dcc.Interval(id = "interval-evaluation", interval = 5000, n_intervals = 0),
    html.Div(id = "metrics-table", children = "Metrics table will appear here."),
    html.Hr(),
    html.H2("Fraud Patterns", className = "section-title"),
    html.Div(id = "fraud-patterns-table", children = "Fraud patterns table will appear here.")
])

# Set up the server
if __name__ == "__main__": 
    app.run_server(debug = True)

def parse_log_entry(log_entry):
    #Parse a single log entry from consumer.log into a dictionary.
    data = {}
    lines = log_entry.split('\n')
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            data[key] = value
    
    # Convert statuses to binary
    data['actual_fraud'] = 1 if data.get('actual_fraud_status', '').upper() == 'FRAUD' else 0
    data['predicted_fraud'] = 1 if data.get('fraud_prediction', '').upper() == 'FRAUD' else 0
    
    # Extract numeric probability
    try:
        data['probability'] = float(data.get('fraud_probability', 0.0))
    except (ValueError, KeyError):
        data['probability'] = 0.0
    
    # Parse amount (remove $ sign)
    if 'amount' in data:
        try:
            data['amount'] = float(data['amount'].replace('$', '').strip())
        except (ValueError, KeyError):
            data['amount'] = 0.0
    
    return data

def calculate_optimal_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = f1_scores.argmax()
    return thresholds[optimal_idx]

def calculate_evaluation_metrics():
    if not os.path.exists("consumer.log"):
        return [], []  # Return empty lists if no data is available
    
    try:
        with open("consumer.log", "r") as f:
            content = f.read()
            transactions = content.split('====================================================================') 
        
        parsed_data = []
        for transaction in transactions:
            if transaction.strip():
                parsed = parse_log_entry(transaction)
                if parsed and 'actual_fraud' in parsed and 'predicted_fraud' in parsed:
                    parsed_data.append(parsed)
        
        if not parsed_data:
            return [], []  # No valid transactions to evaluate
        
        df = pd.DataFrame(parsed_data)
        
        # Check for required columns
        required_cols = ['actual_fraud', 'predicted_fraud', 'probability']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return [{"Metric": "Error", "Value": f"Missing required fields: {', '.join(missing_cols)}"}], []
        
        # Clean data
        df = df.dropna(subset = required_cols)
        df['actual_fraud'] = pd.to_numeric(df['actual_fraud'], errors = 'coerce')
        df['predicted_fraud'] = pd.to_numeric(df['predicted_fraud'], errors = 'coerce')
        df['probability'] = pd.to_numeric(df['probability'], errors = 'coerce')
        df = df.dropna(subset = required_cols)
        
        if len(df) == 0:
            return [{"Metric": "Error", "Value": "No valid transactions after data cleaning"}], []
        
        # Calculate metrics
        y_true = df['actual_fraud']
        y_pred = df['predicted_fraud']
        y_proba = df['probability']
        
        optimal_threshold = calculate_optimal_threshold(y_true, y_proba)
        y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)

        precision = precision_score(y_true, y_pred_adjusted, zero_division = 0)
        recall = recall_score(y_true, y_pred_adjusted, zero_division = 0)
        f1 = f1_score(y_true, y_pred_adjusted, zero_division = 0)
        accuracy = accuracy_score(y_true, y_pred_adjusted) * 100  # Convert accuracy to percentage
        
        # Calculate ROC AUC score
        try:
            roc_auc = roc_auc_score(y_true, y_proba) if len(y_true.unique()) > 1 else "Undefined"
        except ValueError:
            roc_auc = "Undefined"
        
        metrics = [
            {"Metric": "Precision", "Value": f"{precision:.4f}"},
            {"Metric": "Recall", "Value": f"{recall:.4f}"},
            {"Metric": "F1 Score", "Value": f"{f1:.4f}"},
            {"Metric": "Accuracy", "Value": f"{accuracy:.2f}%"},  # Format accuracy as a percentage
            {"Metric": "ROC AUC", "Value": roc_auc},
            {"Metric": "Total Transactions", "Value": len(df)},
            {"Metric": "Actual Fraudulent", "Value": sum(y_true)},
            {"Metric": "Predicted Fraudulent", "Value": sum(y_pred_adjusted)},
            {"Metric": "Average Fraud Probability", "Value": f"{y_proba.mean():.4f}"}
        ]
        
        # Add fraud patterns if any fraud exists
        fraud_patterns = []
        if sum(y_true) > 0:
            fraud_df = df[y_true == 1]
            fraud_patterns = [
                {"Metric": "Most Common Location", "Value": fraud_df['location'].mode()[0] if not fraud_df['location'].mode().empty else "N/A"},
                {"Metric": "Average Fraud Amount", "Value": f"${fraud_df['amount'].mean():.2f}"},
                {"Metric": "Most Common Transaction Type", "Value": fraud_df['type'].mode()[0] if not fraud_df['type'].mode().empty else "N/A"}
            ]
        
        
        return metrics, fraud_patterns

    except Exception as e:
        return [{"Metric": "Error", "Value": f"Error calculating metrics: {str(e)}"}], []

def get_latest_counts():
    #Read latest counts from counts.log
    fraud_count = 0
    non_fraud_count = 0
    
    if os.path.exists("counts.log"):
        with open("counts.log", "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.startswith("Fraud Count:"):
                    fraud_count = int(line.split(":")[1].strip())
                elif line.startswith("Non-Fraud Count:"):
                    non_fraud_count = int(line.split(":")[1].strip())
                if fraud_count > 0 or non_fraud_count > 0:
                    break
                    
    return fraud_count, non_fraud_count