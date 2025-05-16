import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import time
import joblib
from evaluation import calculate_evaluation_metrics
import json

# Set up the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions = True) #__name__ is used to identify the main module

# Enhanced Log Functions 
def read_transaction_logs(file_path):
    #Read logs and preserve terminal formatting, filtering out invalid entries.
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            transactions = [
                t.strip() for t in content.split('=' * 60 + '\n') 
                if t.strip() and "Transaction ID" in t 
            ]
            return transactions[-10:]
    return ["No transactions available"]

def get_model_metrics():
    try:
        model = joblib.load('Model Reports/fraud_detection_model.pkl')
        return html.Div([
            html.H4("Model Information"),
            html.P(f"Model Type: {type(model).__name__}"),
            html.P("Last trained on: " + time.ctime(os.path.getmtime('Model Reports/fraud_detection_model.pkl')))
        ])
    except:
        return html.P("Model not trained yet")

# Load metrics from JSON file 
def load_metrics():
    metrics_file = "Model Reports/model_metrics.json" 
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}

# Serve charts dynamically from the Model Reports folder
def get_chart_images():
    output_folder = "Model Reports"
    if os.path.exists(output_folder):
        return [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')]
    return []

# Layout of the app
app.layout = html.Div([
    html.Div([
        html.H2("Fraud Detection Dashboard", className = "menu-title"),
        html.Div("Dashboard", className = "menu-item", id="menu-dashboard"),
        html.Div("Trained Model", className = "menu-item", id = "menu-trained-model"),
        html.Div("Producer", className = "menu-item", id = "menu-producer"),
        html.Div("Consumer", className = "menu-item", id = "menu-consumer"),
        html.Div("Evaluation", className = "menu-item", id = "menu-evaluation")
       
    ], className = "sidebar"),

    html.Div(id = "page-content", className = "content"),
    #html.Div(id="model-comparison-table", className="comparison-container"),
])

#  Callbacks
@app.callback(
    Output("page-content", "children"),
    [Input("menu-dashboard", "n_clicks"),
     Input("menu-trained-model", "n_clicks"),
     Input("menu-producer", "n_clicks"),
     Input("menu-consumer", "n_clicks"),
     Input("menu-evaluation", "n_clicks")
     ]
)
def display_page(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div([
            html.H1("Fraud Detection Dashboard"),
            html.P("This dashboard provides real-time monitoring and evaluation of transactions for fraud detection.")
        ])

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "menu-dashboard":
        return html.Div([
            html.H1("Fraud Detection Dashboard", className = "section-title"),
            html.Div([
                html.Div([
                    html.H4("📊 Dashboard Overview", className = "metric-card-title"),
                    html.P("This is the landing page that provides summaries and quick links to different sections of the dashboard."),
                ], className = "metric-card"),
                html.Div([
                    html.H4("🚀 Producer", className = "metric-card-title"),
                    html.P("Sends real-time transactions into the system via Kafka. The logs show details of each transaction being streamed."),
                ], className = "metric-card"),

                html.Div([
                    html.H4("🧠 Consumer", className = "metric-card-title"),
                    html.P("Consumes transactions, runs them through the ML model, and predicts if a transaction is fraudulent. Updates the logs and fraud counts in real-time."),
                ], className = "metric-card"),

                html.Div([
                    html.H4("📈 Evaluation", className = "metric-card-title"),
                    html.P("Displays performance metrics like Precision, Recall, F1 Score, and insights such as common fraud patterns, based on recent transactions."),
                ], className = "metric-card"),

                html.Div([
                    html.H4("🤖 Trained Model", className = "metric-card-title"),
                    html.P("Shows current model details: model type, evaluation scores, training time, and visual performance reports."),
                ], className = "metric-card"),

            ], className = "metrics-container")
        ])

    elif button_id == "menu-producer":
        return html.Div([
            html.H3("Live Producer Transactions"),
            dcc.Interval(id = "interval-producer", interval = 2000, n_intervals = 0),
            html.Div(id = "producer-log", className = "log-container")
        ])
    elif button_id == "menu-consumer":
        return html.Div([
            html.H3("Live Consumer Fraud Detection"),
            dcc.Interval(id = "interval-consumer", interval = 2000, n_intervals = 0),
            html.Div(id = "consumer-log", className = "log-container"),
            html.Div(id = "counters", className = "counter-container")  # Define the counters component here
        ])
    elif button_id == "menu-evaluation":
        return html.Div([
            html.H3("Evaluation Metrics"),
            dcc.Interval(id = 'interval-evaluation', interval = 2000, n_intervals = 0),
            html.Div(id = "metrics-table", className = "metrics-container"),  
            html.Div(id = "fraud-patterns-table", className = "metrics-container") 
        ])
    elif button_id == "menu-trained-model":
        return html.Div([
            html.H3("Trained Model Information: "),
            get_model_metrics(),
            html.Div(id = "retrain-output"),
            html.Div(id = "trained-model-content")
        ])

    return html.H1("Select a Section")

def read_count_logs():
    fraud_count = "0"
    non_fraud_count = "0"
    if os.path.exists("counts.log"):
        with open("counts.log", "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.startswith("Fraud Count:"):
                    fraud_count = line.split(":")[1].strip()
                elif line.startswith("Non-Fraud Count:"):
                    non_fraud_count = line.split(":")[1].strip()
                if fraud_count != "0" and non_fraud_count != "0":
                    break
    return fraud_count, non_fraud_count

@app.callback(Output("producer-log", "children"), Input("interval-producer", "n_intervals"))
def update_producer_log(n):
    logs = read_transaction_logs("producer.log")
    output = []
    
    for log in logs:
        if not log.strip():
            continue
            
        lines = log.split('\n')
        transaction_block = html.Div(className = "transaction-block", children = [
            html.Div("=" * 60, className = "transaction-divider"),
            html.Div("New Transaction Sent:", className = "transaction-header"),
            html.Div("-" * 60, className = "transaction-divider"),
            *[html.Div(line.strip(), className = "transaction-line") 
              for line in lines if line.strip() and '===' not in line and '---' not in line],
            html.Div("=" * 60, className = "transaction-divider")
        ])
        output.append(html.Pre(transaction_block, className = "log-pre"))
    
    return output if output else "No transactions available"

@app.callback(
    [Output("consumer-log", "children"), Output("counters", "children")],
    Input("interval-consumer", "n_intervals")
)
def update_consumer_log_and_counters(n):
    logs = read_transaction_logs("consumer.log")
    output = []
    
    for log in logs:
        lines = log.split('\n')
        transaction_block = html.Div(className = "transaction-block", children = [
            html.Div("=" * 60, className = "transaction-divider"),
            html.Div("New Transaction Received:", className = "transaction-header"),
            html.Div("-" * 60, className = "transaction-divider"),
            *[html.Div(line.strip(), className = "transaction-line") 
              for line in lines if line.strip() and '===' not in line and '---' not in line],
            html.Div("=" * 60, className = "transaction-divider")
        ])
        output.append(html.Pre(transaction_block, className = "log-pre"))
    
    # Get counts
    fraud_count, non_fraud_count = read_count_logs()
    
    counters = html.Div([
        html.Div(f"Fraudulent: {fraud_count}", className = "counter", id = "fraud-counter"),
        html.Div(f"Non-Fraudulent: {non_fraud_count}", className = "counter", id = "non-fraud-counter")
    ], className = "counters-panel")
    
    return (output if output else "No transactions available"), counters

@app.callback(
    [Output("metrics-table", "children"), Output("fraud-patterns-table", "children")],
    Input("interval-evaluation", "n_intervals")
)
def update_evaluation_tables(n):
    metrics, fraud_patterns = calculate_evaluation_metrics()
    
    # Create metrics table
    metrics_table = html.Table(
        # Table header
        [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))] +
        # Table body
        [html.Tbody([
            *[html.Tr([html.Td(metric["Metric"]), html.Td(metric["Value"])]) for metric in metrics]
        ])],
        className = "metrics-table"
    )
    
    # Create fraud patterns table
    fraud_patterns_table = html.Table(
        # Table header
        [html.Thead(html.Tr([html.Th("Fraud Pattern"), html.Th("Value")]))] +
        # Table body
        [html.Tbody([
            *[html.Tr([html.Td(pattern["Metric"]), html.Td(pattern["Value"])]) for pattern in fraud_patterns]
        ])],
        className = "fraud-patterns-table"
    )
    
    return metrics_table, fraud_patterns_table

@app.callback(
    Output("trained-model-content", "children"),
    Input("menu-trained-model", "n_clicks")
)
def display_trained_model(n_clicks):
    if n_clicks is None:
        return html.Div("No model training information available.")
    
    metrics = load_metrics()
    charts = get_chart_images()
    
    # Create a table for metrics
    metrics_table = html.Table(
        # Table header
        [html.Thead(html.Tr([html.Th("Model"), html.Th("Accuracy"), html.Th("Precision"), html.Th("Recall"), 
                             html.Th("F1-Score"), html.Th("ROC AUC"), html.Th("Specificity"), 
                             html.Th("False Positive Rate"), html.Th("False Negative Rate"), 
                             html.Th("Matthews Corrcoef"), html.Th("Log Loss")]))] +
        # Table body
        [html.Tbody([
            html.Tr([
                html.Td(model_name),
                html.Td(f"{model_metrics['accuracy']:.4f}"),
                html.Td(f"{model_metrics['precision']:.4f}"),
                html.Td(f"{model_metrics['recall']:.4f}"),
                html.Td(f"{model_metrics['f1-score']:.4f}"),
                html.Td(f"{model_metrics['roc_auc']:.4f}"),
                html.Td(f"{model_metrics['specificity']:.4f}"),
                html.Td(f"{model_metrics['false_positive_rate']:.4f}"),
                html.Td(f"{model_metrics['false_negative_rate']:.4f}"),
                html.Td(f"{model_metrics['matthews_corrcoef']:.4f}"),
                html.Td(f"{model_metrics['log_loss']:.4f}")
            ]) for model_name, model_metrics in metrics.items()
        ])],
        className="metrics-table"
    )
    
    # Group charts by topics
    grouped_charts = {}
    for chart in charts:
        if "confusion_matrix" in chart:
            topic = "Confusion Matrix"
        elif "roc_curve" in chart:
            topic = "ROC Curve"
        elif "metrics_report" in chart:
            topic = "Metrics Report"
        else:
            topic = "Feature Importance"
        
        if topic not in grouped_charts:
            grouped_charts[topic] = []
        grouped_charts[topic].append(chart)
    
    # Create a grid layout for charts
    charts_display = []
    for topic, topic_charts in grouped_charts.items():
        topic_section = html.Div([
            html.H4(topic),
            html.Div([
                html.Img(
                    src = f"data:image/png;base64,{base64.b64encode(open(chart, 'rb').read()).decode('ascii')}",
                    style = {"width": "30%", "margin": "10px"}
                ) for chart in topic_charts
            ], className = "chart-grid")
        ], className = "topic-section")
        charts_display.append(topic_section)
    
    return html.Div([
        html.H3("Model Metrics"),
        metrics_table,
        html.H3("Model Charts"),
        html.Div(charts_display, className = "charts-container")
    ])

@app.callback(
    Output("model-comparison-table", "children"),
    Input("interval-evaluation", "n_intervals")
)
def update_model_comparison_table(n):
    metrics = load_metrics()
    comparison_table = html.Table(
        [html.Thead(html.Tr([html.Th("Model"), html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score"), html.Th("Accuracy (%)")]))] +
        [html.Tbody([
            html.Tr([
                html.Td(model_name),
                html.Td(f"{model_metrics.get('Precision', 0):.4f}"),  # Corrected format specifier
                html.Td(f"{model_metrics.get('Recall', 0):.4f}"),
                html.Td(f"{model_metrics.get('F1-score', 0):.4f}"),  # Corrected format specifier
                html.Td(f"{model_metrics.get('Accuracy (%)', 0):.2f}%") 
            ]) for model_name, model_metrics in metrics.items()
        ])],
        className="model-comparison-table"
    )
    return comparison_table

if __name__ == "__main__":
    app.run(debug = True)