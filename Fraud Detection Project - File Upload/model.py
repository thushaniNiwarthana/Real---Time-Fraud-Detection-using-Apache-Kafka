import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import joblib
import os
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from features import FEATURE_ORDER, CATEGORICAL_FEATURES
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, matthews_corrcoef, log_loss
import json

def save_plot_as_base64(fig):
     #Save a matplotlib figure as a base64-encoded string.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    base64_string = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_string

def preprocess_data(df):
    #Preprocess the data consistently for training and prediction.

    # Data Cleaning
    df.drop(columns = ['Customer_ID', 'Merchant_ID', 'Transaction_ID', 
                     'Customer_Contact', 'Transaction_Description', 
                     'Customer_Email', 'Transaction_Device'], 
            inplace = True, errors = 'ignore')  # Use errors='ignore' to avoid KeyError
    
    # Combine Date and Time if both columns exist
    if 'Transaction_Date' in df.columns and 'Transaction_Time' in df.columns:
        df['Transaction_DateTime'] = pd.to_datetime(df['Transaction_Date'] + ' ' + df['Transaction_Time'])
        df.drop(columns = ['Transaction_Date', 'Transaction_Time'], inplace = True, errors = 'ignore')
    else:
        print("Warning: 'Transaction_Date' or 'Transaction_Time' column is missing. Skipping date-time combination.")

    # Handle missing values and duplicates
    df.drop_duplicates(inplace = True)
    df.fillna(0, inplace = True)
    
    # Extract datetime features if 'Transaction_DateTime' exists
    if 'Transaction_DateTime' in df.columns:
        df['year'] = df['Transaction_DateTime'].dt.year
        df['month'] = df['Transaction_DateTime'].dt.month
        df['day'] = df['Transaction_DateTime'].dt.day
        df['hour'] = df['Transaction_DateTime'].dt.hour
        df['minute'] = df['Transaction_DateTime'].dt.minute
        df['second'] = df['Transaction_DateTime'].dt.second
        df.drop(columns = ['Transaction_DateTime'], inplace = True, errors = 'ignore')
    else:
        print("Warning: 'Transaction_DateTime' column is missing. Skipping datetime feature extraction.")
    
    # Encode Categorical Variables
    label_encoder = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])
    
    # Ensure all features are present and in correct order
    for feature in FEATURE_ORDER:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value
    
    return df

def train_and_evaluate_model():
    # Define the folder path for saving PNG files
    output_folder = "Model Reports"
    os.makedirs(output_folder, exist_ok = True)  # Create the folder if it doesn't exist

    # Start the timer
    start_time = time.time()

    # Load and preprocess data
    file_path = 'transactions.csv'
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    
    # Verify target column exists (use either 'Is_Fraud' or 'is_fraud')
    target_column = 'is_fraud' if 'is_fraud' in df.columns else 'Is_Fraud'
    if target_column not in df.columns:
        raise ValueError("Could not find target column ('is_fraud' or 'Is_Fraud') in dataset")
    
    # Split data
    X = df[FEATURE_ORDER]  # Use only the features we need in correct order
    Y = df[target_column]
    
    # Check for zero samples in the target column
    if len(Y[Y == 1]) == 0 or len(Y[Y == 0]) == 0:
        raise ValueError("The target column has zero samples for one of the classes. Please check the dataset.")

    # Balance the dataset using SMOTE if needed
    smote = SMOTE(random_state = 42)
    X_balanced, Y_balanced = smote.fit_resample(X, Y)
    
    # Split balanced data
    x_train, x_test, y_train, y_test = train_test_split(X_balanced, Y_balanced, test_size = 0.2, random_state = 42)
    
    # Feature Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, os.path.join(output_folder, 'scaler.pkl'))
    
    # Train models
    models = {
        "Logistic Regression": LogisticRegression(class_weight = 'balanced', random_state = 42),
        "Random Forest": RandomForestClassifier(random_state = 42, class_weight = 'balanced'),
        "XGBoost": XGBClassifier(scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])), 
    }
    
    metrics = {}
    
    # Add a progress bar for training models
    for name, model in tqdm(models.items(), desc = "Training Models", unit = "model"):
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]  # For probability-based metrics
        
        # Save metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()  # Extract confusion matrix components
        
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,  # Convert accuracy to percentage
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc_score(y_test, y_proba),
            'specificity': tn / (tn + fp),
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
            'log_loss': log_loss(y_test, y_proba)
        }
        
        print(f"Accuracy for {name}: {metrics[name]['accuracy']:.2f}%")
        
        # Save classification report as a text file
        report_filename = os.path.join(output_folder, f"classification_report_{name.replace(' ', '_').lower()}.txt")
        with open(report_filename, "w") as f:
            f.write(classification_report(y_test, y_pred))
        print(f"Classification report saved as {report_filename}")
        
        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        chart_filename = os.path.join(output_folder, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
        fig.savefig(chart_filename)
        print(f'Confusion matrix saved as {chart_filename}')
        plt.close(fig)
        
        # Save metrics bar chart
        fig, ax = plt.subplots(figsize = (7, 5))
        metric_names = list(metrics[name].keys())
        metric_values = list(metrics[name].values())
        sns.barplot(x=metric_names, y=metric_values, ax = ax)
        ax.set_title(f'Metrics Report - {name}')
        ax.set_ylim(0, 1)  # Ensure the y-axis is between 0 and 1
        ax.set_ylabel('Score')
        ax.set_xlabel('Metrics')
        report_chart_filename = os.path.join(output_folder, f"metrics_report_{name.replace(' ', '_').lower()}.png")
        fig.savefig(report_chart_filename)
        print(f"Metrics report saved as {report_chart_filename}")
        plt.close(fig)
        
        # Save feature importance (if applicable)
        if hasattr(model, "feature_importances_"):
            feature_importances = pd.Series(model.feature_importances_, index = FEATURE_ORDER)
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.nlargest(10).plot(kind='barh', ax=ax)
            ax.set_title(f"Feature Importance - {name}")
            feature_importance_filename = os.path.join(output_folder, f"feature_importance_{name.replace(' ', '_').lower()}.png")
            fig.savefig(feature_importance_filename)
            print(f"Feature importance saved as {feature_importance_filename}")
            plt.close(fig)
    
    # Save the best model (XGBoost in this case)
    joblib.dump(models["XGBoost"], os.path.join(output_folder, 'fraud_detection_model.pkl'))
    print(f"Best model saved as {os.path.join(output_folder, 'fraud_detection_model.pkl')}")

    # Save metrics to a JSON file
    metrics_file = os.path.join(output_folder, "model_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # End the timer
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # Convert seconds to minutes
    print(f"Training completed in {elapsed_time:.2f} minutes.")
    
    return metrics

if __name__ == "__main__":
    metrics = train_and_evaluate_model()
    print("Training completed. \nAll reports and charts saved.")