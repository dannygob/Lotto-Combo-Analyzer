# comparison.py
# Classification model comparison utilities (Random Forest vs XGBoost).

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, classification_report


def prepare_classification_data(data, look_back=5, num_classes=52):
    """
    Prepare sequential data for classification models.

    Each sample consists of:
        - X: the previous `look_back` numbers
        - y: the next number in the sequence

    Parameters:
        data (list): Flattened list of numbers.
        look_back (int): Number of previous steps to use as features.
        num_classes (int): Maximum valid class label.

    Returns:
        tuple: (X, y) numpy arrays.
    """
    X, y = [], []

    for i in range(len(data) - look_back):
        window = data[i:i + look_back]
        next_val = data[i + look_back]

        # Ensure all values fall within the valid class range.
        if all(1 <= val <= num_classes for val in window) and 1 <= next_val <= num_classes:
            X.append(window)
            y.append(next_val)

    return np.array(X), np.array(y)


def train_random_forest(X, y):
    """
    Train and evaluate a Random Forest classifier.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.

    Returns:
        dict: Evaluation metrics and trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "model": rf,
    }

    return metrics


def train_xgboost(X, y):
    """
    Train and evaluate an XGBoost classifier.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.

    Returns:
        dict: Evaluation metrics and trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "model": xgb,
    }

    return metrics


def compare_models(rf_metrics, xgb_metrics):
    """
    Print a simple comparison between Random Forest and XGBoost.

    Parameters:
        rf_metrics (dict): Metrics from Random Forest.
        xgb_metrics (dict): Metrics from XGBoost.
    """
    print("🔍 Model Comparison:")
    print(f"Random Forest Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Random Forest Log Loss: {rf_metrics['log_loss']:.4f}")
    print(f"XGBoost Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"XGBoost Log Loss: {xgb_metrics['log_loss']:.4f}")
