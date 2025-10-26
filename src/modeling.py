# modeling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import optuna

def prepare_sequences(data, look_back=5, num_classes=52):
    """
    Prepare input-output sequences for LSTM training.

    Parameters:
        data (list): Flattened list of numbers.
        look_back (int): Number of previous steps to consider.
        num_classes (int): Total number of classes (lottery numbers).

    Returns:
        tuple: X (inputs), y_cat (categorical outputs)
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        if all(1 <= val <= num_classes for val in data[i:i+look_back]) and 1 <= data[i+look_back] <= num_classes:
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
    X = np.array(X).reshape((-1, look_back, 1))
    y_adjusted = np.array(y) - 1
    y_cat = to_categorical(y_adjusted, num_classes=num_classes)
    return X, y_cat

def build_lstm_model(units, look_back, num_classes, lr):
    """
    Build and compile an LSTM model.

    Parameters:
        units (int): Number of LSTM units.
        look_back (int): Input sequence length.
        num_classes (int): Output classes.
        lr (float): Learning rate.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 1)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss=CategoricalCrossentropy())
    return model

def optimize_lstm(X, y_cat, look_back, num_classes, n_trials=20):
    """
    Use Optuna to find optimal LSTM hyperparameters.

    Parameters:
        X (np.array): Input sequences.
        y_cat (np.array): Categorical outputs.
        look_back (int): Sequence length.
        num_classes (int): Number of output classes.
        n_trials (int): Number of optimization trials.

    Returns:
        dict: Best parameters found.
    """
    def objective(trial):
        units = trial.suggest_int("units", 32, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model = build_lstm_model(units, look_back, num_classes, lr)
        model.fit(X, y_cat, epochs=10, batch_size=32, verbose=0)
        pred = model.predict(X)
        return log_loss(y_cat, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def cross_validate_model(X, y_cat, best_params, look_back, num_classes, n_splits=5):
    """
    Perform K-Fold cross-validation on the LSTM model.

    Parameters:
        X (np.array): Input sequences.
        y_cat (np.array): Categorical outputs.
        best_params (dict): Optimized hyperparameters.
        look_back (int): Sequence length.
        num_classes (int): Output classes.
        n_splits (int): Number of folds.

    Returns:
        pd.DataFrame: DataFrame with evaluation metrics.
    """
    kf = KFold(n_splits=n_splits)
    metrics = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_cat[train_idx], y_cat[val_idx]

        model = build_lstm_model(best_params["units"], look_back, num_classes, best_params["lr"])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        pred = model.predict(X_val)

        mse = mean_squared_error(y_val, pred)
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        ll = log_loss(y_val, pred)

        metrics.append((mse, mae, r2, ll))

    metrics_df = pd.DataFrame(metrics, columns=["MSE", "MAE", "R2", "LogLoss"])
    return metrics_df

def predict_combination(model, X_last, combo_size, history_set):
    """
    Predict a full lottery combination from model output.

    Parameters:
        model (Sequential): Trained LSTM model.
        X_last (np.array): Last input sequence.
        combo_size (int): Number of numbers per combination.
        history_set (set): Set of historical combinations.

    Returns:
        tuple: Predicted combination (sorted).
    """
    pred = model.predict(X_last.reshape(1, X_last.shape[0], 1))[0]
    top_indices = np.argsort(pred)[-combo_size:]
    combo_pred = tuple(sorted(top_indices + 1))

    if combo_pred in history_set:
        for i in range(len(pred) - combo_size):
            alt_combo = tuple(sorted(np.argsort(pred)[i:i+combo_size] + 1))
            if alt_combo not in history_set:
                combo_pred = alt_combo
                break

    return combo_pred
