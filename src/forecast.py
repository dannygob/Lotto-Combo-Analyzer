# forecast.py

import numpy as np

def forecast_combination_by_index(model, df, combo_cols, index, combo_size=6, history_set=None):
    """
    Generate a personalized forecast based on a specific draw index.

    Parameters:
        model (Sequential): Trained prediction model.
        df (pd.DataFrame): DataFrame with lottery history.
        combo_cols (list): List of number columns.
        index (int): Index of the draw to use as seed.
        combo_size (int): Size of the predicted combination.
        history_set (set): Set of historical combinations.

    Returns:
        tuple: Predicted combination (sorted).
    """
    if index < 5 or index >= len(df):
        raise ValueError("Index must be between 5 and len(df)-1")

    seed_sequence = df[combo_cols].iloc[index - 5:index].values.flatten().tolist()
    X_seed = np.array(seed_sequence).reshape((1, len(seed_sequence), 1))

    pred = model.predict(X_seed)[0]
    sorted_indices = np.argsort(pred)[::-1]

    for i in range(len(sorted_indices) - combo_size + 1):
        combo = tuple(sorted(sorted_indices[i:i+combo_size] + 1))
        if history_set is None or combo not in history_set:
            return combo

    return tuple(sorted(sorted_indices[:combo_size] + 1))

def interactive_forecast(model, df, combo_cols, history_set):
    """
    Prompt user for draw index and generate forecast.

    Parameters:
        model (Sequential): Trained model.
        df (pd.DataFrame): Lottery history.
        combo_cols (list): Number columns.
        history_set (set): Set of historical combinations.
    """
    try:
        index = int(input("🔢 Enter draw index to forecast from (e.g. 100): "))
        combo = forecast_combination_by_index(model, df, combo_cols, index, combo_size=6, history_set=history_set)
        print(f"🎯 Forecasted combination for draw {index}: {combo}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
