# preprocessing.py

import pandas as pd

def load_dataset(path):
    """
    Load the lottery dataset from a CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(path)
    return df

def extract_combo_columns(df):
    """
    Extract columns that represent lottery numbers.

    Parameters:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        list: List of column names that start with 'numero '.
    """
    combo_cols = [col for col in df.columns if col.startswith('numero ')]
    return combo_cols

def clean_numeric_columns(df, combo_cols):
    """
    Convert combo columns to numeric and drop rows with missing values.

    Parameters:
        df (pd.DataFrame): DataFrame with raw data.
        combo_cols (list): List of combo column names.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.dropna(subset=combo_cols)
    for col in combo_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def sort_combinations(df, combo_cols):
    """
    Sort each combination of numbers in ascending order.

    Parameters:
        df (pd.DataFrame): DataFrame with combo columns.
        combo_cols (list): List of combo column names.

    Returns:
        pd.Series: Series of sorted tuples.
    """
    df_sorted = df[combo_cols].apply(lambda row: tuple(sorted(row)), axis=1)
    return df_sorted

def prepare_history_set(df_sorted):
    """
    Create a set of historical combinations for uniqueness checks.

    Parameters:
        df_sorted (pd.Series): Sorted combinations.

    Returns:
        set: Set of unique combinations.
    """
    return set(df_sorted)
