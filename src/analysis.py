# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def calculate_frequency(df, combo_cols):
    """
    Calculate frequency of each number across all combinations.

    Parameters:
        df (pd.DataFrame): DataFrame containing lottery numbers.
        combo_cols (list): List of column names with numbers.

    Returns:
        pd.DataFrame: Frequency table sorted by most common numbers.
    """
    numbers = df[combo_cols].values.flatten()
    freq = Counter(numbers)
    freq_df = pd.DataFrame(freq.items(), columns=["Number", "Frequency"]).sort_values(by="Frequency", ascending=False)
    return freq_df

def plot_top_frequencies(freq_df, top_n=15):
    """
    Plot the top N most frequent numbers.

    Parameters:
        freq_df (pd.DataFrame): Frequency table.
        top_n (int): Number of top values to display.
    """
    plt.figure(figsize=(10,6))
    sns.barplot(x="Number", y="Frequency", hue="Number", data=freq_df.head(top_n), palette="viridis", legend=False)
    plt.title(f"Top {top_n} Most Frequent Numbers")
    plt.tight_layout()
    plt.show()

def add_sum_column(df, combo_cols):
    """
    Add a column with the sum of each combination.

    Parameters:
        df (pd.DataFrame): DataFrame with lottery numbers.
        combo_cols (list): List of number columns.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Suma' column.
    """
    df['Suma'] = df[combo_cols].sum(axis=1)
    return df

def plot_sum_distribution(df):
    """
    Plot histogram and boxplot of total sums.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Suma' column.
    """
    plt.figure(figsize=(10,6))
    sns.histplot(df['Suma'], bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Total Sum per Draw")
    plt.xlabel("Total Sum")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    sns.boxplot(x=df['Suma'], color='lightgreen')
    plt.title("Boxplot of Total Sum per Draw")
    plt.tight_layout()
    plt.show()

def describe_sum_statistics(df):
    """
    Print descriptive statistics of the sum column.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Suma' column.
    """
    print("📊 Descriptive Statistics of Total Sum:")
    print(df['Suma'].describe())

def add_differential_column(df):
    """
    Add a column with the difference between consecutive draw sums.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Suma' column.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Diferencial' column.
    """
    df['Diferencial'] = df['Suma'].diff()
    return df

def plot_differential_distribution(df):
    """
    Plot histogram, boxplot, and time series of differentials.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Diferencial' column.
    """
    plt.figure(figsize=(10,6))
    sns.histplot(df['Diferencial'].dropna(), bins=30, kde=True, color='salmon')
    plt.title("Distribution of Differentials Between Draws")
    plt.xlabel("Differential")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    sns.boxplot(x=df['Diferencial'].dropna(), color='lightblue')
    plt.title("Boxplot of Draw Differentials")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(df.index, df['Diferencial'], marker='o', linestyle='-', color='gray')
    plt.title("Time Series of Draw Differentials")
    plt.xlabel("Draw Index")
    plt.ylabel("Differential")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def classify_changes(df):
    """
    Classify each differential as 'Up', 'Down', or 'Same'.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Diferencial' column.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Cambio' column.
    """
    df['Cambio'] = df['Diferencial'].apply(lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Same'))
    return df

def plot_change_classification(df):
    """
    Plot frequency of change classifications.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Cambio' column.
    """
    cambio_counts = df['Cambio'].value_counts().reset_index()
    cambio_counts.columns = ['Cambio', 'Cantidad']
    plt.figure(figsize=(6,4))
    sns.barplot(x="Cambio", y="Cantidad", hue="Cambio", data=cambio_counts, palette='Set2', legend=False)
    plt.title("Frequency of Changes Between Draws")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
