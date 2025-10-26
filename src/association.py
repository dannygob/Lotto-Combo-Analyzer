# association.py

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def prepare_transactions(df, combo_cols):
    """
    Convert lottery combinations into transaction format for Apriori.

    Parameters:
        df (pd.DataFrame): DataFrame with lottery numbers.
        combo_cols (list): List of number columns.

    Returns:
        list: List of transactions (each combination as a list of strings).
    """
    transactions = df[combo_cols].astype(str).values.tolist()
    return transactions

def encode_transactions(transactions):
    """
    Encode transactions into a boolean DataFrame.

    Parameters:
        transactions (list): List of transactions.

    Returns:
        pd.DataFrame: Encoded DataFrame for Apriori.
    """
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df_encoded

def apply_apriori(df_encoded, min_support=0.01):
    """
    Apply Apriori algorithm to find frequent itemsets.

    Parameters:
        df_encoded (pd.DataFrame): Boolean DataFrame of transactions.
        min_support (float): Minimum support threshold.

    Returns:
        pd.DataFrame: Frequent itemsets.
    """
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence=0.3, min_lift=1.0):
    """
    Generate association rules from frequent itemsets.

    Parameters:
        frequent_itemsets (pd.DataFrame): Output from Apriori.
        min_confidence (float): Minimum confidence threshold.
        min_lift (float): Minimum lift threshold.

    Returns:
        pd.DataFrame: Association rules.
    """
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
    return rules

def plot_lift_distribution(rules):
    """
    Plot the distribution of lift values from association rules.

    Parameters:
        rules (pd.DataFrame): Association rules.
    """
    plt.figure(figsize=(8,5))
    plt.hist(rules['lift'], bins=30, color='purple', edgecolor='black')
    plt.title("Distribution of Lift Values in Association Rules")
    plt.xlabel("Lift")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
