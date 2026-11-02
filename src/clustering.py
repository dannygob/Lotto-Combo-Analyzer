# clustering.py
# KMeans clustering utilities for lottery combination analysis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def apply_kmeans(df, combo_cols, n_clusters=5):
    """
    Apply KMeans clustering to lottery combinations.

    Parameters:
        df (pd.DataFrame): DataFrame containing lottery numbers.
        combo_cols (list): List of column names containing the numbers.
        n_clusters (int): Number of clusters to form.

    Returns:
        pd.Series: Cluster labels assigned to each row.
    """
    # Replace missing values with column means to avoid NaN issues during clustering.
    X = df[combo_cols].fillna(df[combo_cols].mean()).values

    # Initialize and fit KMeans.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Predict cluster labels.
    clusters = kmeans.fit_predict(X)

    return pd.Series(clusters, name="Cluster")


def plot_elbow_method(df, combo_cols, max_k=10):
    """
    Plot the elbow method to help determine the optimal number of clusters.

    Parameters:
        df (pd.DataFrame): DataFrame containing lottery numbers.
        combo_cols (list): List of number columns.
        max_k (int): Maximum number of clusters to evaluate.

    Notes:
        The elbow point indicates a good trade-off between inertia reduction and model complexity.
    """
    X = df[combo_cols].fillna(df[combo_cols].mean()).values
    inertia = []

    # Compute inertia for each k.
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        inertia.append(kmeans.inertia_)

    # Plot inertia curve.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia, marker="o")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cluster_distribution(df):
    """
    Plot the distribution of combinations across clusters.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Cluster' column.

    Notes:
        Useful for understanding how many combinations fall into each cluster.
    """
    import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.countplot(x="Cluster", hue="Cluster", data=df, palette="tab10", legend=False)
    plt.title("Distribution of Combinations by Cluster")
    plt.tight_layout()
    plt.show()
