# clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def apply_kmeans(df, combo_cols, n_clusters=5):
    """
    Apply KMeans clustering to lottery combinations.

    Parameters:
        df (pd.DataFrame): DataFrame containing lottery numbers.
        combo_cols (list): List of column names with numbers.
        n_clusters (int): Number of clusters to form.

    Returns:
        pd.Series: Cluster labels for each row.
    """
    X = df[combo_cols].fillna(df[combo_cols].mean()).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return pd.Series(clusters, name="Cluster")

def plot_elbow_method(df, combo_cols, max_k=10):
    """
    Plot the elbow method to determine optimal number of clusters.

    Parameters:
        df (pd.DataFrame): DataFrame with lottery numbers.
        combo_cols (list): List of number columns.
        max_k (int): Maximum number of clusters to test.
    """
    X = df[combo_cols].fillna(df[combo_cols].mean()).values
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
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
        df (pd.DataFrame): DataFrame with 'Cluster' column.
    """
    import seaborn as sns
    plt.figure(figsize=(8,5))
    sns.countplot(x="Cluster", hue="Cluster", data=df, palette="tab10", legend=False)
    plt.title("Distribution of Combinations by Cluster")
    plt.tight_layout()
    plt.show()
