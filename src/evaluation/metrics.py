"""
Evaluation metrics for clustering quality.
"""
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score: measures how similar each point is to its own cluster
    vs other clusters. Range [-1, 1]; higher is better.
    Returns NaN if only 1 cluster or all points in same cluster.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    return float(silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42))


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies-Bouldin score: average ratio of within-cluster scatter to
    between-cluster separation. Lower is better. Returns NaN if < 2 clusters.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    return float(davies_bouldin_score(X, labels))


def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    total = 0.0
    for c in range(len(centroids)):
        mask = labels == c
        if mask.sum() > 0:
            diff = X[mask] - centroids[c]
            total += float(np.einsum("ij,ij->", diff, diff))
    return total


def cluster_summary(X: np.ndarray, labels: np.ndarray) -> dict:
    """Return per-cluster size counts."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(c): int(n) for c, n in zip(unique, counts)}
