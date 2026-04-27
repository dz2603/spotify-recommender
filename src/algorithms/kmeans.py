"""
K-Means clustering from scratch — pure numpy, no sklearn.
"""
import numpy as np


def _init_centroids(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """K-means++ initialisation for better convergence."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    # Pick first centroid randomly
    idx = rng.integers(0, n)
    centroids = [X[idx]]

    for _ in range(k - 1):
        # Squared distance from each point to its nearest centroid
        dists = np.array([
            min(np.sum((x - c) ** 2) for c in centroids) for x in X
        ])
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids.append(X[idx])

    return np.array(centroids)


def _assign_labels(X, centroids, batch_size=500):
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        X_batch = X[start:end]

        # distances shape: (batch_size, k)
        distances = np.empty((end - start, centroids.shape[0]), dtype=np.float32)

        for c in range(centroids.shape[0]):
            diff = X_batch - centroids[c]
            distances[:, c] = np.sum(diff * diff, axis=1)

        labels[start:end] = np.argmin(distances, axis=1)

    return labels                  # (n,)


def _update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute centroids as the mean of their assigned points."""
    d = X.shape[1]
    centroids = np.zeros((k, d))
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = X[mask].mean(axis=0)
    return centroids


def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of squared distances from each point to its centroid."""
    total = 0.0
    for c in range(len(centroids)):
        mask = labels == c
        if mask.sum() > 0:
            diff = X[mask] - centroids[c]
            total += float(np.einsum("ij,ij->", diff, diff))
    return total


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> dict:
    """
    Run K-Means from scratch.

    Returns:
        {
          "labels":    (n,) int array of cluster assignments,
          "centroids": (k, d) float array of final centroids,
          "inertia":   float — final within-cluster sum of squares,
          "inertia_history": list of inertia per iteration,
          "n_iter":    int — iterations until convergence,
        }
    """
    centroids = _init_centroids(X, k, random_state)
    inertia_history = []

    for i in range(max_iter):
        labels = _assign_labels(X, centroids)
        new_centroids = _update_centroids(X, labels, k)
        inertia = compute_inertia(X, labels, new_centroids)
        inertia_history.append(inertia)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return {
        "labels": labels,
        "centroids": centroids,
        "inertia": inertia,
        "inertia_history": inertia_history,
        "n_iter": i + 1,
    }


def elbow_analysis(X: np.ndarray, k_range: range, random_state: int = 42) -> dict:
    """Run K-Means for each K value and collect inertia (elbow method)."""
    results = {}
    for k in k_range:
        res = kmeans(X, k=k, random_state=random_state)
        results[k] = res["inertia"]
    return results
