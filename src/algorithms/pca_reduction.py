"""
PCA dimensionality reduction using sklearn.
"""
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X: np.ndarray, n_components: int = 2, feature_names: list[str] = None) -> dict:
    """
    Fit PCA and return 2D (or n_components-D) embeddings.

    Returns:
        {
          "X_reduced":          (n, n_components) array,
          "explained_variance_ratio": (n_components,) array,
          "cumulative_variance": float — total variance explained,
          "n_components":        int,
          "pca":                 fitted PCA object (for transform of new data),
          "axis_labels":         list of str — e.g. ["+energy -acousticness", ...],
        }
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    axis_labels = [f"PC{i+1}" for i in range(n_components)]

    return {
        "X_reduced": X_reduced,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "n_components": n_components,
        "pca": pca,
        "axis_labels": axis_labels,
    }


def pca_variance_sweep(X: np.ndarray, max_components: int = 20, feature_names: list[str] = None) -> dict:
    """Compute explained variance ratio for 1..max_components to help choose n_components."""
    n_max = min(max_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_max, random_state=42)
    pca.fit(X)

    axis_labels = [f"PC{i+1}" for i in range(n_max)]

    return {
        "ratios": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
        "axis_labels": axis_labels,
    }
