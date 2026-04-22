"""
Gaussian Mixture Model (GMM) — soft clustering using sklearn.
Provides a clean interface that mirrors kmeans.py for easy side-by-side comparison.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gmm(
    X: np.ndarray,
    k: int,
    covariance_type: str = "full",
    random_state: int = 42,
    max_iter: int = 100,
) -> dict:
    """
    Fit a GMM with k components.

    Returns:
        {
          "labels":       (n,) int — hard assignment (argmax of responsibilities),
          "proba":        (n, k) float — soft membership probabilities,
          "bic":          float — BIC score (lower = better model fit),
          "aic":          float — AIC score,
          "log_likelihood": float,
          "n_iter":       int,
          "converged":    bool,
        }
    """
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
    )
    gmm.fit(X)

    proba = gmm.predict_proba(X)   # (n, k) — soft memberships
    labels = np.argmax(proba, axis=1)  # hard assignment for visualisation

    return {
        "labels": labels,
        "proba": proba,
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "log_likelihood": float(gmm.lower_bound_),
        "n_iter": gmm.n_iter_,
        "converged": gmm.converged_,
    }


def gmm_bic_sweep(
    X: np.ndarray,
    k_range: range,
    covariance_type: str = "full",
    random_state: int = 42,
) -> dict:
    """Run GMM for each K and collect BIC scores (model selection)."""
    bic_scores = {}
    for k in k_range:
        res = fit_gmm(X, k=k, covariance_type=covariance_type, random_state=random_state)
        bic_scores[k] = res["bic"]
    return bic_scores
