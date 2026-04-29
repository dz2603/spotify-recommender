import random

import pandas as pd
import streamlit as st

from src.evaluation.recommender_metrics import evaluate_recommendations
from src.ui.common import get_recommendation_results
from src.ui.components import divider


def render_metric_comparison(data, X_num, X_weighted, query_idx, k, neighbor_lookup, tables_ok, ref_row, feat_cols):
    """Render cosine vs euclidean vs baseline metrics comparison table."""

    results_cosine = get_recommendation_results(
        data=data, X_num=X_num, X_weighted=X_weighted,
        query_idx=query_idx, k=k, metric="cosine",
        neighbor_lookup=neighbor_lookup, tables_ok=tables_ok,
    )
    results_euclidean = get_recommendation_results(
        data=data, X_num=X_num, X_weighted=X_weighted,
        query_idx=query_idx, k=k, metric="euclidean",
        neighbor_lookup=neighbor_lookup, tables_ok=tables_ok,
    )

    if not results_cosine or not results_euclidean:
        st.warning("Could not compute recommendations for one or both metrics.")
        return

    # --- Shared helper: mirrors exactly how recommended_rows_for_eval is built ---
    def _build_eval_rows(results):
        rec_indices = [r["index"] for r in results]
        rows = data.iloc[rec_indices].copy()
        for j, col in enumerate(feat_cols):
            if col in rows.columns:
                rows[col] = X_num[rec_indices, j]
        return rows

    metrics_cosine = evaluate_recommendations(
        query_row=ref_row,
        recommended_rows=_build_eval_rows(results_cosine),
        feature_cols=feat_cols,
    )
    metrics_euclidean = evaluate_recommendations(
        query_row=ref_row,
        recommended_rows=_build_eval_rows(results_euclidean),
        feature_cols=feat_cols,
    )

    # --- Baseline: random K positional indices, excluding query_idx ---
    all_pos = [i for i in range(len(data)) if i != query_idx]
    rng = random.Random(42)
    random_pos = rng.sample(all_pos, k)

    # Build eval rows identically to cosine/euclidean
    baseline_rows = data.iloc[random_pos].copy()
    for j, col in enumerate(feat_cols):
        if col in baseline_rows.columns:
            baseline_rows[col] = X_num[random_pos, j]

    metrics_baseline = evaluate_recommendations(
        query_row=ref_row,
        recommended_rows=baseline_rows,
        feature_cols=feat_cols,
    )

    # --- Comparison table ---
    st.subheader("📊 Cosine vs Euclidean vs Baseline — Metrics Comparison")

    metric_defs = [
        ("Same Genre Rate",  "same_genre_rate",    "rate"),
        ("Same Artist Rate", "same_artist_rate",   "rate"),
        ("Avg Popularity",   "average_popularity", "float"),
        ("Popularity Gap",   "popularity_gap",     "float"),
        ("Diversity Score",  "diversity_score",    "precise"),
    ]

    def _fmt(val, kind):
        if val is None:
            return "N/A"
        if kind == "rate":
            return f"{val:.0%}"
        if kind == "precise":
            return f"{val:.4f}"
        return f"{val:.2f}"

    comparison_rows = []
    for label, key, kind in metric_defs:
        comparison_rows.append({
            "Metric":    label,
            "Cosine":    _fmt(metrics_cosine[key],    kind),
            "Euclidean": _fmt(metrics_euclidean[key], kind),
            "Baseline":  _fmt(metrics_baseline[key],  kind),
        })

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)