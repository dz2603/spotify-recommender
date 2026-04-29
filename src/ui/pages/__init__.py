
import pandas as pd
import streamlit as st

from src.evaluation.recommender_metrics import evaluate_recommendations
from src.ui.common import get_recommendation_results
from src.ui.components import divider


def render_metric_comparison(data, X_num, X_weighted, query_idx, k, neighbor_lookup, tables_ok, ref_row, feat_cols):
    """Render a side-by-side cosine vs euclidean comparison table."""

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

    # --- Evaluate both ---
    def _get_eval_rows(results):
        rec_indices = [r["index"] for r in results]
        rows = data.iloc[rec_indices].copy()
        for j, col in enumerate(feat_cols):
            if col in rows.columns:
                rows[col] = X_num[rec_indices, j]
        return rows

    metrics_cosine = evaluate_recommendations(
        query_row=ref_row,
        recommended_rows=_get_eval_rows(results_cosine),
        feature_cols=feat_cols,
    )
    metrics_euclidean = evaluate_recommendations(
        query_row=ref_row,
        recommended_rows=_get_eval_rows(results_euclidean),
        feature_cols=feat_cols,
    )

    # --- Metrics comparison table ---
    st.subheader("📊 Cosine vs Euclidean — Metrics Comparison")

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
        cos_val = metrics_cosine[key]
        euc_val = metrics_euclidean[key]
        comparison_rows.append({
            "Metric":     label,
            "Cosine":     _fmt(cos_val, kind),
            "Euclidean":  _fmt(euc_val, kind),
        })

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    divider()

    # --- Side-by-side track tables ---
    st.subheader("🎵 Recommended Tracks: Side-by-Side")
    col_cos, col_euc = st.columns(2)

    def _build_track_rows(results, score_label):
        rows = []
        for rank, r in enumerate(results, 1):
            row = data.iloc[r["index"]]
            rows.append({
                "Rank":       rank,
                "Track":      row["track_name"],
                "Artist(s)":  row["artists"],
                "Genre":      row.get("track_genre", "N/A"),
                "Popularity": row["popularity"],
                score_label:  f"{r['score']:.4f}",
            })
        return pd.DataFrame(rows)

    with col_cos:
        st.markdown("#### 🔵 Cosine Similarity")
        st.dataframe(_build_track_rows(results_cosine, "Score"), use_container_width=True, hide_index=True)

    with col_euc:
        st.markdown("#### 🟠 Euclidean Distance")
        st.dataframe(_build_track_rows(results_euclidean, "Distance"), use_container_width=True, hide_index=True)

    with st.expander("💡 How to interpret this comparison"):
        st.markdown("""
        - **Cosine** measures the *angle* between feature vectors — ignores magnitude, focuses on shape.
        - **Euclidean** measures *straight-line distance* — sensitive to absolute differences in each feature.
        - A high **Same Genre Rate** in one metric means it respects genre boundaries better.
        - A lower **Popularity Gap** means recommendations better match the reference track's popularity.
        - A higher **Diversity Score** means the top-K list covers more variety in audio-feature space.
        """)