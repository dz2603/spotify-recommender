import pandas as pd
import streamlit as st

from src.evaluation.recommender_metrics import evaluate_recommendations
from src.ui.common import (
    get_recommendation_results,
    load_data,
    load_lookup_tables,
    resolve_song_query,
)
from src.ui.components import divider, page_header, render_metric_cards, render_reference_track


def page_recommendation_evaluation():
    page_header(
        "🧪 Recommendation Evaluation",
        """
        This page evaluates whether the KNN recommendations are musically reasonable.
        Instead of only showing recommended songs, we report simple metrics such as
        same-genre rate, same-artist rate, average popularity, popularity gap,
        and diversity among recommended tracks.
        """,
    )

    _, neighbor_lookup, tables_ok = load_lookup_tables()

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input(
            "Search for a reference song to evaluate",
            placeholder="e.g. Gangnam Style, Shape of You, Blinding Lights",
        )
    with col2:
        k = st.slider("Number of recommendations (K)", 5, 20, 10)

    metric = st.radio(
        "Similarity metric",
        ["cosine", "euclidean"],
        horizontal=True,
        help="Cosine: angle-based similarity. Euclidean: straight-line distance.",
    )

    divider()

    if not query:
        st.info("Type a song name above to evaluate its recommendations.")
        return

    data, X_num, feat_cols, X_weighted = load_data(sample_n=None)

    query_idx = resolve_song_query(data, query)
    if query_idx is None:
        st.warning(f"No songs found matching **{query}**.")
        return

    ref_row = data.iloc[query_idx]
    render_reference_track(ref_row)

    results = get_recommendation_results(
        data=data,
        X_num=X_num,
        X_weighted=X_weighted,
        query_idx=query_idx,
        k=k,
        metric=metric,
        neighbor_lookup=neighbor_lookup,
        tables_ok=tables_ok,
    )

    if not results:
        st.warning("No recommendation results were generated.")
        return

    rec_indices = [r["index"] for r in results]
    recommended_rows = data.iloc[rec_indices].copy()
    recommended_rows_for_eval = recommended_rows.copy()
    for j, col in enumerate(feat_cols):
        if col in recommended_rows_for_eval.columns:
            recommended_rows_for_eval[col] = X_num[rec_indices, j]

    metrics = evaluate_recommendations(
        query_row=ref_row,
        recommended_rows=recommended_rows_for_eval,
        feature_cols=feat_cols,
    )

    st.subheader("📈 Evaluation Metrics")
    same_genre = metrics["same_genre_rate"]
    same_artist = metrics["same_artist_rate"]
    avg_pop = metrics["average_popularity"]
    pop_gap = metrics["popularity_gap"]
    diversity = metrics["diversity_score"]

    render_metric_cards(
        [
            ("Same Genre Rate", "N/A" if same_genre is None else f"{same_genre:.0%}"),
            ("Same Artist Rate", "N/A" if same_artist is None else f"{same_artist:.0%}"),
            ("Avg Popularity", "N/A" if avg_pop is None else f"{avg_pop:.2f}"),
            ("Popularity Gap", "N/A" if pop_gap is None else f"{pop_gap:.2f}"),
            ("Diversity Score", "N/A" if diversity is None else f"{diversity:.4f}"),
        ]
    )

    divider()
    st.subheader("🎵 Recommended Songs Used for Evaluation")
    score_col = "Similarity Score" if metric == "cosine" else "Distance"

    rows = []
    for rank, r in enumerate(results, 1):
        row = data.iloc[r["index"]]
        rows.append(
            {
                "Rank": rank,
                "Track": row["track_name"],
                "Artist(s)": row["artists"],
                "Genre": row.get("track_genre", "N/A"),
                "Popularity": row["popularity"],
                score_col: f"{r['score']:.4f}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("ℹ️ How to interpret these metrics"):
        st.markdown(
            """
            - **Same Genre Rate**: Higher means the recommender tends to stay within the same genre.
            - **Same Artist Rate**: Higher means recommendations often come from the same artist.
            - **Average Popularity**: Shows whether recommendations are mostly mainstream or niche.
            - **Popularity Gap**: Lower means the recommendations have similar popularity to the reference song.
            - **Diversity Score**: Higher means the recommended songs are more diverse in standardized audio-feature space.
            """
        )
