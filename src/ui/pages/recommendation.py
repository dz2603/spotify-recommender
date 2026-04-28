import pandas as pd
import streamlit as st

from src.algorithms.knn import knn_query
from src.ui.common import (
    explain_recommendation_features,
    load_data,
    load_lookup_tables,
    spotify_player,
)


def page_recommendation():
    st.header("🎵 Song Recommendation")
    st.markdown(
        "Find songs similar to a reference track using **KNN** with "
        "cosine or Euclidean distance on audio features."
    )

    _, neighbor_lookup, tables_ok = load_lookup_tables()

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search for a song", placeholder="e.g. Gangnam Style")
    with col2:
        k = st.slider("Number of recommendations (K)", 5, 20, 10)

    metric = st.radio(
        "Similarity metric",
        ["cosine", "euclidean"],
        horizontal=True,
        help="Cosine: angle-based (ignores magnitude). Euclidean: straight-line distance.",
    )

    st.markdown("---")

    if not query:
        st.info("Type a song name above to get recommendations.")
        return

    data, X_num, feat_cols, _ = load_data(sample_n=None)
    name_col = data["track_name"].str.lower()
    idxs = name_col[name_col.str.contains(query.lower(), na=False)].index.tolist()

    if not idxs:
        st.warning(f"No songs found matching **{query}**.")
        return

    if len(idxs) > 1:
        idxs_sorted = sorted(idxs[:20], key=lambda i: data.iloc[i]["popularity"], reverse=True)
        opts = [f"{data.iloc[i]['track_name']} — {data.iloc[i]['artists']}" for i in idxs_sorted]
        chosen = st.selectbox("Multiple matches — pick one:", opts)
        query_idx = idxs_sorted[opts.index(chosen)]
    else:
        query_idx = idxs[0]

    ref_row = data.iloc[query_idx]
    artists_str = (
        ", ".join(ref_row["artists"])
        if isinstance(ref_row["artists"], list)
        else str(ref_row["artists"])
    )
    st.success(f"🎧 Reference: **{ref_row['track_name']}** by *{artists_str}*")
    spotify_player(ref_row["track_id"], height=152)

    ref_track_id = ref_row["track_id"]
    if tables_ok and ref_track_id in neighbor_lookup and metric == "cosine":
        raw = neighbor_lookup[ref_track_id][:k]
        results = [
            {"index": data[data["track_id"] == n["track_id"]].index[0], "score": n["score"]}
            for n in raw
        ]
    else:
        with st.spinner("Computing neighbors..."):
            results = knn_query(X_num, query_idx, k=k, metric=metric)

    rows = []
    explanation_details = []
    score_col = "Similarity Score" if metric == "cosine" else "Distance"
    ref_vec = X_num[query_idx]
    for rank, r in enumerate(results, 1):
        row = data.iloc[r["index"]]
        cand_vec = X_num[r["index"]]
        why_summary, why_details = explain_recommendation_features(
            ref_vec, cand_vec, feat_cols, top_n=3
        )
        explanation_details.append(
            {
                "rank": rank,
                "track": row["track_name"],
                "artists": row["artists"],
                "details": why_details,
            }
        )
        rows.append(
            {
                "Rank": rank,
                "Track": row["track_name"],
                "Artist(s)": row["artists"],
                score_col: f"{r['score']:.4f}",
                "Why recommended": f"Similar {why_summary}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("🔎 Why these songs were recommended"):
        st.caption(
            "Top matching features are based on smallest absolute difference in "
            "standardized audio-feature values (z-scores)."
        )
        for item in explanation_details:
            artists_text = (
                ", ".join(item["artists"])
                if isinstance(item["artists"], list)
                else str(item["artists"])
            )
            st.markdown(f"**#{item['rank']} {item['track']}** — {artists_text}")
            st.dataframe(
                pd.DataFrame(item["details"]).round(3),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("### ⏯️ Listen to Recommendations")
    cols = st.columns(2)
    for i, r in enumerate(results):
        with cols[i % 2]:
            row = data.iloc[r["index"]]
            st.caption(f"#{i + 1}: {row['track_name']} — {row['artists']}")
            spotify_player(row["track_id"], height=80)

    with st.expander("ℹ️ How KNN from scratch works"):
        st.markdown(
            """
**Cosine similarity** (from scratch):
1. L2-normalise every row: `X_norm = X / ||X||`
2. Query similarity: `sim = X_norm @ X_norm[query_idx]`
3. Return top-K indices (using `np.argpartition`, O(n) not O(n log n))

**Euclidean distance** (from scratch):
1. `diff = X - X[query_idx]`
2. `dist = sqrt(einsum("ij,ij->i", diff, diff))`
3. Return K smallest distances
"""
        )

    st.info(
        "💡 **Professional Note on Data Preprocessing**: "
        "All audio features (Tempo, Loudness, Energy, etc.) have been **Standardized** (using `StandardScaler`). "
        "This ensures that features with larger numerical ranges do not unfairly dominate the similarity calculation, "
        "making the recommendation balanced across all acoustic dimensions."
    )
