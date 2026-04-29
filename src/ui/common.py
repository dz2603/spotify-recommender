import json
import os
from typing import Optional, List, Tuple, Dict

import numpy as np
import streamlit as st
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder

from src.algorithms.kmeans import kmeans
from src.algorithms.knn import knn_query
from src.ui import theme


SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SRC)
DATA_DIR = os.path.join(REPO, "data")
RESULTS_DIR = os.path.join(REPO, "results", "sample_recommendations")
RECOMMENDER_RECIPE_VERSION = "precomputed-style-v2-genre1.0-cluster20"


@st.cache_data(show_spinner="Loading dataset...")
def load_data(sample_n: Optional[int] = None, random_state: int = 42):
    """
    Returns (data_df, X_dense, feature_cols, X_weighted).
    X_dense: standardized numeric audio features only (for clustering / dim-red).
    """
    from src.data_preprocessing.datapreprocessing import build_weighted_feature_matrix

    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    X_weighted, data, pipeline, _ = build_weighted_feature_matrix(
        csv_path=csv_path,
        base_weight=1.0,
        explicit_weight=0.5,
        genre_weight=1.0,
        artist_weight=1.25,
    )
    num_features = pipeline["num_features"]
    X_num = pipeline["scaler"].transform(data[num_features])

    if sample_n and sample_n < len(data):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(data), size=sample_n, replace=False)
        data = data.iloc[idx].reset_index(drop=True)
        X_num = X_num[idx]

    return data, X_num, num_features, X_weighted


@st.cache_data(show_spinner="Loading lookup tables...")
def load_lookup_tables():
    sl_path = os.path.join(RESULTS_DIR, "song_lookup.json")
    nl_path = os.path.join(RESULTS_DIR, "neighbor_lookup.json")
    if not os.path.exists(sl_path) or not os.path.exists(nl_path):
        return None, None, False
    with open(sl_path, encoding="utf-8") as f:
        song_lookup = json.load(f)
    with open(nl_path, encoding="utf-8") as f:
        neighbor_lookup = json.load(f)
    return song_lookup, neighbor_lookup, True


@st.cache_data(show_spinner="Building cluster-aware recommendation matrix...")
def build_precomputed_style_matrix(
    _data,
    _X_weighted,
    cluster_k: int = 20,
    cluster_weight: float = 1.0,
    recipe_version: str = RECOMMENDER_RECIPE_VERSION,
):
    """
    Emulate the precomputed recommendation feature matrix live.

    Recipe matches `build_lookup_tables`: start with the weighted sparse feature
    matrix, run K-Means on raw audio columns, then append one-hot cluster labels.

    recipe_version intentionally participates in Streamlit's cache key. The
    dataframe and sparse matrix args are underscore-prefixed because they are
    expensive/problematic to hash directly.
    """
    _ = recipe_version
    audio_cols = [
        "duration_ms",
        "popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]
    X_dense_for_kmeans = _data[audio_cols].to_numpy(dtype=np.float32)
    kmeans_result = kmeans(X_dense_for_kmeans, k=cluster_k)
    clusters = kmeans_result["labels"]

    encoder = OneHotEncoder(sparse_output=True)
    cluster_feature = encoder.fit_transform(clusters.reshape(-1, 1)) * cluster_weight
    X_with_cluster = hstack([_X_weighted.astype(np.float32), cluster_feature]).tocsr()
    return X_with_cluster, clusters


def spotify_player(track_id: str, height: int = 80):
    """Embed a Spotify player for a given track ID."""
    iframe = f"""
    <div style="margin-bottom:{theme.SPOTIFY_EMBED_MARGIN_BOTTOM};">
      <div style="border-radius:{theme.SPOTIFY_EMBED_RADIUS}; overflow:hidden; background:{theme.BACKGROUND}; line-height:0; box-shadow: inset 0 0 0 1px {theme.BACKGROUND};">
        <iframe
          src="https://open.spotify.com/embed/track/{track_id}?theme=0"
          width="100%"
          height="{height}"
          frameborder="0"
          allowtransparency="true"
          allow="encrypted-media"
          style="display:block; border:none; margin:0; background:{theme.SPOTIFY_EMBED_BG};"
        ></iframe>
      </div>
    </div>
    """
    st.markdown(iframe, unsafe_allow_html=True)


def _pretty_feature_name(name: str) -> str:
    return name.replace("_", " ").title()


def explain_recommendation_features(
    ref_vec: np.ndarray, cand_vec: np.ndarray, feat_cols: List[str], top_n: int = 3
) -> Tuple[str, List[dict]]:
    """Return short textual reason and detailed top-matching features."""
    deltas = np.abs(cand_vec - ref_vec)
    top_idx = np.argsort(deltas)[:top_n]
    summary = ", ".join(_pretty_feature_name(feat_cols[i]) for i in top_idx)
    details = [
        {
            "Feature": _pretty_feature_name(feat_cols[i]),
            "Reference (z-score)": float(ref_vec[i]),
            "Recommended (z-score)": float(cand_vec[i]),
            "Abs Delta": float(deltas[i]),
        }
        for i in top_idx
    ]
    return summary, details


def resolve_song_query(data, query: str, selectbox_label: str = "Multiple matches — pick one:"):
    """
    Resolve a text query to a selected dataset row index.

    Returns None if no matching song exists. When multiple songs match, the most
    popular 20 matches are shown in a Streamlit selectbox.
    """
    name_col = data["track_name"].str.lower()
    idxs = name_col[name_col.str.contains(query.lower(), na=False)].index.tolist()

    if not idxs:
        return None

    if len(idxs) == 1:
        return idxs[0]

    idxs_sorted = sorted(idxs[:20], key=lambda i: data.iloc[i]["popularity"], reverse=True)
    opts = [f"{data.iloc[i]['track_name']} — {data.iloc[i]['artists']}" for i in idxs_sorted]
    chosen = st.selectbox(selectbox_label, opts)
    return idxs_sorted[opts.index(chosen)]


def get_recommendation_results(
    data,
    X_num: np.ndarray,
    X_weighted,
    query_idx: int,
    k: int,
    metric: str,
    neighbor_lookup: Optional[Dict],
    tables_ok: bool,
):
    """
    Return recommendation result dicts with `index` and `score`.

    Cosine uses the precomputed JSON lookup when it has enough neighbors for K.
    For larger K, it falls back to the same live feature recipe:
    weighted matrix + K-Means cluster one-hot feature. Euclidean stays on the
    numeric feature matrix because the augmented sparse matrix is not meaningful
    for straight-line distance.
    """
    X_query = X_num
    if metric == "cosine":
        ref_track_id = data.iloc[query_idx]["track_id"]
        precomputed_k = len(next(iter(neighbor_lookup.values()), [])) if tables_ok else 0
        if (
            tables_ok
            and neighbor_lookup
            and ref_track_id in neighbor_lookup
            and k <= precomputed_k
        ):
            results = []
            for n in neighbor_lookup[ref_track_id][:k]:
                matched = data[data["track_id"] == n["track_id"]]
                if not matched.empty:
                    results.append({"index": matched.index[0], "score": n["score"]})
            return results

        X_query, _ = build_precomputed_style_matrix(
            data,
            X_weighted,
            recipe_version=RECOMMENDER_RECIPE_VERSION,
        )

    with st.spinner("Computing neighbors..."):
        return knn_query(X_query, query_idx, k=k, metric=metric)


def profile_clusters(X_num: np.ndarray, labels: np.ndarray, feat_cols: List[str]) -> dict:
    """Generate short human-readable summary labels for clusters."""
    profiles = {}
    for c in np.unique(labels):
        mask = labels == c
        if not np.any(mask):
            profiles[c] = f"Cluster {c} (empty)"
            continue
        centroid = X_num[mask].mean(axis=0)
        top_idx = np.argsort(np.abs(centroid))[::-1]
        desc = []
        for idx in top_idx[:2]:
            val = centroid[idx]
            feat_name = feat_cols[idx]
            direction = "High" if val > 0 else "Low"
            desc.append(f"{direction} {feat_name.capitalize()}")
        profiles[c] = f"C{c} ({', '.join(desc)})"
    return profiles
