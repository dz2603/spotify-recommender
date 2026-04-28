import json
import os
from typing import Optional

import numpy as np
import streamlit as st


SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SRC)
DATA_DIR = os.path.join(REPO, "data")
RESULTS_DIR = os.path.join(REPO, "results", "sample_recommendations")


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
        genre_weight=1.5,
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


def spotify_player(track_id: str, height: int = 80):
    """Embed a Spotify player for a given track ID."""
    iframe = f"""
    <div style="margin-bottom:12px;">
      <div style="border-radius:12px; overflow:hidden; background:#0e1117; line-height:0; box-shadow: inset 0 0 0 1px #0e1117;">
        <iframe
          src="https://open.spotify.com/embed/track/{track_id}?theme=0"
          width="100%"
          height="{height}"
          frameborder="0"
          allowtransparency="true"
          allow="encrypted-media"
          style="display:block; border:none; margin:0; background:#121212;"
        ></iframe>
      </div>
    </div>
    """
    st.markdown(iframe, unsafe_allow_html=True)


def _pretty_feature_name(name: str) -> str:
    return name.replace("_", " ").title()


def explain_recommendation_features(
    ref_vec: np.ndarray, cand_vec: np.ndarray, feat_cols: list[str], top_n: int = 3
) -> tuple[str, list[dict]]:
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


def profile_clusters(X_num: np.ndarray, labels: np.ndarray, feat_cols: list[str]) -> dict:
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
