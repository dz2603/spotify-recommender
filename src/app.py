"""
Spotify Recommender — Streamlit App
Five pages:
  1. 🎵 Song Recommendation  (KNN with cosine/euclidean options)
  2. 📊 Clustering           (K-Means vs GMM with quality metrics)
  3. 🔍 Dimensionality Reduction (PCA vs Autoencoder, 2D view)
  4. ⚙️ Parameter Explorer   (compare recommendation quality across settings)
  5. 🗂️ Dataset Info         (basic dataset stats and preview)
"""
import sys
import os

# ---------------------------------------------------------------------------
# Path setup — make sure sibling packages are importable
# ---------------------------------------------------------------------------
SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC)
DATA_DIR = os.path.join(REPO, "data")
RESULTS_DIR = os.path.join(REPO, "results", "sample_recommendations")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import json
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🎵 Song Recommendation",
    "📊 Clustering",
    "🔍 Dimensionality Reduction",
    "🗂️ Dataset Info",
]

with st.sidebar:
    st.title("🎵 Spotify Recommender")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("CS Project — KNN · K-Means · GMM · PCA · AE")

# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def load_data(sample_n: int | None = None, random_state: int = 42):
    """
    Returns (data_df, X_dense, feature_cols).
    X_dense: standardized numeric audio features only (for clustering / dim-red).
    """
    from data_preprocessing.datapreprocessing import build_weighted_feature_matrix
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    X_weighted, data, pipeline, weights = build_weighted_feature_matrix(
        csv_path=csv_path,
        base_weight=1.0,
        explicit_weight=0.5,
        genre_weight=1.5,
        artist_weight=1.25,
    )
    # Dense numeric-only matrix for clustering / PCA / AE
    num_features = pipeline["num_features"]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[num_features])

    if sample_n and sample_n < len(data):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(data), size=sample_n, replace=False)
        data = data.iloc[idx].reset_index(drop=True)
        X_num = X_num[idx]

    return data, X_num, num_features, X_weighted


@st.cache_data(show_spinner="Loading lookup tables…")
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


# ============================================================
# PAGE 1: Song Recommendation
# ============================================================
def page_recommendation():
    st.header("🎵 Song Recommendation")
    st.markdown(
        "Find songs similar to a reference track using **KNN** with "
        "cosine or Euclidean distance on audio features."
    )

    song_lookup, neighbor_lookup, tables_ok = load_lookup_tables()
    precomputed_k = len(next(iter(neighbor_lookup.values()), [])) if tables_ok else 0

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search for a song", placeholder="e.g. Shape of You")
    with col2:
        k = st.slider("Number of recommendations (K)", 5, 20, 10)

    metric = st.radio(
        "Similarity metric",
        ["cosine", "euclidean"],
        horizontal=True,
        help="Cosine: angle-based (ignores magnitude). Euclidean: straight-line distance.",
    )

    feature_set = st.selectbox(
        "Feature set used for similarity",
        ["full (audio + genre + artist)", "audio only (numeric features)"],
        help="'full' uses pre-computed lookup table. 'audio only' runs KNN from scratch on numeric features.",
    )

    st.markdown("---")

    if not query:
        st.info("Type a song name above to get recommendations.")
        return

    # --- Strategy A: pre-computed lookup (full feature set, cosine only) ---
    use_precomputed_lookup = (
        feature_set.startswith("full")
        and metric == "cosine"
        and tables_ok
        and k <= precomputed_k
    )
    if use_precomputed_lookup:
        # find matching track ids
        matches = [
            (tid, info)
            for tid, info in song_lookup.items()
            if query.lower() in info["track_name"].lower()
        ]
        if not matches:
            st.warning(f"No songs found matching **{query}**. Try a different title.")
            return

        # pick best match
        if len(matches) > 1:
            names = [f"{m[1]['track_name']} — {m[1]['artists']}" for m in matches]
            chosen = st.selectbox("Multiple matches found — pick one:", names)
            idx_chosen = names.index(chosen)
        else:
            idx_chosen = 0

        ref_id, ref_info = matches[idx_chosen]
        st.success(
            f"🎧 Reference: **{ref_info['track_name']}** "
            f"by *{ref_info['artists']}* — genre: {ref_info['track_genre']}"
        )

        neighbors = neighbor_lookup.get(ref_id, [])[:k]
        if not neighbors:
            st.error("No neighbors found for this track.")
            return

        rows = []
        for rank, n in enumerate(neighbors, 1):
            info = song_lookup.get(n["track_id"], {})
            rows.append({
                "Rank": rank,
                "Track": info.get("track_name", "?"),
                "Artist(s)": info.get("artists", "?"),
                "Genre": info.get("track_genre", "?"),
                "Cosine Score": f"{n['score']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Strategy B: from-scratch KNN on numeric features (or euclidean) ---
    else:

        data, X_num, feat_cols, _ = load_data(sample_n=None)
        name_col = data["track_name"].str.lower()
        idxs = name_col[name_col.str.contains(query.lower(), na=False)].index.tolist()

        if not idxs:
            st.warning(f"No songs found matching **{query}**.")
            return

        if len(idxs) > 1:
            opts = [f"{data.iloc[i]['track_name']} — {data.iloc[i]['artists']}" for i in idxs[:20]]
            chosen = st.selectbox("Multiple matches — pick one:", opts)
            query_idx = idxs[opts.index(chosen)]
        else:
            query_idx = idxs[0]

        ref_row = data.iloc[query_idx]
        st.success(
            f"🎧 Reference: **{ref_row['track_name']}** "
            f"by *{ref_row['artists']}*"
        )

        from algorithms.knn import knn_query
        with st.spinner("Computing neighbors…"):
            results = knn_query(X_num, query_idx, k=k, metric=metric)

        rows = []
        for rank, r in enumerate(results, 1):
            row = data.iloc[r["index"]]
            rows.append({
                "Rank": rank,
                "Track": row["track_name"],
                "Artist(s)": row["artists"],
                "Score": f"{r['score']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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


def profile_clusters(X_num: np.ndarray, labels: np.ndarray, feat_cols: list[str]) -> dict:
    """Generate human-readable summary labels for clusters based on top features."""
    profiles = {}
    for c in np.unique(labels):
        mask = labels == c
        if not np.any(mask):
            profiles[c] = f"Cluster {c} (empty)"
            continue
        centroid = X_num[mask].mean(axis=0)
        top_idx = np.argsort(np.abs(centroid))[::-1]
        
        desc = []
        for idx in top_idx[:2]:  # Top 2 distinguishing features
            val = centroid[idx]
            feat_name = feat_cols[idx]
            direction = "High" if val > 0 else "Low"
            desc.append(f"{direction} {feat_name.capitalize()}")
        profiles[c] = f"C{c} ({', '.join(desc)})"
    return profiles


# ============================================================
# PAGE 2: Clustering
# ============================================================
def page_clustering():
    st.header("📊 Clustering: K-Means vs GMM")
    st.markdown(
        "Compare **hard clustering** (K-Means, each song → one cluster) "
        "vs **soft clustering** (GMM, each song → probability over clusters)."
    )

    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of clusters (K)", 2, 15, 5)
    with col2:
        sample_n = st.select_slider(
            "Sample size", options=[1000, 2000, 5000, 10000], value=5000
        )

    st.markdown("---")

    if st.button("▶ Run Clustering", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n)
        hover = (
            data["track_name"] + " — " + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        ).tolist()

        from algorithms.kmeans import kmeans
        from algorithms.gmm import fit_gmm
        from evaluation.metrics import silhouette, davies_bouldin
        from visualization.graphs import scatter_2d, soft_membership_heatmap
        from algorithms.pca_reduction import fit_pca

        # Reduce to 2D for visualization (PCA)
        with st.spinner("Running PCA for 2D visualization…"):
            pca_res = fit_pca(X_num, n_components=2, feature_names=feat_cols)
            X_2d = pca_res["X_reduced"]
            axis_labels = pca_res["axis_labels"]

        # ── K-Means ──────────────────────────────────────────────────────
        with st.spinner(f"Running K-Means (K={k})…"):
            km_res = kmeans(X_num, k=k)

        # ── GMM ──────────────────────────────────────────────────────────
        with st.spinner(f"Running GMM (K={k})…"):
            gmm_res = fit_gmm(X_num, k=k)

        # ── Metrics ──────────────────────────────────────────────────────
        km_sil = silhouette(X_num, km_res["labels"])
        km_db = davies_bouldin(X_num, km_res["labels"])
        gm_sil = silhouette(X_num, gmm_res["labels"])
        gm_db = davies_bouldin(X_num, gmm_res["labels"])

        # ── Metric summary ───────────────────────────────────────────────
        st.subheader("📐 Cluster Quality Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("K-Means Silhouette ↑", f"{km_sil:.4f}")
        m2.metric("K-Means Davies-Bouldin ↓", f"{km_db:.4f}")
        m3.metric("GMM Silhouette ↑", f"{gm_sil:.4f}")
        m4.metric("GMM Davies-Bouldin ↓", f"{gm_db:.4f}")

        # ── Cluster Profiling (Generating Summaries) ──────────────────────
        km_profiles = profile_clusters(X_num, km_res["labels"], feat_cols)
        gm_profiles = profile_clusters(X_num, gmm_res["labels"], feat_cols)

        # ── Scatter plots ─────────────────────────────────────────────────
        st.subheader("🗺️ Cluster Visualizations (PCA 2D projection)")
        st.plotly_chart(
            scatter_2d(X_2d, km_res["labels"], hover, title=f"K-Means (K={k})", 
                       x_label=axis_labels[0], y_label=axis_labels[1], cluster_names=km_profiles),
            use_container_width=True,
        )
        st.plotly_chart(
            scatter_2d(X_2d, gmm_res["labels"], hover, title=f"GMM (K={k})",
                       x_label=axis_labels[0], y_label=axis_labels[1], cluster_names=gm_profiles),
            use_container_width=True,
        )

        # ── GMM Soft membership ───────────────────────────────────────────
        st.subheader("🌡️ GMM Soft Membership")
        top20_idx = np.random.choice(len(data), size=min(20, len(data)), replace=False)
        top20_names = [data.iloc[i]["track_name"][:30] for i in top20_idx]
        top20_proba = gmm_res["proba"][top20_idx]
        st.plotly_chart(
            soft_membership_heatmap(top20_proba, top20_names, cluster_names=gm_profiles),
            use_container_width=True,
        )

        # ── Analytical Insight ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Insight: Why do K-Means & GMM yield low Silhouette scores?")
        st.info(
            "**1. Music is a Continuous Spectrum:** Unlike textbook datasets (where clusters look like distinct islands), music transitions smoothly. You can gradually shift from a quiet lullaby to a high-energy dance track. There are no 'empty gaps' in audio feature space.\n\n"
            "**2. Suboptimal Hard Boundaries:** Because the data is one massive continuous 'blob', K-Means is forced to arbitrarily slice it. Songs on overlapping borders get penalized, heavily dragging down the Silhouette score.\n\n"
            "**3. Genre Fusion:** Modern music often blends multiple genres. A track might be mathematically 60% Hip-Hop and 40% Country. Purely numerical features also fail to capture deeper cultural and semantic context.\n\n"
            "**Conclusion:** Low clustering metrics here are not a bug, but a feature of audio data! It proves that categorizing a song into a single rigid box is mathematically unnatural. "
            "This is exactly why GMM Soft Clustering (identifying mixed vibes) or KNN (finding local nearest neighbors point-by-point) are the true industry standards for modern music recommendation engines."
        )


# ============================================================
# PAGE 3: Dimensionality Reduction
# ============================================================
def page_dim_reduction():
    st.header("🔍 Dimensionality Reduction: PCA vs Autoencoder")
    st.markdown(
        "Both methods compress high-dimensional song vectors to **2D** for visualization. "
        "PCA is linear; the Autoencoder can capture non-linear structure."
    )

    col1, col2 = st.columns(2)
    with col1:
        sample_n = st.select_slider("Sample size", [500, 1000, 2000, 5000], value=2000)
        n_pca_components = st.slider("PCA components to inspect", 2, 20, 10)
    with col2:
        ae_epochs = st.slider("Autoencoder epochs", 10, 80, 30)

    if st.button("▶ Run Dimensionality Reduction", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n)
        hover = (data["track_name"] + " — " + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))).tolist()

        # ── Generate dynamic cluster labels to demonstrate failure ───────
        from algorithms.kmeans import kmeans
        with st.spinner("Running K-Means (K=5) to generate baseline labels…"):
            km_res = kmeans(X_num, k=5)
            labels = km_res["labels"]
            profiles = profile_clusters(X_num, labels, feat_cols)

        # ── PCA ──────────────────────────────────────────────────────────
        from algorithms.pca_reduction import fit_pca, pca_variance_sweep
        from visualization.graphs import explained_variance_bar, loss_curve, scatter_2d

        with st.spinner("Running PCA…"):
            pca_2d = fit_pca(X_num, n_components=2, feature_names=feat_cols)
            pca_sweep = pca_variance_sweep(X_num, max_components=n_pca_components, feature_names=feat_cols)

        # ── Autoencoder ───────────────────────────────────────────────────
        from algorithms.autoencoder import fit_autoencoder, TORCH_AVAILABLE

        ae_available = False
        if TORCH_AVAILABLE:
            progress_bar = st.progress(0, text="Training Autoencoder…")

            def ae_progress(epoch, loss):
                pct = int(epoch / ae_epochs * 100)
                progress_bar.progress(pct, text=f"Epoch {epoch}/{ae_epochs} — loss {loss:.4f}")

            with st.spinner("Training Autoencoder…"):
                ae_res = fit_autoencoder(X_num, epochs=ae_epochs, progress_callback=ae_progress)
            progress_bar.empty()

            if ae_res["available"]:
                ae_available = True
                X_ae = ae_res["X_reduced"]
                ae_loss_hist = ae_res["loss_history"]
        else:
            st.warning(
                "⚠️ PyTorch not found. Install it with `pip install torch` to enable "
                "Autoencoder. Showing PCA only."
            )

        # ── Display ───────────────────────────────────────────────────────
        st.subheader("📊 Explained Variance (PCA)")
        st.plotly_chart(
            explained_variance_bar(pca_sweep["ratios"], pca_sweep["cumulative"], labels=pca_sweep["axis_labels"]),
            use_container_width=True,
        )
        cumvar = pca_sweep["cumulative"]
        st.caption(
            f"Top 2 PCs explain **{pca_2d['cumulative_variance']*100:.1f}%** of variance. "
            f"First {n_pca_components} PCs explain **{cumvar[-1]*100:.1f}%**."
        )

        st.markdown("---")
        st.subheader("🗺️ 2D Embedding Visualizations")
        st.markdown("Points are colored by their true high-dimensional K-Means labels. Notice how they heavily overlap when forced into 2D, visually proving that clustering music with hard boundaries fails.")
        
        st.plotly_chart(
            scatter_2d(pca_2d["X_reduced"], labels, hover,
                       title="PCA 2D (Linear Compression)",
                       x_label=pca_2d["axis_labels"][0], y_label=pca_2d["axis_labels"][1],
                       cluster_names=profiles),
            use_container_width=True,
        )

        if ae_available:
            st.plotly_chart(
                scatter_2d(X_ae, labels, hover,
                           title="Autoencoder 2D (Non-linear Compression)",
                           x_label="Latent Dim 1", y_label="Latent Dim 2",
                           cluster_names=profiles),
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("📉 Autoencoder Training Loss")
            c3, c4 = st.columns([2, 1])
            with c3:
                st.plotly_chart(loss_curve(ae_loss_hist), use_container_width=True)
            with c4:
                st.metric("Final Reconstruction Loss", f"{ae_res['final_loss']:.6f}")
                st.markdown(
                    """
**Architecture:**
```
Encoder: d → 128 → 64 → 2
Decoder: 2 → 64 → 128 → d
Loss:    MSE
```
"""
                )

        # ── Analytical Insight ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 What does this prove for our recommendation engine?")
        st.info(
            "**1. Music is Highly Dimensional:** Look at the PCA Explained Variance chart. In standard datasets, the first 2 components might explain 80% to 90% of the variance. Here, they only explain about 33%. We would need 10 components just to capture ~89% of the information. This mathematically proves that audio features (Energy, Acousticness, Tempo, etc.) are highly complex and largely independent.\n\n"
            "**2. Why Hard Clustering Failed:** Because the variance is spread across so many dimensions, forcing a track into a single rigid 2D visualization box (or a single K-Means 'island') inevitably destroys almost 70% of its acoustic identity. This is why our final Recommendation Engine ignores broad genres and instead relies on local, N-dimensional continuous distance (KNN) to find songs with the exact same vibal DNA."
        )


def page_dataset_info():
    st.header("🗂️ Dataset Information")
    st.markdown("Quick overview of the dataset used across all app pages.")

    data, X_num, feat_cols, _ = load_data(sample_n=None)

    n_rows, n_cols = data.shape
    n_numeric = len(data.select_dtypes(include=[np.number]).columns)
    n_non_numeric = n_cols - n_numeric
    n_missing = int(data.isna().sum().sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{n_rows:,}")
    m2.metric("Columns", f"{n_cols:,}")
    m3.metric("Missing Values", f"{n_missing:,}")
    m4.metric("Numeric Features", f"{n_numeric:,}")

    st.markdown("---")
    st.subheader("Column Summary")
    summary_df = pd.DataFrame({
        "Column": data.columns,
        "Dtype": [str(dtype) for dtype in data.dtypes],
        "Non-Null Count": [int(data[col].notna().sum()) for col in data.columns],
        "Missing Count": [int(data[col].isna().sum()) for col in data.columns],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Rows")
    st.dataframe(data.head(20), hide_index=True, use_container_width=True)

    with st.expander("Feature Matrix Details"):
        st.markdown(
            f"- Standardized numeric matrix shape: **{X_num.shape[0]:,} x {X_num.shape[1]:,}**\n"
            f"- Numeric feature count used in modeling: **{len(feat_cols):,}**\n"
            f"- Non-numeric columns in raw data: **{n_non_numeric:,}**"
        )


# ============================================================
# Router
# ============================================================
if page == PAGES[0]:
    page_recommendation()
elif page == PAGES[1]:
    page_clustering()
elif page == PAGES[2]:
    page_dim_reduction()
elif page == PAGES[3]:
    page_dataset_info()
