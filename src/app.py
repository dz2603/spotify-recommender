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
    "⚙️ Parameter Explorer",
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


# ============================================================
# PAGE 2: Clustering
# ============================================================
def page_clustering():
    st.header("📊 Clustering: K-Means vs GMM")
    st.markdown(
        "Compare **hard clustering** (K-Means, each song → one cluster) "
        "vs **soft clustering** (GMM, each song → probability over clusters)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.slider("Number of clusters (K)", 2, 15, 5)
    with col2:
        sample_n = st.select_slider(
            "Sample size", options=[1000, 2000, 5000, 10000], value=5000
        )
    with col3:
        run_elbow = st.checkbox("Run elbow / BIC sweep (K=2..10)", value=False)

    st.markdown("---")

    if st.button("▶ Run Clustering", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n)
        hover = (
            data["track_name"] + " — " + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        ).tolist()

        from algorithms.kmeans import kmeans, elbow_analysis
        from algorithms.gmm import fit_gmm, gmm_bic_sweep
        from evaluation.metrics import silhouette, davies_bouldin
        from visualization.graphs import (
            scatter_2d, elbow_curve, bic_curve, soft_membership_heatmap
        )
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

        m5, m6 = st.columns(2)
        m5.metric("K-Means Inertia", f"{km_res['inertia']:.1f}")
        m6.metric("GMM BIC (lower=better)", f"{gmm_res['bic']:.1f}")

        # ── Scatter plots ─────────────────────────────────────────────────
        st.subheader("🗺️ Cluster Visualizations (PCA 2D projection)")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.plotly_chart(
                scatter_2d(X_2d, km_res["labels"], hover, title=f"K-Means (K={k})", 
                           x_label=axis_labels[0], y_label=axis_labels[1]),
                use_container_width=True,
            )
        with sc2:
            st.plotly_chart(
                scatter_2d(X_2d, gmm_res["labels"], hover, title=f"GMM (K={k})",
                           x_label=axis_labels[0], y_label=axis_labels[1]),
                use_container_width=True,
            )

        # ── GMM Soft membership ───────────────────────────────────────────
        st.subheader("🌡️ GMM Soft Membership (top 20 songs)")
        top20_idx = np.random.choice(len(data), size=min(20, len(data)), replace=False)
        top20_names = [data.iloc[i]["track_name"][:30] for i in top20_idx]
        top20_proba = gmm_res["proba"][top20_idx]
        st.plotly_chart(
            soft_membership_heatmap(top20_proba, top20_names),
            use_container_width=True,
        )

        # ── Elbow / BIC sweep ─────────────────────────────────────────────
        if run_elbow:
            st.subheader("📈 Elbow & BIC Sweep (K = 2…10)")
            k_range = range(2, 11)
            with st.spinner("Running elbow analysis…"):
                elbow_data = elbow_analysis(X_num, k_range)
            with st.spinner("Running GMM BIC sweep…"):
                bic_data = gmm_bic_sweep(X_num, k_range)

            el1, el2 = st.columns(2)
            with el1:
                st.plotly_chart(
                    elbow_curve(list(elbow_data.keys()), list(elbow_data.values())),
                    use_container_width=True,
                )
            with el2:
                st.plotly_chart(
                    bic_curve(list(bic_data.keys()), list(bic_data.values())),
                    use_container_width=True,
                )

        # ── Comparison table ──────────────────────────────────────────────
        st.subheader("📋 Hard vs Soft: What's the difference?")
        st.markdown(
            """
| Property | K-Means (Hard) | GMM (Soft) |
|---|---|---|
| Assignment | Each song → exactly 1 cluster | Each song → probability over all clusters |
| Cluster shape | Spherical (Euclidean) | Elliptical (covariance matrix) |
| Interpretability | Simple, fast | Richer, probabilistic |
| Ambiguous songs | Forced to one cluster | High entropy across clusters |
| Selection criteria | Inertia / Elbow | BIC / AIC |
"""
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
        color_by = st.selectbox("Color points by", ["K-Means cluster (K=8)", "Genre (top 10)", "None"])

    if st.button("▶ Run Dimensionality Reduction", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n)
        hover = (data["track_name"] + " — " + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))).tolist()

        # ── Colour labels ─────────────────────────────────────────────────
        if color_by.startswith("K-Means"):
            from algorithms.kmeans import kmeans
            with st.spinner("Running K-Means for coloring…"):
                km_res = kmeans(X_num, k=8)
            labels = km_res["labels"]
            label_note = "colored by K-Means cluster (K=8)"

        elif color_by.startswith("Genre"):
            # top-10 genres
            if "track_genre" in data.columns:
                genre_series = data["track_genre"]
                if isinstance(genre_series.iloc[0], list):
                    flat = genre_series.apply(lambda g: g[0] if g else "unknown")
                else:
                    flat = genre_series.apply(lambda g: str(g)[:20])
                top10 = flat.value_counts().index[:10].tolist()
                labels = flat.apply(lambda g: top10.index(g) if g in top10 else 10).values
            else:
                labels = np.zeros(len(data), dtype=int)
            label_note = "colored by genre"
        else:
            labels = np.zeros(len(data), dtype=int)
            label_note = "no coloring"

        # ── PCA ──────────────────────────────────────────────────────────
        from algorithms.pca_reduction import fit_pca, pca_variance_sweep
        from visualization.graphs import (
            scatter_2d, comparison_scatter,
            explained_variance_bar, loss_curve,
        )

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

        if ae_available:
            st.subheader(f"🗺️ PCA vs Autoencoder — 2D scatter ({label_note})")
            st.plotly_chart(
                comparison_scatter(
                    pca_2d["X_reduced"], X_ae, labels, hover,
                    pca_x_label=pca_2d["axis_labels"][0],
                    pca_y_label=pca_2d["axis_labels"][1]
                ),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    scatter_2d(pca_2d["X_reduced"], labels, hover,
                               title=f"PCA 2D ({label_note})",
                               x_label=pca_2d["axis_labels"][0], y_label=pca_2d["axis_labels"][1]),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    scatter_2d(X_ae, labels, hover,
                               title=f"Autoencoder 2D ({label_note})",
                               x_label="Latent Dim 1", y_label="Latent Dim 2"),
                    use_container_width=True,
                )

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
        else:
            st.subheader(f"🗺️ PCA 2D scatter ({label_note})")
            st.plotly_chart(
                scatter_2d(pca_2d["X_reduced"], labels, hover,
                           title=f"PCA 2D ({label_note})",
                           x_label=pca_2d["axis_labels"][0], y_label=pca_2d["axis_labels"][1]),
                use_container_width=True,
            )

        # ── Comparison table ───────────────────────────────────────────────
        st.subheader("📋 PCA vs Autoencoder Comparison")
        st.markdown(
            """
| Property | PCA | Autoencoder |
|---|---|---|
| Type | Linear projection | Non-linear (neural network) |
| Training | Eigen-decomposition (no gradient) | Gradient descent (epochs needed) |
| Interpretability | Principal directions = linear combos of features | Latent space less interpretable |
| Speed | Very fast | Slower (GPU helps) |
| Reconstruction quality | Good for linear structure | Better for complex manifolds |
| Variance explained | Quantifiable (explained variance ratio) | Measured by reconstruction loss |
"""
        )


# ============================================================
# PAGE 4: Parameter Explorer
# ============================================================
def page_parameter_explorer():
    st.header("⚙️ Parameter Explorer")
    st.markdown(
        "Compare how **K**, **similarity metric**, and **feature set** "
        "influence recommendation quality."
    )

    st.subheader("🔬 Recommendation Comparison")
    query = st.text_input("Reference song", placeholder="e.g. Blinding Lights")
    k = st.slider("K (neighbors)", 5, 20, 10, key="pe_k")

    if query and st.button("▶ Compare Metrics", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=None)
        name_col = data["track_name"].str.lower()
        idxs = name_col[name_col.str.contains(query.lower(), na=False)].index.tolist()

        if not idxs:
            st.warning(f"No songs found matching **{query}**.")
            return

        query_idx = idxs[0]
        ref_row = data.iloc[query_idx]
        st.success(f"🎧 Reference: **{ref_row['track_name']}** by *{ref_row['artists']}*")

        from algorithms.knn import knn_query

        col1, col2 = st.columns(2)
        for col, metric in zip([col1, col2], ["cosine", "euclidean"]):
            with col:
                st.markdown(f"#### {metric.capitalize()} Distance")
                with st.spinner(f"KNN ({metric})…"):
                    results = knn_query(X_num, query_idx, k=k, metric=metric)
                rows = []
                for rank, r in enumerate(results, 1):
                    row = data.iloc[r["index"]]
                    rows.append({
                        "Rank": rank,
                        "Track": row["track_name"],
                        "Artist": row["artists"],
                        "Score": f"{r['score']:.4f}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Overlap analysis ───────────────────────────────────────────────
        st.subheader("🔄 Overlap Between Cosine and Euclidean")
        cos_res = knn_query(X_num, query_idx, k=k, metric="cosine")
        euc_res = knn_query(X_num, query_idx, k=k, metric="euclidean")
        cos_idx = {r["index"] for r in cos_res}
        euc_idx = {r["index"] for r in euc_res}
        overlap = cos_idx & euc_idx

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Cosine-only results", len(cos_idx - euc_idx))
        oc2.metric("Shared results", len(overlap))
        oc3.metric("Euclidean-only results", len(euc_idx - cos_idx))

        if overlap:
            shared = [data.iloc[i]["track_name"] for i in list(overlap)[:10]]
            st.markdown("**Songs recommended by BOTH metrics:** " + ", ".join(shared))

    # ── Clustering parameter comparison ───────────────────────────────────
    st.markdown("---")
    st.subheader("🎛️ Clustering Parameter Comparison")

    cc1, cc2 = st.columns(2)
    with cc1:
        k_a = st.slider("K (setting A)", 2, 15, 3, key="ka")
    with cc2:
        k_b = st.slider("K (setting B)", 2, 15, 8, key="kb")

    sample_n_pe = st.select_slider("Sample size (clustering)", [1000, 2000, 5000], value=2000, key="pe_sn")

    if st.button("▶ Compare Clustering Settings", type="primary", key="cmp_btn"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n_pe)
        hover = (data["track_name"] + " — " + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))).tolist()

        from algorithms.kmeans import kmeans
        from algorithms.gmm import fit_gmm
        from evaluation.metrics import silhouette, davies_bouldin
        from algorithms.pca_reduction import fit_pca
        from visualization.graphs import scatter_2d

        with st.spinner("Running PCA…"):
            pca_res = fit_pca(X_num, n_components=2, feature_names=feat_cols)
            X_2d = pca_res["X_reduced"]
            axis_labels = pca_res["axis_labels"]

        results = {}
        for k_val in [k_a, k_b]:
            with st.spinner(f"K-Means K={k_val}…"):
                km = kmeans(X_num, k=k_val)
            with st.spinner(f"GMM K={k_val}…"):
                gm = fit_gmm(X_num, k=k_val)
            results[k_val] = {
                "km": km, "gm": gm,
                "km_sil": silhouette(X_num, km["labels"]),
                "gm_sil": silhouette(X_num, gm["labels"]),
                "km_db": davies_bouldin(X_num, km["labels"]),
                "gm_db": davies_bouldin(X_num, gm["labels"]),
            }

        # Metrics table
        rows = []
        for k_val in [k_a, k_b]:
            r = results[k_val]
            rows.append({
                "K": k_val,
                "KM Silhouette ↑": f"{r['km_sil']:.4f}",
                "KM Davies-Bouldin ↓": f"{r['km_db']:.4f}",
                "KM Inertia": f"{r['km']['inertia']:.1f}",
                "GMM Silhouette ↑": f"{r['gm_sil']:.4f}",
                "GMM Davies-Bouldin ↓": f"{r['gm_db']:.4f}",
                "GMM BIC": f"{r['gm']['bic']:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Scatter grid
        col_a, col_b = st.columns(2)
        for col, k_val in zip([col_a, col_b], [k_a, k_b]):
            with col:
                st.markdown(f"**K = {k_val}**")
                st.plotly_chart(
                    scatter_2d(X_2d, results[k_val]["km"]["labels"], hover,
                               title=f"K-Means K={k_val}",
                               x_label=axis_labels[0], y_label=axis_labels[1]),
                    use_container_width=True,
                )
                st.plotly_chart(
                    scatter_2d(X_2d, results[k_val]["gm"]["labels"], hover,
                               title=f"GMM K={k_val}",
                               x_label=axis_labels[0], y_label=axis_labels[1]),
                    use_container_width=True,
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
    page_parameter_explorer()
elif page == PAGES[4]:
    page_dataset_info()
