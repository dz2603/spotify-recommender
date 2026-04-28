import numpy as np
import streamlit as st

from src.ui.common import load_data, profile_clusters
from src.ui.components import divider, page_header, render_metric_cards


def page_clustering():
    page_header(
        "📊 Clustering: K-Means vs GMM",
        "Compare **hard clustering** (K-Means, each song → one cluster) "
        "vs **soft clustering** (GMM, each song → probability over clusters).",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.slider("Number of clusters (K)", 2, 15, 5)
    with col2:
        sample_n = st.select_slider(
            "Sample size", options=[1000, 2000, 5000, 10000], value=5000
        )
    with col3:
        st.write("")
        use_3d = st.checkbox(
            "Show clusters in 3D", value=False, help="Use 3 PCA components for visualization"
        )

    divider()

    if st.button("▶ Run Clustering", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n)
        hover = (
            data["track_name"]
            + " — "
            + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        ).tolist()

        from src.algorithms.gmm import fit_gmm
        from src.algorithms.kmeans import kmeans
        from src.algorithms.pca_reduction import fit_pca
        from src.evaluation.metrics import davies_bouldin, silhouette
        from src.visualization.graphs import scatter_2d, scatter_3d, soft_membership_heatmap

        n_viz = 3 if use_3d else 2
        with st.spinner(f"Running PCA for {n_viz}D visualization..."):
            pca_res = fit_pca(X_num, n_components=n_viz, feature_names=feat_cols)
            X_viz = pca_res["X_reduced"]
            axis_labels = pca_res["axis_labels"]

        with st.spinner(f"Running K-Means (K={k})..."):
            km_res = kmeans(X_num, k=k)

        with st.spinner(f"Running GMM (K={k})..."):
            gmm_res = fit_gmm(X_num, k=k)

        km_sil = silhouette(X_num, km_res["labels"])
        km_db = davies_bouldin(X_num, km_res["labels"])
        gm_sil = silhouette(X_num, gmm_res["labels"])
        gm_db = davies_bouldin(X_num, gmm_res["labels"])

        st.subheader("📐 Cluster Quality Metrics")
        render_metric_cards(
            [
                ("K-Means Silhouette ↑", f"{km_sil:.4f}"),
                ("K-Means Davies-Bouldin ↓", f"{km_db:.4f}"),
                ("GMM Silhouette ↑", f"{gm_sil:.4f}"),
                ("GMM Davies-Bouldin ↓", f"{gm_db:.4f}"),
            ]
        )

        km_profiles = profile_clusters(X_num, km_res["labels"], feat_cols)
        gm_profiles = profile_clusters(X_num, gmm_res["labels"], feat_cols)

        viz_title = f"(PCA {n_viz}D projection)"
        st.subheader(f"🗺️ Cluster Visualizations {viz_title}")

        if use_3d:
            st.plotly_chart(
                scatter_3d(
                    X_viz,
                    km_res["labels"],
                    hover,
                    title=f"K-Means (K={k})",
                    axis_labels=axis_labels,
                    cluster_names=km_profiles,
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                scatter_3d(
                    X_viz,
                    gmm_res["labels"],
                    hover,
                    title=f"GMM (K={k})",
                    axis_labels=axis_labels,
                    cluster_names=gm_profiles,
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                scatter_2d(
                    X_viz,
                    km_res["labels"],
                    hover,
                    title=f"K-Means (K={k})",
                    x_label=axis_labels[0],
                    y_label=axis_labels[1],
                    cluster_names=km_profiles,
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                scatter_2d(
                    X_viz,
                    gmm_res["labels"],
                    hover,
                    title=f"GMM (K={k})",
                    x_label=axis_labels[0],
                    y_label=axis_labels[1],
                    cluster_names=gm_profiles,
                ),
                use_container_width=True,
            )

        st.subheader("🌡️ GMM Soft Membership")
        top20_idx = np.random.choice(len(data), size=min(20, len(data)), replace=False)
        top20_names = [data.iloc[i]["track_name"][:30] for i in top20_idx]
        top20_proba = gmm_res["proba"][top20_idx]
        st.plotly_chart(
            soft_membership_heatmap(top20_proba, top20_names, cluster_names=gm_profiles),
            use_container_width=True,
        )

        divider()
        st.subheader("💡 Insight: Why do K-Means & GMM yield low Silhouette scores?")
        st.info(
            "**1. Music is a Continuous Spectrum:** Unlike textbook datasets (where clusters look like distinct islands), music transitions smoothly. You can gradually shift from a quiet lullaby to a high-energy dance track. There are no 'empty gaps' in audio feature space.\n\n"
            "**2. Suboptimal Hard Boundaries:** Because the data is one massive continuous 'blob', K-Means is forced to arbitrarily slice it. Songs on overlapping borders get penalized, heavily dragging down the Silhouette score.\n\n"
            "**3. Genre Fusion:** Modern music often blends multiple genres. A track might be mathematically 60% Hip-Hop and 40% Country. Purely numerical features also fail to capture deeper cultural and semantic context.\n\n"
            "**Conclusion:** Low clustering metrics here are not a bug, but a feature of audio data! It proves that categorizing a song into a single rigid box is mathematically unnatural. "
            "This is exactly why GMM Soft Clustering (identifying mixed vibes) or KNN (finding local nearest neighbors point-by-point) are true industry standards for modern recommendation engines."
        )

        st.subheader("💡 How are the Cluster Labels Generated?")
        st.info(
            "Notice the labels like `C0 (High Energy, Low Acousticness)`? These are generated dynamically using **Centroid Analysis** in the original high-dimensional space (not PCA!):\n\n"
            "**1. Standardization:** Our `StandardScaler` normalizes features so the global average of the Spotify dataset is 0. Positive means above average; negative means below average.\n\n"
            "**2. Centroid Calculation:** For each cluster, we average the features of all its songs to find its multi-dimensional center (the centroid).\n\n"
            "**3. Top 2 Extreme Traits:** We sort the centroid's values by absolute magnitude to find which features deviate the furthest from 0. If the top feature is `energy (+1.5)` and the second is `acousticness (-1.2)`, we automatically name it `High Energy, Low Acousticness`! This mathematically identifies the true acoustic DNA of the group."
        )
