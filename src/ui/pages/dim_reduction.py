import streamlit as st

from src.ui.common import load_data, profile_clusters


def page_dim_reduction():
    st.header("🔍 Dimensionality Reduction: PCA vs Autoencoder")
    st.markdown(
        "Both methods compress high-dimensional song vectors to **2D or 3D** for visualization. "
        "PCA is linear; the Autoencoder can capture non-linear structure."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sample_n = st.select_slider("Sample size", [500, 1000, 2000, 5000], value=2000)
        n_pca_components = st.slider("PCA components to inspect", 2, 20, 10)
    with col2:
        ae_epochs = st.slider("Autoencoder epochs", 10, 80, 30)
    with col3:
        st.write("")
        use_3d = st.checkbox(
            "Show embeddings in 3D", value=False, help="Use 3D latent space for visualization"
        )

    if st.button("▶ Run Dimensionality Reduction", type="primary"):
        data, X_num, feat_cols, _ = load_data(sample_n=sample_n)
        hover = (
            data["track_name"]
            + " — "
            + data["artists"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        ).tolist()

        from src.algorithms.kmeans import kmeans
        from src.algorithms.pca_reduction import fit_pca, pca_variance_sweep
        from src.visualization.graphs import explained_variance_bar, loss_curve, scatter_2d, scatter_3d

        with st.spinner("Running K-Means (K=5) to generate baseline labels..."):
            km_res = kmeans(X_num, k=5)
            labels = km_res["labels"]
            profiles = profile_clusters(X_num, labels, feat_cols)

        n_viz = 3 if use_3d else 2
        with st.spinner(f"Running PCA ({n_viz}D)..."):
            pca_res = fit_pca(X_num, n_components=n_viz, feature_names=feat_cols)
            pca_sweep = pca_variance_sweep(
                X_num, max_components=n_pca_components, feature_names=feat_cols
            )

        from src.algorithms.autoencoder import TORCH_AVAILABLE, fit_autoencoder

        ae_available = False
        if TORCH_AVAILABLE:
            progress_bar = st.progress(0, text="Training Autoencoder...")

            def ae_progress(epoch, loss):
                pct = int(epoch / ae_epochs * 100)
                progress_bar.progress(pct, text=f"Epoch {epoch}/{ae_epochs} — loss {loss:.4f}")

            with st.spinner(f"Training Autoencoder ({n_viz}D latent)..."):
                ae_res = fit_autoencoder(
                    X_num, epochs=ae_epochs, latent_dim=n_viz, progress_callback=ae_progress
                )
            progress_bar.empty()

            if ae_res["available"]:
                ae_available = True
                ae_loss_hist = ae_res["loss_history"]
        else:
            st.warning(
                "⚠️ PyTorch not found. Install it with `pip install torch` to enable "
                "Autoencoder. Showing PCA only."
            )

        st.subheader("📊 Explained Variance (PCA)")
        st.plotly_chart(
            explained_variance_bar(
                pca_sweep["ratios"], pca_sweep["cumulative"], labels=pca_sweep["axis_labels"]
            ),
            use_container_width=True,
        )
        cumvar = pca_sweep["cumulative"]
        st.caption(
            f"Top {n_viz} PCs explain **{pca_res['cumulative_variance']*100:.1f}%** of variance. "
            f"First {n_pca_components} PCs explain **{cumvar[-1]*100:.1f}%**."
        )

        st.markdown("---")
        viz_title = f"{n_viz}D Embedding Visualizations"
        st.subheader(f"🗺️ {viz_title}")
        st.markdown(
            f"Points are colored by their true high-dimensional K-Means labels. Notice how they heavily overlap when forced into {n_viz}D, visually proving that clustering music with hard boundaries fails."
        )

        if use_3d:
            st.plotly_chart(
                scatter_3d(
                    pca_res["X_reduced"],
                    labels,
                    hover,
                    title="PCA 3D (Linear Compression)",
                    axis_labels=pca_res["axis_labels"],
                    cluster_names=profiles,
                ),
                use_container_width=True,
            )
            if ae_available:
                st.plotly_chart(
                    scatter_3d(
                        ae_res["X_reduced"],
                        labels,
                        hover,
                        title="Autoencoder 3D (Non-linear Compression)",
                        axis_labels=["Latent 1", "Latent 2", "Latent 3"],
                        cluster_names=profiles,
                    ),
                    use_container_width=True,
                )
        else:
            st.plotly_chart(
                scatter_2d(
                    pca_res["X_reduced"],
                    labels,
                    hover,
                    title="PCA 2D (Linear Compression)",
                    x_label=pca_res["axis_labels"][0],
                    y_label=pca_res["axis_labels"][1],
                    cluster_names=profiles,
                ),
                use_container_width=True,
            )
            if ae_available:
                st.plotly_chart(
                    scatter_2d(
                        ae_res["X_reduced"],
                        labels,
                        hover,
                        title="Autoencoder 2D (Non-linear Compression)",
                        x_label="Latent Dim 1",
                        y_label="Latent Dim 2",
                        cluster_names=profiles,
                    ),
                    use_container_width=True,
                )

        if ae_available:
            st.markdown("---")
            st.subheader("📉 Autoencoder Training Loss")
            c3, c4 = st.columns([2, 1])
            with c3:
                st.plotly_chart(loss_curve(ae_loss_hist), use_container_width=True)
            with c4:
                st.metric("Final Reconstruction Loss", f"{ae_res['final_loss']:.6f}")

        st.markdown("---")
        st.subheader("💡 What does this prove for our recommendation engine?")
        st.info(
            "**1. Music is Highly Dimensional:** Look at the PCA Explained Variance chart. In standard datasets, the first 2 components might explain 80% to 90% of the variance. Here, they only explain about 33%. We would need 10 components just to capture ~89% of the information. This mathematically proves that audio features (Energy, Acousticness, Tempo, etc.) are highly complex and largely independent.\n\n"
            "**2. Why Hard Clustering Failed:** Because the variance is spread across so many dimensions, forcing a track into a single rigid 2D visualization box (or a single K-Means 'island') inevitably destroys almost 70% of its acoustic identity. This is why our final Recommendation Engine ignores broad genres and instead relies on local, N-dimensional continuous distance (KNN) to find songs with the exact same vibal DNA."
        )
