"""
Plotly-based visualization functions for the Spotify Recommender App.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Bold


def scatter_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    hover_texts: list[str] = None,
    title: str = "2D Scatter",
    x_label: str = "Component 1",
    y_label: str = "Component 2",
    cluster_names: dict = None,
) -> go.Figure:
    """Color-coded 2D scatter plot for cluster visualization."""
    n_clusters = len(np.unique(labels))
    colors = [PALETTE[int(l) % len(PALETTE)] for l in labels]

    fig = go.Figure()
    for c in np.unique(labels):
        mask = labels == c
        name = cluster_names[c] if cluster_names and c in cluster_names else f"Cluster {c}"
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode="markers",
            name=name,
            marker=dict(color=PALETTE[int(c) % len(PALETTE)], size=5, opacity=0.7),
            text=[hover_texts[i] for i in np.where(mask)[0]] if hover_texts else None,
            hoverinfo="text+name" if hover_texts else "x+y+name",
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="Cluster",
        height=500,
        template="plotly_white",
    )
    return fig


def comparison_scatter(
    X_pca: np.ndarray,
    X_ae: np.ndarray,
    labels: np.ndarray,
    hover_texts: list[str] = None,
    pca_x_label: str = "PC1",
    pca_y_label: str = "PC2",
) -> go.Figure:
    """Side-by-side PCA vs Autoencoder scatter plots."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["PCA 2D", "Autoencoder 2D"])

    for c in np.unique(labels):
        mask = labels == c
        color = PALETTE[int(c) % len(PALETTE)]
        ht = [hover_texts[i] for i in np.where(mask)[0]] if hover_texts else None

        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1],
            mode="markers", name=f"Cluster {c}",
            marker=dict(color=color, size=5, opacity=0.65),
            text=ht, hoverinfo="text+name" if ht else "x+y+name",
            showlegend=bool(c == np.unique(labels)[0]),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=X_ae[mask, 0], y=X_ae[mask, 1],
            mode="markers", name=f"Cluster {c}",
            marker=dict(color=color, size=5, opacity=0.65),
            text=ht, hoverinfo="text+name" if ht else "x+y+name",
            showlegend=False,
        ), row=1, col=2)

    fig.update_xaxes(title_text=pca_x_label, row=1, col=1)
    fig.update_yaxes(title_text=pca_y_label, row=1, col=1)
    fig.update_xaxes(title_text="Latent 1", row=1, col=2)
    fig.update_yaxes(title_text="Latent 2", row=1, col=2)

    fig.update_layout(height=500, template="plotly_white", title_text="PCA vs Autoencoder Embeddings")
    return fig


def elbow_curve(k_values: list[int], inertia_values: list[float]) -> go.Figure:
    """K-Means elbow curve."""
    fig = go.Figure(go.Scatter(
        x=k_values, y=inertia_values,
        mode="lines+markers",
        marker=dict(size=8, color="#636EFA"),
        line=dict(width=2),
    ))
    fig.update_layout(
        title="K-Means Elbow Curve",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (WCSS)",
        template="plotly_white",
        height=400,
    )
    return fig


def bic_curve(k_values: list[int], bic_values: list[float]) -> go.Figure:
    """GMM BIC score vs K curve."""
    fig = go.Figure(go.Scatter(
        x=k_values, y=bic_values,
        mode="lines+markers",
        marker=dict(size=8, color="#EF553B"),
        line=dict(width=2),
    ))
    fig.update_layout(
        title="GMM BIC Score (lower = better)",
        xaxis_title="Number of Components (K)",
        yaxis_title="BIC",
        template="plotly_white",
        height=400,
    )
    return fig


def explained_variance_bar(ratios: np.ndarray, cumulative: np.ndarray, labels: list[str] = None) -> go.Figure:
    """PCA explained variance bar chart with cumulative line."""
    n = len(ratios)
    if labels is not None and len(labels) == n:
        components = labels
    else:
        components = [f"PC{i+1}" for i in range(n)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=components, y=ratios * 100,
        name="Individual", marker_color="#636EFA",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=components, y=cumulative * 100,
        mode="lines+markers", name="Cumulative",
        line=dict(color="#EF553B", width=2),
    ), secondary_y=True)

    fig.update_layout(
        title="PCA Explained Variance",
        template="plotly_white",
        height=400,
        legend=dict(x=0.6, y=0.2),
    )
    fig.update_yaxes(title_text="Individual (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
    return fig


def loss_curve(loss_history: list[float]) -> go.Figure:
    """Autoencoder training loss curve."""
    fig = go.Figure(go.Scatter(
        y=loss_history, x=list(range(1, len(loss_history) + 1)),
        mode="lines+markers",
        marker=dict(size=6, color="#00CC96"),
        line=dict(width=2),
    ))
    fig.update_layout(
        title="Autoencoder Reconstruction Loss",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        template="plotly_white",
        height=350,
    )
    return fig


def soft_membership_heatmap(proba: np.ndarray, track_names: list[str], cluster_names: dict = None) -> go.Figure:
    """GMM soft membership heatmap for a sample of songs."""
    k = proba.shape[1]
    fig = go.Figure(go.Heatmap(
        z=proba,
        x=[cluster_names[i] if cluster_names and i in cluster_names else f"Cluster {i}" for i in range(k)],
        y=track_names,
        colorscale="Blues",
        colorbar=dict(title="Probability"),
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title="GMM Soft Membership Probabilities (sample)",
        height=max(400, len(track_names) * 22 + 100),
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200),
    )
    return fig
