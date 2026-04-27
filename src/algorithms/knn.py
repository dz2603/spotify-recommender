"""
KNN from scratch — pure numpy implementation.
No sklearn neighbors used; only numpy for all similarity/distance computation.
"""
import json
import numpy as np
from src.algorithms.kmeans import kmeans
from src.data_preprocessing.datapreprocessing import build_weighted_feature_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Similarity / Distance primitives (all pure numpy)
# ---------------------------------------------------------------------------



def cosine_similarity_row(X_norm: np.ndarray, query_idx: int) -> np.ndarray:
    """
    Cosine similarity between one query row and all rows in X_norm.
    X_norm must already be L2-normalised.
    Returns shape (n,) similarity scores in [-1, 1].
    """
    return X_norm @ X_norm[query_idx]  # dot product of normalized vectors


def euclidean_distance_row(X: np.ndarray, query_idx: int) -> np.ndarray:
    """
    Euclidean distance between one query row and all rows in X.
    Returns shape (n,) distances.
    """
    diff = X - X[query_idx]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


# ---------------------------------------------------------------------------
# KNN query (from scratch)
# ---------------------------------------------------------------------------

def knn_query(
    X: np.ndarray,
    query_idx: int,
    k: int = 10,
    metric: str = "cosine",
) -> list[dict]:
    """
    Return the top-k nearest neighbors for a given song index.

    Args:
        X        : feature matrix (n_songs, n_features), raw unnormalized
        query_idx: row index of the reference song
        k        : number of neighbors to return
        metric   : "cosine" or "euclidean"

    Returns:
        List of dicts with keys 'index' and 'score' (higher = more similar for
        cosine; lower = closer for euclidean — we negate for consistency).
    """
    n = X.shape[0]
    if metric == "cosine":
        X_norm = normalize(X, norm="l2", axis=1)
        scores = cosine_similarity_row(X_norm, query_idx)
        scores[query_idx] = -2.0  # exclude self
        top_idx = np.argpartition(scores, -(k + 1))[-(k + 1):]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_idx = top_idx[top_idx != query_idx][:k]
        return [{"index": int(i), "score": float(scores[i])} for i in top_idx]
    elif metric == "euclidean":
        dists = euclidean_distance_row(X, query_idx)
        dists[query_idx] = 1e18  # exclude self
        top_idx = np.argpartition(dists, k)[:k]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return [{"index": int(i), "score": float(-dists[i])} for i in top_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")


# ---------------------------------------------------------------------------
# Batch lookup table builder (kept for pre-computation, uses above primitives)
# ---------------------------------------------------------------------------
def build_lookup_tables(csv_path=r"data\dataset.csv", top_k=10):
    """Pre-compute and save neighbor_lookup.json and song_lookup.json."""

    X_weighted, data, pipeline, weights = build_weighted_feature_matrix(
        csv_path=csv_path,
        genre_weight=1.0
    )

    data = data.reset_index(drop=True)

    # keep sparse matrix for KNN
    X_sparse = X_weighted.astype(np.float32)

    # make dense ONLY for k-means
    audio_cols = [
        "duration_ms", "popularity", "danceability", "energy", "key",
        "loudness", "mode", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo",
        "time_signature"
    ]

    X_dense_for_kmeans = data[audio_cols].to_numpy(dtype=np.float32)

    # run k-means
    k = 20
    kmeans_result = kmeans(X_dense_for_kmeans, k=k)

    clusters = kmeans_result["labels"]
    centroids = kmeans_result["centroids"]

    data["cluster"] = clusters

    # add cluster assignment as a new KNN feature
    cluster_weight = 1.0
    encoder = OneHotEncoder(sparse_output =True)
    cluster_feature = encoder.fit_transform(data[["cluster"]])
    cluster_feature = cluster_feature * cluster_weight

    # combine original weighted features + cluster feature
    from scipy.sparse import csr_matrix, hstack
    X_with_cluster = hstack([X_sparse, cluster_feature])

    # Keep metadata columns, now including real cluster
    data = data[["track_id", "track_name", "artists", "track_genre", "cluster"]].copy()

    n_songs = X_with_cluster.shape[0]
    song_lookup = {}
    neighbor_lookup = {}

    for i in range(n_songs):
        track_id = data.iloc[i]["track_id"]
        song_lookup[track_id] = {
            "track_name": data.iloc[i]["track_name"],
            "artists": data.iloc[i]["artists"],
            "track_genre": data.iloc[i]["track_genre"],
            "cluster": int(data.iloc[i]["cluster"]),
        }

    BATCH = 500
    X_norm = normalize(X_with_cluster, norm="l2", axis=1)

    for start in range(0, n_songs, BATCH):
        end = min(start + BATCH, n_songs)
        print(f"Processing rows {start}-{end - 1}...")

        batch = X_norm[start:end]
        sims = batch @ X_norm.T
        sims = sims.toarray()

        for local_i in range(end - start):
            global_i = start + local_i
            row = sims[local_i].copy()
            row[global_i] = -2.0

            top_k_idx = np.argpartition(row, -(top_k + 1))[-(top_k + 1):]
            top_k_idx = top_k_idx[np.argsort(-row[top_k_idx])]
            top_k_idx = top_k_idx[top_k_idx != global_i][:top_k]

            track_id = data.iloc[global_i]["track_id"]
            neighbor_lookup[track_id] = [
                {
                    "track_id": data.iloc[j]["track_id"],
                    "track_name": data.iloc[j]["track_name"],
                    "artists": data.iloc[j]["artists"],
                    "cluster": int(data.iloc[j]["cluster"]),
                    "score": round(float(row[j]), 6),
                }
                for j in top_k_idx
            ]

    return song_lookup, neighbor_lookup


if __name__ == "__main__":
    song_lookup, neighbor_lookup = build_lookup_tables()
    with open("song_lookup.json", "w", encoding="utf-8") as f:
        json.dump(song_lookup, f, ensure_ascii=False)
    with open("neighbor_lookup.json", "w", encoding="utf-8") as f:
        json.dump(neighbor_lookup, f, ensure_ascii=False)
    print("Saved song_lookup.json and neighbor_lookup.json")