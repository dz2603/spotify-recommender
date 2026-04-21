import json
import numpy as np
from sklearn.preprocessing import normalize
from datapreprocessing import build_weighted_feature_matrix

TOP_K = 10
BATCH_SIZE = 500  

def build_lookup_tables():
    X_weighted, data, pipeline, weights = build_weighted_feature_matrix(genre_weight=1.0)

    # Keep only needed columns
    data = data.reset_index(drop=True)
    data = data[["track_id", "track_name", "artists", "track_genre"]].copy()

    # Normalize rows so cosine similarity becomes dot product
    X_norm = normalize(X_weighted, norm="l2", axis=1)

    n_songs = X_norm.shape[0]

    neighbor_lookup = {}
    song_lookup = {}

    # Metadata lookup
    for i in range(n_songs):
        track_id = data.iloc[i]["track_id"]
        song_lookup[track_id] = {
            "track_name": data.iloc[i]["track_name"],
            "artists": data.iloc[i]["artists"],
            "track_genre": data.iloc[i]["track_genre"]
        }

    # Compute neighbors in batches
    for start in range(0, n_songs, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_songs)
        print(f"Processing rows {start} to {end - 1}...")

        batch = X_norm[start:end]

        # calculate cosine similarities
        sims = batch @ X_norm.T   

        for local_i in range(end - start):
            global_i = start + local_i
            row = sims.getrow(local_i)

            indices = row.indices
            values = row.data

            # remove self-matches
            mask = indices != global_i
            indices = indices[mask]
            values = values[mask]

            if len(values) == 0:
                top_neighbors = []
            else:
                # take top k without fully sorting everything
                top_k_idx = np.argpartition(values, -TOP_K)[-TOP_K:]
                top_indices = indices[top_k_idx]
                top_values = values[top_k_idx]

                # sort these top k descending
                order = np.argsort(-top_values)
                top_indices = top_indices[order]
                top_values = top_values[order]

                top_neighbors = []
                for neighbor_idx, score in zip(top_indices, top_values):
                    top_neighbors.append({
                        "track_id": data.iloc[neighbor_idx]["track_id"],
                        "score": round(float(score), 6)
                    })

            current_track_id = data.iloc[global_i]["track_id"]
            neighbor_lookup[current_track_id] = top_neighbors

    return song_lookup, neighbor_lookup

# save results to a json so that we can just look up results, rather than rerunning constantly
if __name__ == "__main__":
    song_lookup, neighbor_lookup = build_lookup_tables()

    with open("song_lookup.json", "w", encoding="utf-8") as f:
        json.dump(song_lookup, f, ensure_ascii=False)

    with open("neighbor_lookup.json", "w", encoding="utf-8") as f:
        json.dump(neighbor_lookup, f, ensure_ascii=False)

    print("Saved song_lookup.json and neighbor_lookup.json")