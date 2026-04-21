import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def split_and_clean_artists(series):
    values = set()
    for entry in series:
        for artist in str(entry).split(";"):
            artist = artist.strip()
            if artist:
                values.add(artist)
    return sorted(values)

# Function to build our feature matrix with weighted features
# Weights can be adjusted according to user input
def build_weighted_feature_matrix(
    csv_path="dataset.csv",
    base_weight=1.0,
    explicit_weight=0.5,
    genre_weight=1.5,
    artist_weight=1.25
):
    data = pd.read_csv(csv_path)
    data = data.dropna()

    grouped = data.groupby("track_id", as_index=False).agg({
        "track_name": "first",
        "album_name": "first",
        "popularity": "first",
        "duration_ms": "first",
        "explicit": "first",
        "danceability": "mean",
        "energy": "mean",
        "key": "first",
        "loudness": "mean",
        "mode": "first",
        "speechiness": "mean",
        "acousticness": "mean",
        "instrumentalness": "mean",
        "liveness": "mean",
        "valence": "mean",
        "tempo": "mean",
        "time_signature": "first",
        "artists": split_and_clean_artists,
        "track_genre": split_and_clean_artists
    })

    data = grouped.reset_index(drop=True)
    data = data.drop_duplicates(subset="track_name", keep="first").reset_index(drop=True)

    num_features = [
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
        "time_signature"
    ]
    bin_features = ["explicit"]

    # numeric features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[num_features])

    # binary explicit feature
    X_bin = data[bin_features].astype(float).values

    # multi-label encode genres
    genre_mlb = MultiLabelBinarizer(sparse_output=True)
    X_genre = genre_mlb.fit_transform(data["track_genre"])

    # multi-label encode artists
    artist_mlb = MultiLabelBinarizer(sparse_output=True)
    X_artist = artist_mlb.fit_transform(data["artists"])

    # apply weights directly to each block
    X_num = X_num * base_weight
    X_bin = X_bin * explicit_weight
    X_genre = X_genre * genre_weight
    X_artist = X_artist * artist_weight

    # combine into one sparse matrix
    X_weighted = hstack([
        csr_matrix(X_num),
        csr_matrix(X_bin),
        X_genre,
        X_artist
    ]).tocsr()

    pipeline = {
        "scaler": scaler,
        "genre_mlb": genre_mlb,
        "artist_mlb": artist_mlb,
        "num_features": num_features,
        "bin_features": bin_features
    }

    weights = {
        "base_weight": base_weight,
        "explicit_weight": explicit_weight,
        "genre_weight": genre_weight,
        "artist_weight": artist_weight
    }

    return X_weighted, data, pipeline, weights