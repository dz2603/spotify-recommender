import numpy as np


def _normalize_to_set(value):
    """
    Convert a value into a normalized set.
    This handles strings, lists, tuples, sets, and missing values.
    """
    if value is None:
        return set()

    if isinstance(value, float) and np.isnan(value):
        return set()

    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if v is not None}

    return {str(value).strip().lower()}


def _overlap_rate(query_value, recommended_values):
    """
    Compute the share of recommended rows that overlap with the query value.
    This is useful for fields like artists or genres, which may be strings or lists.
    """
    query_set = _normalize_to_set(query_value)

    if len(query_set) == 0:
        return None

    matches = recommended_values.apply(
        lambda value: len(_normalize_to_set(value).intersection(query_set)) > 0
    )

    return float(matches.mean())


def same_artist_rate(query_row, recommended_rows):
    """
    Percentage of recommended songs that share at least one artist
    with the query song.
    """
    if recommended_rows is None or len(recommended_rows) == 0:
        return 0.0

    if "artists" not in recommended_rows.columns or "artists" not in query_row:
        return None

    return _overlap_rate(query_row["artists"], recommended_rows["artists"])


def same_genre_rate(query_row, recommended_rows):
    """
    Percentage of recommended songs that share the same genre
    with the query song.
    """
    if recommended_rows is None or len(recommended_rows) == 0:
        return 0.0

    if "track_genre" not in recommended_rows.columns or "track_genre" not in query_row:
        return None

    return _overlap_rate(query_row["track_genre"], recommended_rows["track_genre"])


def average_popularity(recommended_rows):
    """
    Average popularity score of the recommended songs.
    """
    if recommended_rows is None or len(recommended_rows) == 0:
        return 0.0

    if "popularity" not in recommended_rows.columns:
        return None

    return float(recommended_rows["popularity"].mean())


def popularity_gap(query_row, recommended_rows):
    """
    Absolute difference between the query song's popularity and the
    average popularity of the recommended songs.
    """
    if recommended_rows is None or len(recommended_rows) == 0:
        return 0.0

    if "popularity" not in recommended_rows.columns or "popularity" not in query_row:
        return None

    return float(abs(recommended_rows["popularity"].mean() - query_row["popularity"]))


def diversity_score(recommended_rows, feature_cols):
    """
    Average pairwise Euclidean distance among recommended songs.
    A higher score means the recommendations are more diverse.
    """
    if recommended_rows is None or len(recommended_rows) <= 1:
        return 0.0

    available_cols = [col for col in feature_cols if col in recommended_rows.columns]

    if len(available_cols) == 0:
        return None

    X = recommended_rows[available_cols].to_numpy()
    distances = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            distances.append(np.linalg.norm(X[i] - X[j]))

    return float(np.mean(distances))


def evaluate_recommendations(query_row, recommended_rows, feature_cols):
    """
    Return a dictionary of recommendation evaluation metrics.
    """
    return {
        "same_artist_rate": same_artist_rate(query_row, recommended_rows),
        "same_genre_rate": same_genre_rate(query_row, recommended_rows),
        "average_popularity": average_popularity(recommended_rows),
        "popularity_gap": popularity_gap(query_row, recommended_rows),
        "diversity_score": diversity_score(recommended_rows, feature_cols),
    }