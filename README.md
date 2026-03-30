# spotify-recommender

This project builds a Spotify song recommendation and clustering system using audio features from a Spotify tracks dataset. Given a reference song, the system recommends similar songs based on feature similarity. We also use clustering to analyze broader groupings in the dataset.

## Methods Implemented
- vector similarity for recommendation
- K-means clustering
- PCA visualization

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
