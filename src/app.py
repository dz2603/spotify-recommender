"""
Spotify Recommender — Streamlit App

Five pages (see PAGES in the sidebar router):
  1. 🎵 Song Recommendation — KNN recommendations (cosine / euclidean)
  2. ✏️ Recommendation Evaluation — sanity metrics across songs/settings
  3. 📊 Clustering — K-Means vs GMM, metrics and plots
  4. 🔍 Dimensionality Reduction — PCA vs autoencoder embeddings
  5. 🗂️ Dataset Info — table shape, column summary, sample rows
"""
import os
import sys

import streamlit as st

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.ui.pages.clustering import page_clustering
from src.ui.pages.dataset_info import page_dataset_info
from src.ui.pages.dim_reduction import page_dim_reduction
from src.ui.pages.recommendation import page_recommendation
from src.ui.pages.recommendation_evaluation import page_recommendation_evaluation

st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "🎵 Song Recommendation",
    "✏️ Recommendation Evaluation",
    "📊 Clustering",
    "🔍 Dimensionality Reduction",
    "🗂️ Dataset Info",
]

with st.sidebar:
    st.title("🎵 Spotify Recommender")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="sidebar_nav")
    st.markdown("---")
    st.caption("CS Project — KNN · K-Means · GMM · PCA · AE")

if page == PAGES[0]:
    page_recommendation()
elif page == PAGES[1]:
    page_recommendation_evaluation()
elif page == PAGES[2]:
    page_clustering()
elif page == PAGES[3]:
    page_dim_reduction()
elif page == PAGES[4]:
    page_dataset_info()
