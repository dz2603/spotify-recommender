# 🎵 Spotify Recommender System

A professional machine learning project exploring music similarity, clustering patterns, and latent space representations using a Spotify audio features dataset. Built as a comprehensive demonstration of core data science algorithms.

---

## 🚀 Core Features

### 1. **Intelligent Song Recommendation**
- **KNN from Scratch**: Uses a custom-built K-Nearest Neighbors implementation (pure NumPy) for maximum transparency.
- **Dual Metrics**: Toggle between **Cosine Similarity** (directional alignment) and **Euclidean Distance** (geometric proximity).
- **Live Preview**: Integrated **Spotify Embed Players** allow for instant auditioning of the original song and all recommendations.

### 2. **Advanced Clustering Analysis**
- **Hard vs. Soft Clustering**: Compare **K-Means (implemented from scratch)** against Gaussian Mixture Models (GMM).
- **Cluster Profiling**: Centroids are automatically analyzed to generate descriptive labels (e.g., *"High Energy, Low Acousticness"*) by identifying the most extreme acoustic traits.
- **Quality Metrics**: Real-time calculation of **Silhouette Scores** and **Davies-Bouldin Index** to evaluate cluster separation.

### 3. **Dimensionality Reduction**
- **Linear vs. Non-Linear**: Compare **PCA** and **Neural Autoencoders** (PyTorch) for compressing 10+ audio dimensions into a visualizable space.
- **Variance Analysis**: Interactive bar charts showing individual and cumulative explained variance.

### 4. **Modern UI & 3D Visualization**
- **3D Projections**: Toggle between 2D and **Interactive 3D Scatter Plots** to explore high-dimensional data clouds.
- **Scientific Explanations**: The UI includes professional notes on **Feature Standardization** and the mathematical nature of continuous musical spectrums.

---

## 🛠️ Technical Highlights

### **Algorithms Implemented "From Scratch"**
To ensure complete understanding of the underlying mathematics, the following algorithms were implemented manually without higher-level library wrappers:
- **K-Nearest Neighbors (KNN)**: Custom similarity scoring engines using L2-normalization and vector dot products.
- **K-Means Clustering**: Full iterative logic including **K-means++ initialization**, assignment steps, and centroid updates.
- **Neural Autoencoder**: Custom PyTorch architecture (`128 -> 64 -> latent`) with a manual reconstruction loss training loop.

### **Data Preprocessing**
All numeric features (Tempo, Energy, Loudness, etc.) are processed through a **StandardScaler (z-score normalization)**. This ensures that features with larger raw values do not disproportionately dominate distance calculations.

---

## 📂 Project Structure
```text
spotify-recommender/
├── data/               # Dataset storage
├── src/
│   ├── algorithms/     # ML implementations (KNN, K-Means, GMM, PCA, AE)
│   ├── visualization/  # Plotly-based graph generation
│   ├── evaluation/     # Metrics (Silhouette, DBI)
│   └── app.py          # Main Streamlit interface
└── requirements.txt    # Project dependencies
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dz2603/spotify-recommender.git
   cd spotify-recommender
   ```

2. **Environment Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python3 -m streamlit run src/app.py
   ```

---

## 📊 Dataset Reference
The system utilizes a Spotify tracks dataset containing audio features for thousands of songs across multiple genres. Features included: *Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, and Tempo.*
