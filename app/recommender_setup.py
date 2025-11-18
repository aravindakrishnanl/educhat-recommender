# app/recommender_setup.py

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from pathlib import Path
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity # Needed for K-Means logic

warnings.filterwarnings('ignore')

# Define BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data Loading and Preprocessing Helpers ---

def safe_list_to_string(data):
    """Convert list or string to clean, lowercase string format."""
    if isinstance(data, list):
        # Convert to lowercase here for universal matching
        return " ".join([str(item).lower() for item in data if item])
    elif isinstance(data, str):
        return data.lower()
    else:
        return ""

def combine_career_text(row):
    """Combines all relevant text fields for TF-IDF training."""
    parts = []
    parts.append(str(row.get('career_name', '')))
    parts.append(str(row.get('category', '')))
    parts.append(safe_list_to_string(row.get('required_skills', [])))
    parts.append(safe_list_to_string(row.get('recommended_courses', [])))
    parts.append(safe_list_to_string(row.get('projects', [])))
    parts.append(str(row.get('roadmap', '')))
    return " ".join([p for p in parts if p])

# --- Global ML Components ---
careers_df = pd.DataFrame()
tfidf = None
tfidf_matrix = None # Store matrix for K-Means similarity check
career_svd_matrix = None
kmeans = None
careers_data = None
n_clusters = 8
n_components = 100

def initialize_ml_models():
    """Loads data, trains TF-IDF, SVD, and K-Means models."""
    global careers_df, tfidf, tfidf_matrix, career_svd_matrix, kmeans, careers_data, n_components, n_clusters

    try:
        json_path = os.path.join(BASE_DIR, "careers_dataset_50plus.json")
        with open(json_path, "r") as f:
            careers_data = json.load(f)
        
        careers_df = pd.DataFrame(careers_data)
        careers_df['combined_text'] = careers_df.apply(combine_career_text, axis=1)

    except FileNotFoundError:
        print("ERROR: careers_dataset_50plus.json not found. ML models cannot be initialized.")
        return

    # 1. TF-IDF
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1
    )
    tfidf_matrix = tfidf.fit_transform(careers_df['combined_text'])
    
    # Update cluster/component limits based on data size
    n_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    n_clusters = min(8, len(careers_df) // 5)
    
    # 2. SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    career_svd_matrix = svd.fit_transform(tfidf_matrix)

    # 3. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    careers_df['cluster'] = cluster_labels
    
    print(f"✅ ML Models Initialized: K-Means ({n_clusters} clusters), SVD ({n_components} components).")
    return svd # Return SVD object for use in the core function

# Initialize the models when this file is imported
SVD_MODEL = initialize_ml_models()