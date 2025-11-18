# app/recommender_core.py

import pandas as pd
import networkx as nx
import os 
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from . import recommender_setup as setup 
from . import models as setup_models # Needed to query Workspace model
import numpy as np 
from typing import Any, Dict

# Define BASE_DIR 
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Graph Data Loading ---

def load_graph_data():
    """Loads careers data and builds the NetworkX graph for the GRAPH model."""
    try:
        json_path = os.path.join(BASE_DIR, "careers_dataset_50plus.json")
        df = pd.read_json(json_path) 
    except FileNotFoundError:
        print("ERROR: Graph data file not found.")
        return pd.DataFrame(), nx.Graph()

    def fix_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        return [s.strip() for s in str(x).split(",")]

    df["required_skills"] = df["required_skills"].apply(fix_list)
    df["recommended_courses"] = df["recommended_courses"].apply(fix_list)
    df["projects"] = df["projects"].apply(fix_list)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["career_name"], type="career")
        for skill in row["required_skills"]:
            G.add_node(skill.lower(), type="skill") 
            G.add_edge(row["career_name"], skill.lower(), weight=3) 
        for course in row["recommended_courses"]:
            G.add_node(course.lower(), type="course")
            G.add_edge(row["career_name"], course.lower(), weight=1)
        for proj in row["projects"]:
            G.add_node(proj.lower(), type="project")
            G.add_edge(row["career_name"], proj.lower(), weight=1)
    
    return df, G

# Load Graph Data globally once
DF_GRAPH, G_CORE = load_graph_data()

# --- Helper function for ML Models (K-Means/SVD) ---

def combine_user_input_ml(user_data: Dict[str, Any]) -> str:
    """Combines user profile into a single text string for TF-IDF vectorization."""
    user_text_parts = []
    
    user_text_parts.append(user_data.get('branch', ''))
    user_text_parts.append(user_data.get('interests', ''))
    
    user_text_parts.append(' '.join(user_data.get('skills', []))) 
    user_text_parts.append(' '.join(user_data.get('projects_completed', [])))
    user_text_parts.append(' '.join(user_data.get('certifications_completed', [])))

    return ' '.join([p.lower() for p in user_text_parts if p])

# --- ONLINE LEARNING FUNCTION ---

def update_graph_weights(career_name: str, user_id: int, db: Any):
    """
    Increases edge weights in the core graph between the user's relevant nodes 
    and a positively rated career's requirements.
    """
    global G_CORE
    global DF_GRAPH
    
    # 1. Look up the user's most recent workspace for context
    try:
        user_workspace = db.query(setup_models.Workspace).filter(
            setup_models.Workspace.user_id == user_id
        ).order_by(setup_models.Workspace.created_at.desc()).first()
        
        if not user_workspace:
            return

        user_skills = [s.strip().lower() for s in user_workspace.skills.split(",") if s.strip()]
        user_interests = [i.strip().lower() for i in user_workspace.interests.split(",") if i.strip()]
        
    except Exception as e:
        print(f"ERROR: Database lookup failed during graph update: {e}")
        return

    # 2. Find the target career's requirements from the DataFrame
    try:
        career_row = DF_GRAPH[DF_GRAPH["career_name"] == career_name].iloc[0]
        required_skills = [s.lower() for s in career_row["required_skills"]] 
    except IndexError:
        print(f"ERROR: Career '{career_name}' not found in the dataset.")
        return

    # 3. Identify matching nodes
    target_nodes = set(required_skills)
    user_nodes = set(user_skills + user_interests)
    matching_nodes = target_nodes.intersection(user_nodes)
    
    # 4. Permanently increase the weights in the core graph (G_CORE)
    for node in matching_nodes:
        if G_CORE.has_edge(career_name, node):
            current_weight = G_CORE[career_name][node].get("weight", 3.0)
            G_CORE[career_name][node]["weight"] = current_weight + 0.5
            
            print(f"Graph Update: Strengthened edge between {career_name} and {node}.")
        
# --- RECOMMENDATION MODEL 1: GRAPH ---

def recommend_graph_model(user_name: str, user_skills_str: str, user_interests_str: str, top_k=5):
    """Recommends careers using the NetworkX Graph Model."""
    if DF_GRAPH.empty:
        return []

    # Prepare inputs
    user_skills = [s.strip().lower() for s in user_skills_str.split(",") if s.strip()]
    user_interests = [i.strip().lower() for i in user_interests_str.split(",") if i.strip()]

    # Add temporary user nodes
    G_CORE.add_node(user_name, type="user")
    for s in user_skills:
        if s not in G_CORE: G_CORE.add_node(s, type="skill")
        G_CORE.add_edge(user_name, s, weight=3)
    for i in user_interests:
        if i not in G_CORE: G_CORE.add_node(i, type="interest")
        G_CORE.add_edge(user_name, i, weight=2)

    # Compute Scores (using common neighbors)
    scores = []
    for _, row in DF_GRAPH.iterrows():
        career = row["career_name"]
        score = sum(G_CORE[user_name][node]["weight"] + G_CORE[career][node]["weight"]
                    for node in nx.common_neighbors(G_CORE, user_name, career)
                    if G_CORE.has_edge(user_name, node) and G_CORE.has_edge(career, node))
        scores.append((career, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Remove user node (Cleanup)
    G_CORE.remove_node(user_name)
    
    # Format Results
    final_results = []
    for career, score in scores[:top_k]:
        row = DF_GRAPH[DF_GRAPH["career_name"] == career].iloc[0]
        final_results.append({
            "career_name": career,
            "score": round(score, 2), 
            "required_skills": row["required_skills"][:5], 
            "recommended_courses": row["recommended_courses"][:3], 
            "roadmap": row["roadmap"]
        })
    return final_results

# --- RECOMMENDATION MODEL 2: SVD ---

def recommend_svd_model(user_data, top_k=5):
    """Recommends careers using the SVD model (TF-IDF + TruncatedSVD)."""
    if setup.careers_df.empty or setup.SVD_MODEL is None or setup.tfidf is None:
        return []
    
    user_text = combine_user_input_ml(user_data)
    
    user_tfidf = setup.tfidf.transform([user_text])
    user_svd = setup.SVD_MODEL.transform(user_tfidf)

    similarities = cosine_similarity(user_svd, setup.career_svd_matrix)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    recommendations = []
    for idx in top_indices:
        career = setup.careers_data[idx]
        recommendations.append({
            'career_name': career['career_name'],
            'score': float(similarities[idx]), 
            'required_skills': career.get('required_skills', [])[:5],
            'recommended_courses': career.get('recommended_courses', [])[:3],
            'roadmap': career.get('roadmap', 'N/A')
        })

    return recommendations

# --- RECOMMENDATION MODEL 3: K-MEANS ---

def recommend_kmeans_model(user_data, top_k=5):
    """Recommends careers using the K-Means model (Clustering + Similarity)."""
    if setup.careers_df.empty or setup.kmeans is None or setup.tfidf is None:
        return []

    user_text = combine_user_input_ml(user_data)
    user_vector = setup.tfidf.transform([user_text])

    user_cluster = setup.kmeans.predict(user_vector)[0]

    cluster_careers = setup.careers_df[setup.careers_df['cluster'] == user_cluster].copy()

    cluster_indices = cluster_careers.index.tolist()
    
    if not cluster_indices:
        return []

    cluster_tfidf = setup.tfidf_matrix[cluster_indices]

    similarities = cosine_similarity(user_vector, cluster_tfidf)[0]

    top_local_indices = similarities.argsort()[-top_k:][::-1]
    top_global_indices = [cluster_indices[i] for i in top_local_indices]

    recommendations = []
    for idx, local_idx in zip(top_global_indices, top_local_indices):
        career = setup.careers_data[idx]
        recommendations.append({
            'career_name': career['career_name'],
            'score': float(similarities[local_idx]),
            'required_skills': career.get('required_skills', [])[:5],
            'recommended_courses': career.get('recommended_courses', [])[:3],
            'roadmap': career.get('roadmap', 'N/A')
        })

    return recommendations