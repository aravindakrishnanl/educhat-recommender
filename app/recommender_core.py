import pandas as pd
import networkx as nx
import json
from pathlib import Path
import os 

# Define BASE_DIR (two levels up from recommender_core.py, pointing to project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Global Setup (Run once when the server starts) ---
try:
    # FIX: Use absolute path and correct filename
    json_path = os.path.join(BASE_DIR, "careers_dataset_50plus.json")
    df = pd.read_json(json_path) 
except FileNotFoundError:
    print("Error: careers_dataset_50plus.json not found at expected path. Graph will be empty.")
    df = pd.DataFrame() 

def fix_list(x):
    # Ensures list handling is robust
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return [s.strip() for s in str(x).split(",")]


if not df.empty:
    df["required_skills"] = df["required_skills"].apply(fix_list)
    df["recommended_courses"] = df["recommended_courses"].apply(fix_list)
    df["projects"] = df["projects"].apply(fix_list)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["career_name"], type="career")
        for skill in row["required_skills"]:
            # CRITICAL: Always convert dataset requirements to lowercase for consistent nodes
            G.add_node(skill.lower(), type="skill") 
            G.add_edge(row["career_name"], skill.lower(), weight=3) 
        for course in row["recommended_courses"]:
            G.add_node(course.lower(), type="course")
            G.add_edge(row["career_name"], course.lower(), weight=1)
        for proj in row["projects"]:
            G.add_node(proj.lower(), type="project")
            G.add_edge(row["career_name"], proj.lower(), weight=1)
else:
    G = nx.Graph() 

# The core recommendation function
def generate_recommendations(user_name: str, user_skills_str: str, user_interests_str: str, top_k=5):
    """
    Predicts careers based on user input using the pre-built graph G.
    Returns a list of dicts suitable for JSON serialization.
    """
    if df.empty:
        return []

    # CRITICAL: Ensures user input is consistently lowercase and stripped
    user_skills = [s.strip().lower() for s in user_skills_str.split(",") if s.strip()]
    user_interests = [i.strip().lower() for i in user_interests_str.split(",") if i.strip()]

    # 1. Add the temporary user node for this request
    user_skills_nodes = set()
    user_interest_nodes = set()
    G.add_node(user_name, type="user")
    
    for s in user_skills:
        # Only add the node if it doesn't already exist from the dataset build
        if s not in G: G.add_node(s, type="skill")
        G.add_edge(user_name, s, weight=3)
        user_skills_nodes.add(s)

    for i in user_interests:
        if i not in G: G.add_node(i, type="interest")
        G.add_edge(user_name, i, weight=2)
        user_interest_nodes.add(i)

    # 2. Compute Scores
    scores = []
    for _, row in df.iterrows():
        career = row["career_name"]
        
        score = 0
        
        # Check against Skills and Interests nodes supplied by the user
        for node in user_skills_nodes | user_interest_nodes:
            if G.has_edge(career, node):
                # Ensure edges exist before accessing weights (safety check)
                if G.has_edge(user_name, node) and G.has_edge(career, node):
                    edge_user = G[user_name][node]["weight"]
                    edge_career = G[career][node]["weight"]
                    score += edge_user + edge_career
        
        scores.append((career, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # 3. Format Results
    final_results = []
    for career, score in scores[:top_k]:
        row = df[df["career_name"] == career].iloc[0]
        final_results.append({
            "career_name": career,
            "score": round(score, 2), 
            "required_skills": row["required_skills"][:5], 
            "recommended_courses": row["recommended_courses"][:3], 
            "roadmap": row["roadmap"]
        })
        
    # 4. Remove User Node and Edges (Cleanup for next request)
    G.remove_node(user_name)
    
    return final_results