"""
Career Match Recommendation System
===================================
This module provides career-oriented content recommendations based on:
1. User skill mastery profiles
2. Career goals
3. Historical interaction patterns
4. Content-based filtering (embeddings)
5. Collaborative filtering (SVD/Matrix Factorization)

No LLM inference required - uses traditional ML algorithms.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class CareerMatchRecommender:
    """
    Hybrid recommender system for career-focused content recommendations.
    Combines content-based filtering, collaborative filtering, and career alignment.
    """
    
    def __init__(self, 
                 data_path: str = "../database/",
                 n_factors: int = 50,
                 alpha_content: float = 0.3,
                 alpha_collab: float = 0.3,
                 alpha_career: float = 0.4):
        """
        Initialize the recommender system.
        
        Args:
            data_path: Path to database directory
            n_factors: Number of latent factors for SVD
            alpha_content: Weight for content-based filtering
            alpha_collab: Weight for collaborative filtering
            alpha_career: Weight for career alignment
        """
        self.data_path = data_path
        self.n_factors = n_factors
        self.alpha_content = alpha_content
        self.alpha_collab = alpha_collab
        self.alpha_career = alpha_career
        
        # Data containers
        self.users = None
        self.students_profile = None
        self.content = None
        self.interactions = None
        self.recommendation_logs = None
        
        # Computed matrices
        self.content_embeddings = None
        self.user_item_matrix = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        self.career_skill_mapping = None
        
        # Mappings
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        
    def load_data(self):
        """Load all necessary data files."""
        print("Loading data...")
        self.users = pd.read_csv(f"{self.data_path}users.csv")
        self.students_profile = pd.read_csv(f"{self.data_path}students_profile.csv")
        self.content = pd.read_csv(f"{self.data_path}content_items.csv")
        self.interactions = pd.read_csv(f"{self.data_path}interactions.csv")
        self.recommendation_logs = pd.read_csv(f"{self.data_path}recommendation_logs.csv")
        
        print(f"Loaded {len(self.users)} users, {len(self.content)} content items, "
              f"{len(self.interactions)} interactions")
    
    def preprocess_embeddings(self):
        """Extract and process content embeddings."""
        print("Processing content embeddings...")
        
        def parse_embedding(x):
            if pd.isna(x):
                return None
            if isinstance(x, (list, np.ndarray)):
                return np.array(x, dtype=float)
            try:
                return np.array(json.loads(x), dtype=float)
            except Exception:
                return None
        
        self.content['embed'] = self.content.get('embedding_vector', None).apply(parse_embedding)
        
        # Check if embeddings are available
        if 'embed' not in self.content.columns or self.content['embed'].isna().sum() > len(self.content) * 0.4:
            print("Building skill-based vectors from tags...")
            self.content['skills_tags'] = self.content.get('skills_tags', "").fillna("").astype(str)
            
            # Parse skills as lists
            tag_lists = self.content['skills_tags'].apply(
                lambda s: [t.strip().lower() for t in str(s).replace("[", "").replace("]", "").replace("'", "").split(",") if t.strip()]
            )
            
            # Build vocabulary
            all_tags = sorted({t for tags in tag_lists for t in tags})
            tag_to_idx = {t: i for i, t in enumerate(all_tags)}
            
            def tags_to_vec(tags):
                vec = np.zeros(len(all_tags), dtype=float)
                for t in tags:
                    if t in tag_to_idx:
                        vec[tag_to_idx[t]] = 1.0
                return vec
            
            self.content['embed_vec'] = tag_lists.apply(tags_to_vec)
        else:
            # Use existing embeddings
            dims = max([len(e) for e in self.content['embed'].dropna()]) if len(self.content['embed'].dropna()) > 0 else 256
            self.content['embed_vec'] = self.content['embed'].apply(
                lambda e: np.zeros(dims) if e is None else e
            )
        
        # Stack into matrix and normalize
        self.content_embeddings = normalize(np.vstack(self.content['embed_vec'].values))
        
        # Create item mapping
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.content['id'].values)}
        self.idx_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_idx.items()}
        
        print(f"Content embeddings shape: {self.content_embeddings.shape}")
    
    def build_user_item_matrix(self):
        """Build user-item interaction matrix for collaborative filtering."""
        print("Building user-item interaction matrix...")
        
        # Create user mapping
        unique_users = sorted(self.interactions['user_id'].unique())
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        
        # Build sparse matrix
        rows = []
        cols = []
        data = []
        
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            
            if user_id in self.user_id_to_idx and item_id in self.item_id_to_idx:
                user_idx = self.user_id_to_idx[user_id]
                item_idx = self.item_id_to_idx[item_id]
                
                # Compute interaction score
                score = 0
                if row['action'] == 'view':
                    score = 1.0
                elif row['action'] == 'attempt':
                    score = 2.0
                elif row['action'] == 'complete':
                    score = 3.0
                    if row.get('correct', 0) == 1:
                        score = 4.0
                
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(score)
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_id_to_idx), len(self.item_id_to_idx))
        )
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4f}")
    
    def train_collaborative_filter(self):
        """Train SVD-based collaborative filtering model."""
        print("Training collaborative filtering model...")
        
        # Apply SVD
        n_components = min(self.n_factors, min(self.user_item_matrix.shape) - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T
        
        print(f"SVD factors: {self.user_factors.shape}, {self.item_factors.shape}")
    
    def build_career_skill_mapping(self):
        """Create career to required skills mapping."""
        print("Building career-skill mapping...")
        
        # Define skill requirements for each career
        self.career_skill_mapping = {
            'Software Engineer': {
                'Programming': 0.9, 'Mathematics': 0.7, 'Aptitude': 0.6,
                'Communication': 0.5, 'Physics': 0.3
            },
            'Data Scientist': {
                'Mathematics': 0.9, 'Programming': 0.8, 'Aptitude': 0.7,
                'Communication': 0.5, 'Physics': 0.4
            },
            'AI Engineer': {
                'Programming': 0.9, 'Mathematics': 0.9, 'Physics': 0.6,
                'Aptitude': 0.7, 'Communication': 0.5
            },
            'Doctor': {
                'Biology': 0.9, 'Chemistry': 0.8, 'Communication': 0.7,
                'Aptitude': 0.6, 'Physics': 0.4
            },
            'Chemical Engineer': {
                'Chemistry': 0.9, 'Physics': 0.7, 'Mathematics': 0.7,
                'Aptitude': 0.5, 'Programming': 0.4
            },
            'Civil Engineer': {
                'Physics': 0.8, 'Mathematics': 0.8, 'Aptitude': 0.6,
                'Communication': 0.5, 'Programming': 0.3
            },
            'Mechanical Engineer': {
                'Physics': 0.9, 'Mathematics': 0.8, 'Aptitude': 0.6,
                'Programming': 0.4, 'Communication': 0.5
            },
            'Researcher': {
                'Mathematics': 0.8, 'Physics': 0.7, 'Chemistry': 0.7,
                'Biology': 0.7, 'Communication': 0.6, 'Aptitude': 0.6
            },
            'Cybersecurity Analyst': {
                'Programming': 0.8, 'Aptitude': 0.8, 'Mathematics': 0.6,
                'Communication': 0.5, 'Physics': 0.3
            },
            'Entrepreneur': {
                'Communication': 0.9, 'Aptitude': 0.8, 'Mathematics': 0.5,
                'Programming': 0.4
            }
        }
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get complete user profile including skills and career goal."""
        user_info = self.users[self.users['id'] == user_id].iloc[0].to_dict()
        
        # Get skill mastery
        profile_row = self.students_profile[self.students_profile['user_id'] == user_id]
        if len(profile_row) > 0:
            skills_str = profile_row.iloc[0]['skills_mastery']
            skills = json.loads(skills_str.replace("'", '"'))
        else:
            skills = {}
        
        return {
            'user_id': user_id,
            'goal': user_info.get('goal', 'Unknown'),
            'grade': user_info.get('grade', 'Unknown'),
            'skills': skills,
            'preferences': user_info.get('preferences', '[]')
        }
    
    def compute_career_alignment_score(self, user_profile: Dict, content_item: pd.Series) -> float:
        """
        Compute how well a content item aligns with user's career goal.
        
        Args:
            user_profile: User profile dict with skills and career goal
            content_item: Content item row from dataframe
            
        Returns:
            Career alignment score [0, 1]
        """
        career_goal = user_profile['goal']
        user_skills = user_profile['skills']
        
        if career_goal not in self.career_skill_mapping:
            return 0.5  # Default score for unknown careers
        
        required_skills = self.career_skill_mapping[career_goal]
        
        # Extract content skills
        content_skills_str = str(content_item.get('skills_tags', '[]'))
        content_skills = [
            s.strip().lower() for s in content_skills_str.replace("[", "").replace("]", "").replace("'", "").split(",")
            if s.strip()
        ]
        
        # Compute alignment score
        alignment_score = 0.0
        skill_count = 0
        
        for skill in content_skills:
            # Normalize skill names
            skill_normalized = skill.capitalize()
            
            if skill_normalized in required_skills:
                # Weight by career requirement
                career_weight = required_skills[skill_normalized]
                
                # Check user's current mastery
                user_mastery = user_skills.get(skill_normalized, 0.0)
                
                # Prioritize skills with gap (needed but not mastered)
                skill_gap = max(0, career_weight - user_mastery)
                
                # Score combines relevance and learning need
                alignment_score += career_weight * (0.5 + 0.5 * skill_gap)
                skill_count += 1
        
        # Normalize
        if skill_count > 0:
            alignment_score /= skill_count
        
        # Boost for difficulty match
        content_difficulty = content_item.get('difficulty', 2)
        user_avg_mastery = np.mean(list(user_skills.values())) if user_skills else 0.5
        
        # Prefer content slightly above current level
        difficulty_match = 1.0 - abs((content_difficulty / 3.0) - user_avg_mastery) / 2.0
        alignment_score = 0.7 * alignment_score + 0.3 * difficulty_match
        
        return np.clip(alignment_score, 0, 1)
    
    def compute_content_based_score(self, user_id: str, item_idx: int) -> float:
        """
        Compute content-based similarity score using embeddings.
        
        Args:
            user_id: User ID
            item_idx: Content item index
            
        Returns:
            Content similarity score
        """
        # Get user's historical interactions
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return 0.5  # No history, return neutral score
        
        # Get items user has interacted with positively
        positive_items = user_interactions[
            (user_interactions['action'].isin(['complete', 'attempt'])) |
            ((user_interactions['action'] == 'view') & (user_interactions.get('correct', 0) == 1))
        ]['item_id'].values
        
        if len(positive_items) == 0:
            return 0.5
        
        # Get embeddings of positive items
        positive_indices = [self.item_id_to_idx[item_id] for item_id in positive_items if item_id in self.item_id_to_idx]
        
        if len(positive_indices) == 0:
            return 0.5
        
        positive_embeddings = self.content_embeddings[positive_indices]
        
        # Compute average user preference vector
        user_preference_vector = np.mean(positive_embeddings, axis=0).reshape(1, -1)
        
        # Compute similarity with target item
        target_embedding = self.content_embeddings[item_idx].reshape(1, -1)
        similarity = cosine_similarity(user_preference_vector, target_embedding)[0, 0]
        
        # Normalize to [0, 1]
        score = (similarity + 1) / 2
        
        return score
    
    def compute_collaborative_score(self, user_id: str, item_idx: int) -> float:
        """
        Compute collaborative filtering score using SVD.
        
        Args:
            user_id: User ID
            item_idx: Content item index
            
        Returns:
            Predicted rating score
        """
        if user_id not in self.user_id_to_idx:
            return 0.5  # New user, return neutral score
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Predict rating using SVD factors
        predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Normalize to [0, 1]
        # Assuming ratings are in range [0, 4] based on our scoring
        score = np.clip(predicted_rating / 4.0, 0, 1)
        
        return score
    
    def recommend_for_career(self, 
                            user_id: str, 
                            n_recommendations: int = 10,
                            exclude_seen: bool = True) -> List[Dict]:
        """
        Generate career-aligned content recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            exclude_seen: Whether to exclude already interacted items
            
        Returns:
            List of recommended items with scores
        """
        # Get user profile
        user_profile = self.get_user_profile(user_id)
        
        # Get items to consider
        candidate_items = self.content['id'].values
        
        if exclude_seen:
            # Exclude items user has already interacted with
            seen_items = set(self.interactions[self.interactions['user_id'] == user_id]['item_id'].values)
            candidate_items = [item for item in candidate_items if item not in seen_items]
        
        # Compute scores for each candidate
        recommendations = []
        
        for item_id in candidate_items:
            if item_id not in self.item_id_to_idx:
                continue
                
            item_idx = self.item_id_to_idx[item_id]
            content_item = self.content[self.content['id'] == item_id].iloc[0]
            
            # Compute individual scores
            career_score = self.compute_career_alignment_score(user_profile, content_item)
            content_score = self.compute_content_based_score(user_id, item_idx)
            collab_score = self.compute_collaborative_score(user_id, item_idx)
            
            # Combine scores using weighted average
            final_score = (
                self.alpha_career * career_score +
                self.alpha_content * content_score +
                self.alpha_collab * collab_score
            )
            
            recommendations.append({
                'item_id': item_id,
                'title': content_item['title'],
                'type': content_item['type'],
                'difficulty': content_item['difficulty'],
                'skills': content_item['skills_tags'],
                'final_score': final_score,
                'career_score': career_score,
                'content_score': content_score,
                'collab_score': collab_score,
                'url': content_item.get('url', ''),
                'text': content_item.get('text', '')
            })
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def batch_recommend(self, user_ids: List[str], n_recommendations: int = 10) -> Dict[str, List[Dict]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to recommendations
        """
        results = {}
        for user_id in user_ids:
            try:
                results[user_id] = self.recommend_for_career(user_id, n_recommendations)
            except Exception as e:
                print(f"Error recommending for {user_id}: {e}")
                results[user_id] = []
        return results
    
    def fit(self):
        """Train the complete recommendation system."""
        print("=" * 60)
        print("Career Match Recommender - Training Pipeline")
        print("=" * 60)
        
        self.load_data()
        self.preprocess_embeddings()
        self.build_user_item_matrix()
        self.train_collaborative_filter()
        self.build_career_skill_mapping()
        
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)
    
    def evaluate_recommendations(self, user_id: str, recommendations: List[Dict]) -> Dict:
        """
        Evaluate recommendation quality for a user.
        
        Args:
            user_id: User ID
            recommendations: List of recommendations
            
        Returns:
            Evaluation metrics
        """
        user_profile = self.get_user_profile(user_id)
        
        metrics = {
            'user_id': user_id,
            'career_goal': user_profile['goal'],
            'avg_career_alignment': np.mean([r['career_score'] for r in recommendations]),
            'avg_content_similarity': np.mean([r['content_score'] for r in recommendations]),
            'avg_collab_score': np.mean([r['collab_score'] for r in recommendations]),
            'avg_final_score': np.mean([r['final_score'] for r in recommendations]),
            'diversity': len(set([r['type'] for r in recommendations])) / len(recommendations),
            'difficulty_distribution': {
                'level_1': sum(1 for r in recommendations if r['difficulty'] == 1),
                'level_2': sum(1 for r in recommendations if r['difficulty'] == 2),
                'level_3': sum(1 for r in recommendations if r['difficulty'] == 3)
            }
        }
        
        return metrics


def main():
    """Example usage of the career match recommender."""
    # Initialize recommender
    recommender = CareerMatchRecommender(
        data_path="../database/",
        n_factors=50,
        alpha_content=0.3,
        alpha_collab=0.3,
        alpha_career=0.4
    )
    
    # Train the model
    recommender.fit()
    
    # Generate recommendations for sample users
    sample_users = ['user_1', 'user_2', 'user_3', 'user_10', 'user_20']
    
    for user_id in sample_users:
        print(f"\n{'=' * 80}")
        print(f"Recommendations for {user_id}")
        print('=' * 80)
        
        user_profile = recommender.get_user_profile(user_id)
        print(f"\nUser Profile:")
        print(f"  Career Goal: {user_profile['goal']}")
        print(f"  Grade: {user_profile['grade']}")
        print(f"  Top Skills: {sorted(user_profile['skills'].items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        recommendations = recommender.recommend_for_career(user_id, n_recommendations=5)
        
        print(f"\nTop 5 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Type: {rec['type']} | Difficulty: {rec['difficulty']}")
            print(f"   Skills: {rec['skills']}")
            print(f"   Scores - Career: {rec['career_score']:.3f} | "
                  f"Content: {rec['content_score']:.3f} | "
                  f"Collab: {rec['collab_score']:.3f} | "
                  f"Final: {rec['final_score']:.3f}")
        
        # Evaluate
        metrics = recommender.evaluate_recommendations(user_id, recommendations)
        print(f"\nEvaluation Metrics:")
        print(f"  Avg Career Alignment: {metrics['avg_career_alignment']:.3f}")
        print(f"  Content Diversity: {metrics['diversity']:.3f}")
        print(f"  Difficulty Distribution: {metrics['difficulty_distribution']}")


if __name__ == "__main__":
    main()
