"""
Simple API wrapper for Career Match Recommender
Provides easy-to-use functions for integration with chatbot or web services.
"""

import json
from typing import List, Dict, Optional
from career_match_recommender import CareerMatchRecommender


class RecommendationAPI:
    """
    Simple API wrapper for the recommendation system.
    """
    
    def __init__(self, data_path: str = "../database/"):
        """Initialize and train the recommender."""
        self.recommender = CareerMatchRecommender(
            data_path=data_path,
            n_factors=50,
            alpha_content=0.3,
            alpha_collab=0.3,
            alpha_career=0.4
        )
        self.recommender.fit()
        self.is_trained = True
    
    def get_recommendations(self, 
                           user_id: str, 
                           n: int = 10,
                           exclude_seen: bool = True,
                           return_format: str = 'dict') -> List[Dict]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude already seen items
            return_format: 'dict' or 'simple'
            
        Returns:
            List of recommendations
        """
        recs = self.recommender.recommend_for_career(user_id, n, exclude_seen)
        
        if return_format == 'simple':
            # Return simplified format
            return [
                {
                    'item_id': r['item_id'],
                    'title': r['title'],
                    'type': r['type'],
                    'difficulty': r['difficulty'],
                    'score': round(r['final_score'], 3),
                    'url': r['url']
                }
                for r in recs
            ]
        
        return recs
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile information."""
        return self.recommender.get_user_profile(user_id)
    
    def get_skill_gaps(self, user_id: str) -> Dict:
        """
        Get skill gaps for a user based on their career goal.
        
        Returns:
            Dictionary with skill gaps and recommendations
        """
        profile = self.recommender.get_user_profile(user_id)
        career_goal = profile['goal']
        user_skills = profile['skills']
        
        if career_goal not in self.recommender.career_skill_mapping:
            return {'error': f'Unknown career goal: {career_goal}'}
        
        required_skills = self.recommender.career_skill_mapping[career_goal]
        
        skill_gaps = {}
        for skill, required in required_skills.items():
            current = user_skills.get(skill, 0.0)
            gap = max(0, required - current)
            skill_gaps[skill] = {
                'current': round(current, 3),
                'required': round(required, 3),
                'gap': round(gap, 3),
                'status': 'proficient' if gap == 0 else ('needs_improvement' if gap > 0.3 else 'developing')
            }
        
        return {
            'user_id': user_id,
            'career_goal': career_goal,
            'skill_gaps': skill_gaps,
            'priority_skills': sorted(
                skill_gaps.items(),
                key=lambda x: x[1]['gap'],
                reverse=True
            )[:3]
        }
    
    def get_recommendations_by_skill(self, 
                                    user_id: str,
                                    skill: str,
                                    n: int = 5) -> List[Dict]:
        """
        Get recommendations focused on a specific skill.
        
        Args:
            user_id: User ID
            skill: Skill to focus on
            n: Number of recommendations
            
        Returns:
            List of skill-focused recommendations
        """
        all_recs = self.recommender.recommend_for_career(user_id, n_recommendations=50)
        
        # Filter recommendations that include the target skill
        skill_recs = [
            r for r in all_recs
            if skill.lower() in str(r['skills']).lower()
        ]
        
        return skill_recs[:n]
    
    def get_career_suggestions(self, user_id: str) -> Dict:
        """
        Suggest alternative careers based on user's skill profile.
        
        Args:
            user_id: User ID
            
        Returns:
            Career suggestions with match scores
        """
        profile = self.recommender.get_user_profile(user_id)
        user_skills = profile['skills']
        
        career_matches = []
        
        for career, required_skills in self.recommender.career_skill_mapping.items():
            # Calculate match score
            match_score = 0
            skill_count = 0
            
            for skill, required in required_skills.items():
                current = user_skills.get(skill, 0.0)
                # Score based on how close current is to required
                skill_match = 1.0 - min(abs(current - required) / required, 1.0) if required > 0 else 0
                match_score += skill_match
                skill_count += 1
            
            avg_match = match_score / skill_count if skill_count > 0 else 0
            
            career_matches.append({
                'career': career,
                'match_score': round(avg_match, 3),
                'is_current_goal': career == profile['goal']
            })
        
        # Sort by match score
        career_matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {
            'user_id': user_id,
            'current_goal': profile['goal'],
            'career_matches': career_matches[:5],
            'best_alternative': career_matches[1] if len(career_matches) > 1 else None
        }
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict:
        """
        Explain why an item was recommended to a user.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Explanation dictionary
        """
        profile = self.recommender.get_user_profile(user_id)
        content_item = self.recommender.content[self.recommender.content['id'] == item_id].iloc[0]
        item_idx = self.recommender.item_id_to_idx[item_id]
        
        # Compute individual scores
        career_score = self.recommender.compute_career_alignment_score(profile, content_item)
        content_score = self.recommender.compute_content_based_score(user_id, item_idx)
        collab_score = self.recommender.compute_collaborative_score(user_id, item_idx)
        
        final_score = (
            self.recommender.alpha_career * career_score +
            self.recommender.alpha_content * content_score +
            self.recommender.alpha_collab * collab_score
        )
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'item_title': content_item['title'],
            'career_goal': profile['goal'],
            'scores': {
                'career_alignment': round(career_score, 3),
                'content_similarity': round(content_score, 3),
                'collaborative': round(collab_score, 3),
                'final_score': round(final_score, 3)
            },
            'explanation': self._generate_explanation(
                profile, content_item, career_score, content_score, collab_score
            )
        }
    
    def _generate_explanation(self, profile, content_item, career_score, content_score, collab_score):
        """Generate human-readable explanation."""
        explanations = []
        
        # Career alignment
        if career_score > 0.7:
            explanations.append(
                f"This content strongly aligns with your career goal of becoming a {profile['goal']}."
            )
        elif career_score > 0.5:
            explanations.append(
                f"This content is relevant to your career goal of {profile['goal']}."
            )
        
        # Content similarity
        if content_score > 0.6:
            explanations.append(
                "This content is similar to other materials you've engaged with successfully."
            )
        
        # Collaborative
        if collab_score > 0.6:
            explanations.append(
                "Students with similar profiles have found this content helpful."
            )
        
        # Difficulty
        difficulty = content_item.get('difficulty', 2)
        user_avg_mastery = sum(profile['skills'].values()) / len(profile['skills']) if profile['skills'] else 0.5
        
        if abs((difficulty / 3.0) - user_avg_mastery) < 0.2:
            explanations.append(
                f"The difficulty level ({difficulty}/3) matches well with your current skill level."
            )
        
        return " ".join(explanations)
    
    def bulk_recommend(self, user_ids: List[str], n: int = 5) -> Dict[str, List[Dict]]:
        """Generate recommendations for multiple users."""
        return self.recommender.batch_recommend(user_ids, n)


# Example usage functions
def demo_api(user):
    """Demo the API functionality."""
    print("Initializing Career Match Recommendation API...")
    api = RecommendationAPI(data_path="../database/")
    
    print("\n" + "="*80)
    print("Example 1: Get Recommendations for a User")
    print("="*80)
    user_id = user
    recs = api.get_recommendations(user_id, n=5, return_format='simple')
    print(f"\nTop 5 recommendations for {user_id}:")
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['score']})")
    
    print("\n" + "="*80)
    print("Example 2: Get Skill Gaps")
    print("="*80)
    skill_gaps = api.get_skill_gaps(user_id)
    print(f"\nSkill gaps for {user_id} (Goal: {skill_gaps['career_goal']}):")
    print("\nTop 3 priority skills:")
    for skill, info in skill_gaps['priority_skills']:
        print(f"  {skill}: Gap = {info['gap']} ({info['status']})")
    
    print("\n" + "="*80)
    print("Example 3: Get Skill-Focused Recommendations")
    print("="*80)
    skill = "Programming"
    skill_recs = api.get_recommendations_by_skill(user_id, skill, n=3)
    print(f"\nTop 3 {skill} recommendations for {user_id}:")
    for i, rec in enumerate(skill_recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['final_score']:.3f})")
    
    print("\n" + "="*80)
    print("Example 4: Get Career Suggestions")
    print("="*80)
    career_suggestions = api.get_career_suggestions(user_id)
    print(f"\nCareer matches for {user_id}:")
    print(f"Current goal: {career_suggestions['current_goal']}")
    print("\nTop 5 career matches:")
    for career in career_suggestions['career_matches']:
        marker = "✓" if career['is_current_goal'] else " "
        print(f"  {marker} {career['career']}: {career['match_score']:.3f}")
    
    print("\n" + "="*80)
    print("Example 5: Explain a Recommendation")
    print("="*80)
    item_id = recs[0]['item_id']
    explanation = api.explain_recommendation(user_id, item_id)
    print(f"\nWhy was '{explanation['item_title']}' recommended?")
    print(f"Career Goal: {explanation['career_goal']}")
    print(f"\nScores:")
    for score_type, score in explanation['scores'].items():
        print(f"  {score_type}: {score}")
    print(f"\nExplanation: {explanation['explanation']}")


if __name__ == "__main__":
    demo_api("user_2")
