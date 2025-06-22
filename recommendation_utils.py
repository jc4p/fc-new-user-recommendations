"""
Utilities for content recommendation based on cluster analysis
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Set
import random

class ClusterRecommender:
    def __init__(self, clusters_path: str = 'data/quality_clusters.json'):
        """Initialize recommender with cluster data"""
        with open(clusters_path, 'r') as f:
            self.clusters = json.load(f)
        
        # Initialize cluster weights (all equal at start)
        self.cluster_weights = {c['cluster_id']: 1.0 for c in self.clusters}
        
        # Track which posts we've shown
        self.shown_posts = set()
        
        # Track user preferences
        self.liked_clusters = []
        self.disliked_clusters = []
        
    def get_cluster_diversity_score(self, cluster_ids: List[int]) -> float:
        """Calculate how diverse a set of clusters is by theme"""
        themes = [self.get_cluster_by_id(cid)['name'] for cid in cluster_ids]
        unique_themes = len(set(themes))
        return unique_themes / len(themes) if themes else 0
    
    def get_cluster_by_id(self, cluster_id: int) -> Dict:
        """Get cluster data by ID"""
        for cluster in self.clusters:
            if cluster['cluster_id'] == cluster_id:
                return cluster
        return None
    
    def select_next_post(self, num_questions_asked: int) -> Tuple[Dict, int]:
        """
        Select the next post to show the user that maximizes information gain
        Returns: (post_data, cluster_id)
        """
        # Get clusters sorted by weight (excluding very low weights)
        active_clusters = [(cid, w) for cid, w in self.cluster_weights.items() if w > 0.1]
        active_clusters.sort(key=lambda x: x[1], reverse=True)
        
        # Strategy varies by question number
        if num_questions_asked == 0:
            # First question: pick from the largest, highest-quality cluster
            target_cluster = max(self.clusters, key=lambda c: c['size'] * c['avg_quality_score'])
        
        elif num_questions_asked < 3:
            # Early questions: maximize theme diversity
            shown_themes = set()
            for cid in self.liked_clusters + self.disliked_clusters:
                cluster = self.get_cluster_by_id(cid)
                if cluster:
                    shown_themes.add(cluster['name'])
            
            # Find clusters with new themes
            candidates = []
            for cid, weight in active_clusters[:10]:  # Top 10 by weight
                cluster = self.get_cluster_by_id(cid)
                if cluster and cluster['name'] not in shown_themes:
                    candidates.append(cluster)
            
            if not candidates:
                # All themes shown, pick from highest weighted
                candidates = [self.get_cluster_by_id(cid) for cid, _ in active_clusters[:5]]
            
            target_cluster = random.choice(candidates) if candidates else self.clusters[0]
        
        else:
            # Later questions: focus on highest-weighted clusters
            top_cluster_ids = [cid for cid, _ in active_clusters[:5]]
            target_cluster = self.get_cluster_by_id(random.choice(top_cluster_ids))
        
        # Select a representative post from the cluster
        available_posts = [
            (post, i) for i, post in enumerate(target_cluster['representatives'])
            if f"{target_cluster['cluster_id']}_{i}" not in self.shown_posts
        ]
        
        if not available_posts:
            # All posts shown from this cluster, pick any unshown post
            for cluster in self.clusters:
                for i, post in enumerate(cluster['representatives']):
                    post_id = f"{cluster['cluster_id']}_{i}"
                    if post_id not in self.shown_posts:
                        self.shown_posts.add(post_id)
                        return post, cluster['cluster_id']
        
        # Prefer posts from high-quality authors with good engagement
        post, idx = max(available_posts, key=lambda x: 
                       x[0]['author_quality'] * (x[0]['quality_reactions'] + x[0]['quality_replies'] * 2))
        
        post_id = f"{target_cluster['cluster_id']}_{idx}"
        self.shown_posts.add(post_id)
        
        return post, target_cluster['cluster_id']
    
    def update_weights(self, cluster_id: int, liked: bool):
        """Update cluster weights based on user response"""
        target_cluster = self.get_cluster_by_id(cluster_id)
        
        if liked:
            self.liked_clusters.append(cluster_id)
            # Boost similar clusters
            boost_factor = 1.5
            reduction_factor = 0.8
        else:
            self.disliked_clusters.append(cluster_id)
            # Reduce similar clusters
            boost_factor = 0.5
            reduction_factor = 1.2
        
        # Update weights based on similarity
        target_theme = target_cluster['name']
        target_words = set(target_cluster['top_words'])
        
        for cluster in self.clusters:
            cid = cluster['cluster_id']
            
            # Same cluster gets strongest effect
            if cid == cluster_id:
                self.cluster_weights[cid] *= boost_factor
            
            # Similar theme gets medium effect
            elif cluster['name'] == target_theme:
                similarity = len(set(cluster['top_words']) & target_words) / 10
                factor = boost_factor if liked else reduction_factor
                self.cluster_weights[cid] *= (1 + (factor - 1) * similarity)
            
            # Different theme gets opposite effect (slight)
            else:
                factor = reduction_factor if liked else boost_factor
                self.cluster_weights[cid] *= (1 + (factor - 1) * 0.1)
        
        # Normalize weights to prevent explosion/vanishing
        max_weight = max(self.cluster_weights.values())
        if max_weight > 10:
            for cid in self.cluster_weights:
                self.cluster_weights[cid] /= max_weight / 10
    
    def get_recommendations(self, n: int = 10) -> List[Dict]:
        """Get final recommendations based on learned preferences"""
        # Sort clusters by final weight
        sorted_clusters = sorted(
            self.cluster_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        recommendations = []
        used_clusters = set()
        
        # Get posts from top clusters
        for cluster_id, weight in sorted_clusters:
            if len(recommendations) >= n:
                break
            
            if weight < 0.5:  # Skip very low weight clusters
                continue
            
            cluster = self.get_cluster_by_id(cluster_id)
            if not cluster:
                continue
            
            # Add best posts from this cluster
            posts_to_add = min(3, n - len(recommendations))  # Max 3 per cluster
            
            # Sort posts by quality
            sorted_posts = sorted(
                cluster['representatives'],
                key=lambda p: p['author_quality'] * (p['quality_reactions'] + p['quality_replies'] * 2),
                reverse=True
            )
            
            for post in sorted_posts[:posts_to_add]:
                post_info = post.copy()
                post_info['cluster_id'] = cluster_id
                post_info['cluster_theme'] = cluster['name']
                post_info['cluster_keywords'] = cluster['top_words'][:5]
                recommendations.append(post_info)
            
            used_clusters.add(cluster_id)
        
        return recommendations
    
    def get_preference_summary(self) -> Dict:
        """Summarize user preferences based on liked/disliked clusters"""
        liked_themes = {}
        disliked_themes = {}
        
        # Count theme preferences
        for cid in self.liked_clusters:
            cluster = self.get_cluster_by_id(cid)
            if cluster:
                theme = cluster['name']
                liked_themes[theme] = liked_themes.get(theme, 0) + 1
        
        for cid in self.disliked_clusters:
            cluster = self.get_cluster_by_id(cid)
            if cluster:
                theme = cluster['name']
                disliked_themes[theme] = disliked_themes.get(theme, 0) + 1
        
        # Identify top preferences
        preferred_themes = []
        for theme, count in liked_themes.items():
            if count > disliked_themes.get(theme, 0):
                preferred_themes.append(theme)
        
        # Get common keywords from liked clusters
        liked_words = []
        for cid in self.liked_clusters:
            cluster = self.get_cluster_by_id(cid)
            if cluster:
                liked_words.extend(cluster['top_words'][:5])
        
        # Count word frequency
        word_counts = {}
        for word in liked_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'preferred_themes': preferred_themes,
            'top_keywords': [w for w, _ in top_keywords],
            'liked_clusters': self.liked_clusters,
            'disliked_clusters': self.disliked_clusters
        }
    
    def get_recommended_users(self, n: int = 10) -> List[Dict]:
        """Get recommended users based on learned preferences"""
        # Track author scores based on liked clusters
        author_scores = {}
        author_info = {}
        
        # Score authors from liked clusters higher
        for cid in self.liked_clusters:
            cluster = self.get_cluster_by_id(cid)
            if cluster:
                # Add points for authors in liked clusters
                for rep in cluster['representatives']:
                    fid = rep['fid']
                    if fid not in author_scores:
                        author_scores[fid] = 0
                        author_info[fid] = {
                            'fid': fid,
                            'quality_score': rep['author_quality'],
                            'sample_posts': [],
                            'cluster_themes': set()
                        }
                    
                    # Higher score for authors in liked clusters
                    author_scores[fid] += 2.0 * rep['author_quality']
                    author_info[fid]['sample_posts'].append(rep['text'][:150])
                    author_info[fid]['cluster_themes'].add(cluster['name'])
        
        # Penalize authors from disliked clusters
        for cid in self.disliked_clusters:
            cluster = self.get_cluster_by_id(cid)
            if cluster:
                for rep in cluster['representatives']:
                    fid = rep['fid']
                    if fid in author_scores:
                        author_scores[fid] *= 0.5  # Reduce score
        
        # Sort by score and get top N
        sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommended_users = []
        for fid, score in sorted_authors[:n]:
            info = author_info[fid]
            recommended_users.append({
                'fid': fid,
                'score': score,
                'quality_score': info['quality_score'],
                'themes': list(info['cluster_themes']),
                'sample_post': info['sample_posts'][0] if info['sample_posts'] else ''
            })
        
        return recommended_users