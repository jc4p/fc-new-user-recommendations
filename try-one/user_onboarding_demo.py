#!/usr/bin/env python3
"""
Demo: Binary search user onboarding using cluster representatives
Shows diverse posts to new users to quickly find their preferred niche
"""
import json
import random
import numpy as np

class UserOnboarding:
    def __init__(self, cluster_info_path='data/cluster_info.json'):
        """Initialize with cluster information"""
        with open(cluster_info_path, 'r') as f:
            self.clusters = json.load(f)
        
        # Create cluster mapping
        self.cluster_map = {c['cluster_id']: c for c in self.clusters}
        
        # Track user preferences
        self.user_preferences = []
        self.excluded_clusters = set()
        
    def get_diverse_clusters(self, n=3, exclude=None):
        """Get n diverse clusters for showing to user"""
        if exclude is None:
            exclude = self.excluded_clusters
        
        # Get available clusters
        available = [c for c in self.clusters 
                    if c['cluster_id'] not in exclude and c['size'] > 100]
        
        if len(available) < n:
            return available
        
        # Sort by different metrics to ensure diversity
        by_size = sorted(available, key=lambda x: x['size'], reverse=True)
        by_reactions = sorted(available, key=lambda x: x['avg_reactions'], reverse=True)
        by_replies = sorted(available, key=lambda x: x['avg_replies'], reverse=True)
        
        # Pick from different sorted lists
        selected = []
        if len(by_size) > 0:
            selected.append(by_size[0])
        if len(by_reactions) > 1 and by_reactions[1] not in selected:
            selected.append(by_reactions[1])
        if len(by_replies) > 2 and by_replies[2] not in selected:
            selected.append(by_replies[2])
        
        # Fill remaining with random choices
        remaining = [c for c in available if c not in selected]
        while len(selected) < n and remaining:
            choice = random.choice(remaining)
            selected.append(choice)
            remaining.remove(choice)
        
        return selected[:n]
    
    def show_posts_from_cluster(self, cluster, n_posts=2):
        """Show representative posts from a cluster"""
        posts = cluster['representative_posts'][:n_posts]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster['cluster_id']} (Size: {cluster['size']:,} posts)")
        print(f"Average engagement: {cluster['avg_reactions']:.0f} reactions, {cluster['avg_replies']:.0f} replies")
        print(f"{'='*80}")
        
        for i, post in enumerate(posts, 1):
            print(f"\nPost {i}:")
            print(f"Text: {post['text'][:200]}{'...' if len(post['text']) > 200 else ''}")
            print(f"Reactions: {post['reactions']}, Replies: {post['replies']}")
        
        return cluster['cluster_id']
    
    def get_user_preference(self, cluster_ids):
        """Get user's preference among shown clusters"""
        print("\n" + "="*80)
        print("Which type of content do you prefer?")
        print("Enter the cluster number you like most, or 'none' if none appeal to you:")
        print(f"Options: {', '.join(map(str, cluster_ids))} or 'none'")
        
        # Simulate user input (in real app, this would be actual input)
        # For demo, randomly select with bias towards certain clusters
        if random.random() < 0.2:
            return 'none'
        else:
            return random.choice(cluster_ids)
    
    def find_similar_clusters(self, preferred_cluster_id, n=5):
        """Find clusters similar to the preferred one"""
        preferred = self.cluster_map[preferred_cluster_id]
        
        # Simple similarity based on engagement patterns
        similarities = []
        for cluster in self.clusters:
            if cluster['cluster_id'] == preferred_cluster_id:
                continue
            
            # Calculate similarity score
            size_diff = abs(cluster['size'] - preferred['size']) / max(cluster['size'], preferred['size'])
            reaction_diff = abs(cluster['avg_reactions'] - preferred['avg_reactions']) / max(cluster['avg_reactions'], preferred['avg_reactions'])
            reply_diff = abs(cluster['avg_replies'] - preferred['avg_replies']) / max(cluster['avg_replies'], preferred['avg_replies'])
            
            similarity = 1 - (size_diff + reaction_diff + reply_diff) / 3
            similarities.append((cluster['cluster_id'], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [cid for cid, _ in similarities[:n]]
    
    def run_onboarding(self, max_rounds=3):
        """Run the onboarding process"""
        print("Welcome to the Crypto Social Network!")
        print("Let's find your preferred content niche...\n")
        
        round_num = 1
        candidate_clusters = list(range(len(self.clusters)))
        
        while round_num <= max_rounds and len(candidate_clusters) > 3:
            print(f"\n{'*'*80}")
            print(f"ROUND {round_num}")
            print(f"{'*'*80}")
            
            # Show diverse clusters
            clusters_to_show = self.get_diverse_clusters(3)
            shown_ids = []
            
            for cluster in clusters_to_show:
                cluster_id = self.show_posts_from_cluster(cluster)
                shown_ids.append(cluster_id)
            
            # Get preference
            preference = self.get_user_preference(shown_ids)
            
            if preference == 'none':
                # Exclude shown clusters and try different ones
                self.excluded_clusters.update(shown_ids)
                print("\nLet's try some different content types...")
            else:
                # User liked one - find similar clusters
                self.user_preferences.append(preference)
                print(f"\nGreat! You liked cluster {preference}. Finding similar content...")
                
                # Narrow down to similar clusters
                similar = self.find_similar_clusters(preference, n=10)
                candidate_clusters = [preference] + similar
                
                # Exclude very different clusters
                all_clusters = set(range(len(self.clusters)))
                very_different = all_clusters - set(candidate_clusters) - self.excluded_clusters
                self.excluded_clusters.update(list(very_different)[:len(very_different)//2])
            
            round_num += 1
        
        # Final recommendation
        print(f"\n{'*'*80}")
        print("ONBOARDING COMPLETE!")
        print(f"{'*'*80}")
        
        if self.user_preferences:
            print(f"\nBased on your preferences, we recommend content from clusters: {self.user_preferences}")
            print("\nHere are some posts from your preferred niche:")
            
            # Show final recommendations
            for pref_id in self.user_preferences[-1:]:
                cluster = self.cluster_map[pref_id]
                self.show_posts_from_cluster(cluster, n_posts=3)
        else:
            print("\nWe'll show you a variety of content to start. You can refine your preferences as you use the platform!")

def main():
    # Create onboarding instance
    onboarding = UserOnboarding()
    
    # Print cluster statistics
    print("Cluster Statistics:")
    print(f"Total clusters: {len(onboarding.clusters)}")
    print(f"Total posts: {sum(c['size'] for c in onboarding.clusters):,}")
    print(f"Cluster sizes: {min(c['size'] for c in onboarding.clusters):,} - {max(c['size'] for c in onboarding.clusters):,}")
    
    # Run demo
    print("\n" + "="*80)
    print("DEMO: User Onboarding Process")
    print("="*80)
    
    onboarding.run_onboarding()

if __name__ == "__main__":
    main()