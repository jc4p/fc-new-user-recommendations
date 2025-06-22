#!/usr/bin/env python3
"""
Fast clustering of crypto social media posts using K-means
Extract representative posts from each cluster for user onboarding
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
import os
import json

warnings.filterwarnings('ignore')

def load_embeddings(file_path):
    """Load embeddings from parquet file"""
    print(f"Loading embeddings from {file_path}...")
    df = pd.read_parquet(file_path)
    embeddings = np.vstack(df['embedding'].values)
    print(f"Loaded {len(embeddings):,} embeddings of dimension {embeddings.shape[1]}")
    return df, embeddings

def reduce_dimensions_fast(embeddings, n_components=50):
    """Fast dimension reduction using PCA"""
    print(f"\nReducing dimensions from {embeddings.shape[1]} to {n_components} using PCA...")
    
    # Use PCA for faster reduction
    reducer = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    print(f"Dimension reduction complete: {reduced_embeddings.shape}")
    print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    
    return reduced_embeddings, reducer

def cluster_embeddings_fast(embeddings, n_clusters=25):
    """Fast clustering using MiniBatchKMeans"""
    print(f"\nPerforming K-means clustering with K={n_clusters}...")
    
    # Use MiniBatchKMeans for speed
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=5000,
        n_init=10,
        max_iter=100
    )
    
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate cluster statistics
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    
    return labels, kmeans, cluster_sizes

def get_cluster_representatives(df, embeddings, labels, kmeans, top_k=5):
    """Get representative posts from each cluster"""
    print("\nExtracting representative posts from each cluster...")
    
    cluster_info = []
    
    for cluster_id in tqdm(range(kmeans.n_clusters), desc="Processing clusters"):
        # Get indices of posts in this cluster
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get cluster center
        center = kmeans.cluster_centers_[cluster_id]
        
        # Calculate distances to center
        cluster_embeddings = embeddings[cluster_mask]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        
        # Get posts closest to center
        closest_indices = cluster_indices[np.argsort(distances)[:top_k]]
        
        # Also get most popular posts in cluster
        cluster_df = df.iloc[cluster_indices]
        popular_indices = cluster_df.nlargest(min(top_k, len(cluster_df)), 'reaction_count').index
        
        # Combine closest and popular (unique)
        representative_indices = list(dict.fromkeys(
            list(closest_indices) + list(popular_indices)
        ))[:top_k * 2]
        
        # Get cluster statistics
        cluster_stats = {
            'cluster_id': cluster_id,
            'size': len(cluster_indices),
            'avg_reactions': float(cluster_df['reaction_count'].mean()),
            'avg_replies': float(cluster_df['reply_count'].mean()),
            'representative_posts': []
        }
        
        # Add representative posts
        for idx in representative_indices:
            post = df.iloc[idx]
            distance_idx = np.where(cluster_indices == idx)[0]
            cluster_stats['representative_posts'].append({
                'text': post['text'],
                'reactions': int(post['reaction_count']),
                'replies': int(post['reply_count']),
                'fid': int(post['fid']),
                'hash': post['hash_hex'],
                'distance_to_center': float(distances[distance_idx[0]]) if len(distance_idx) > 0 else float('inf')
            })
        
        cluster_info.append(cluster_stats)
    
    return cluster_info

def analyze_cluster_themes(cluster_info, df, labels):
    """Analyze common themes in each cluster"""
    print("\nAnalyzing cluster themes...")
    
    for cluster in cluster_info[:10]:  # Just show first 10 clusters
        cluster_id = cluster['cluster_id']
        cluster_mask = labels == cluster_id
        cluster_texts = df[cluster_mask]['text'].tolist()
        
        # Get word frequencies (simple approach)
        from collections import Counter
        import re
        
        words = []
        for text in cluster_texts[:100]:  # Sample first 100
            # Extract words (alphanumeric, excluding common words)
            text_words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            words.extend(text_words)
        
        # Remove common words
        stop_words = {'that', 'this', 'with', 'from', 'have', 'will', 'your', 'what', 
                      'when', 'where', 'which', 'their', 'would', 'there', 'could',
                      'should', 'about', 'after', 'before', 'because', 'been', 'being',
                      'just', 'like', 'more', 'very', 'much', 'some', 'only'}
        words = [w for w in words if w not in stop_words]
        
        # Get most common words
        word_freq = Counter(words).most_common(10)
        cluster['common_words'] = word_freq
        
        # Print cluster summary
        print(f"\nCluster {cluster_id} (size: {cluster['size']:,})")
        print(f"  Avg reactions: {cluster['avg_reactions']:.1f}, Avg replies: {cluster['avg_replies']:.1f}")
        print(f"  Common words: {', '.join([w[0] for w in word_freq[:5]])}")
        if cluster['representative_posts']:
            print(f"  Sample: {cluster['representative_posts'][0]['text'][:100]}...")

def save_results(cluster_info, kmeans, reducer, labels, output_dir='data'):
    """Save clustering results"""
    print(f"\nSaving results to {output_dir}/...")
    
    # Save cluster info as JSON
    with open(f'{output_dir}/cluster_info.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    # Save models
    with open(f'{output_dir}/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open(f'{output_dir}/pca_reducer.pkl', 'wb') as f:
        pickle.dump(reducer, f)
    
    # Save labels
    np.save(f'{output_dir}/cluster_labels.npy', labels)
    
    # Create summary for easy viewing
    summary = {
        'total_clusters': len(cluster_info),
        'total_posts': len(labels),
        'clusters': []
    }
    
    for cluster in cluster_info:
        summary['clusters'].append({
            'id': cluster['cluster_id'],
            'size': cluster['size'],
            'top_post': cluster['representative_posts'][0]['text'][:200] if cluster['representative_posts'] else '',
            'common_words': [w[0] for w in cluster.get('common_words', [])[:5]]
        })
    
    with open(f'{output_dir}/cluster_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Results saved successfully!")

def main():
    # Load embeddings
    df, embeddings = load_embeddings('data/cast_embeddings_bge.parquet')
    
    # Fast dimension reduction
    reduced_embeddings, reducer = reduce_dimensions_fast(embeddings, n_components=50)
    
    # Fast clustering
    n_clusters = 25  # Good balance for content diversity
    labels, kmeans, cluster_sizes = cluster_embeddings_fast(
        reduced_embeddings, n_clusters=n_clusters
    )
    
    print("\nCluster sizes:")
    print(cluster_sizes.head(10))
    print(f"Min cluster size: {cluster_sizes.min()}, Max: {cluster_sizes.max()}")
    
    # Get representative posts
    cluster_info = get_cluster_representatives(
        df, reduced_embeddings, labels, kmeans, top_k=5
    )
    
    # Analyze themes
    analyze_cluster_themes(cluster_info, df, labels)
    
    # Save results
    save_results(cluster_info, kmeans, reducer, labels)
    
    print("\nClustering complete! Results saved to data/ directory.")
    print(f"Found {n_clusters} distinct content clusters.")
    print("Use cluster_info.json to access representative posts for user onboarding.")

if __name__ == "__main__":
    main()