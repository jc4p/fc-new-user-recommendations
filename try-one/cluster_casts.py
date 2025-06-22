#!/usr/bin/env python3
"""
Cluster crypto social media posts using HDBSCAN and K-means
Extract representative posts from each cluster for user onboarding
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
from tqdm import tqdm
import warnings
import os
import multiprocessing

# Set environment variables for parallel processing
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'

warnings.filterwarnings('ignore')

print(f"Using {multiprocessing.cpu_count()} CPU cores for parallel processing")

def load_embeddings(file_path):
    """Load embeddings from parquet file"""
    print(f"Loading embeddings from {file_path}...")
    df = pd.read_parquet(file_path)
    embeddings = np.vstack(df['embedding'].values)
    print(f"Loaded {len(embeddings):,} embeddings of dimension {embeddings.shape[1]}")
    return df, embeddings

def reduce_dimensions(embeddings, n_components=50):
    """Reduce embedding dimensions for better clustering"""
    print(f"\nReducing dimensions from {embeddings.shape[1]} to {n_components}...")
    
    # Use UMAP for dimension reduction - better for clustering than PCA
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        verbose=True,
        n_jobs=-1  # Use all available cores
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Dimension reduction complete: {reduced_embeddings.shape}")
    
    return reduced_embeddings, reducer

def find_optimal_clusters(embeddings, min_k=15, max_k=30):
    """Find optimal number of clusters using silhouette score"""
    print(f"\nFinding optimal number of clusters (testing {min_k} to {max_k})...")
    
    scores = []
    for k in tqdm(range(min_k, max_k + 1), desc="Testing K values"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=10000)
        scores.append((k, score))
        print(f"  K={k}: silhouette score = {score:.4f}")
    
    # Find best k
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\nBest K: {best_k}")
    
    return best_k, scores

def cluster_embeddings(embeddings, n_clusters=None):
    """Perform clustering using HDBSCAN for initial clustering and K-means for final"""
    print("\nPerforming hierarchical clustering...")
    
    # First, use HDBSCAN to find natural clusters
    print("Step 1: HDBSCAN for natural cluster discovery...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1  # Use all cores for distance calculations
    )
    
    hdbscan_labels = clusterer.fit_predict(embeddings)
    n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_noise = list(hdbscan_labels).count(-1)
    
    print(f"HDBSCAN found {n_hdbscan_clusters} clusters ({n_noise} noise points)")
    
    # If n_clusters not specified, use HDBSCAN result as guide
    if n_clusters is None:
        n_clusters = max(20, min(30, n_hdbscan_clusters))
    
    # Step 2: K-means for final clustering
    print(f"\nStep 2: K-means clustering with K={n_clusters}...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
        max_iter=300
    )
    
    kmeans_labels = kmeans.fit_predict(embeddings)
    
    # Calculate cluster statistics
    cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
    
    return kmeans_labels, kmeans, clusterer, cluster_sizes

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
        popular_indices = cluster_df.nlargest(top_k, 'reaction_count').index
        
        # Combine closest and popular (unique)
        representative_indices = list(dict.fromkeys(
            list(closest_indices) + list(popular_indices)
        ))[:top_k * 2]
        
        # Get cluster statistics
        cluster_stats = {
            'cluster_id': cluster_id,
            'size': len(cluster_indices),
            'avg_reactions': cluster_df['reaction_count'].mean(),
            'avg_replies': cluster_df['reply_count'].mean(),
            'representative_posts': []
        }
        
        # Add representative posts
        for idx in representative_indices:
            post = df.iloc[idx]
            cluster_stats['representative_posts'].append({
                'text': post['text'],
                'reactions': post['reaction_count'],
                'replies': post['reply_count'],
                'fid': post['fid'],
                'hash': post['hash_hex'],
                'distance_to_center': distances[np.where(cluster_indices == idx)[0][0]]
                if idx in cluster_indices else float('inf')
            })
        
        cluster_info.append(cluster_stats)
    
    return cluster_info

def analyze_cluster_themes(cluster_info, df, labels):
    """Analyze common themes in each cluster"""
    print("\nAnalyzing cluster themes...")
    
    for cluster in cluster_info:
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
                      'should', 'about', 'after', 'before', 'because', 'been', 'being'}
        words = [w for w in words if w not in stop_words]
        
        # Get most common words
        word_freq = Counter(words).most_common(10)
        cluster['common_words'] = word_freq
        
        # Print cluster summary
        print(f"\nCluster {cluster_id} (size: {cluster['size']})")
        print(f"  Avg reactions: {cluster['avg_reactions']:.1f}")
        print(f"  Common words: {', '.join([w[0] for w in word_freq[:5]])}")
        print(f"  Sample post: {cluster['representative_posts'][0]['text'][:100]}...")

def save_results(cluster_info, kmeans, reducer, labels, output_dir='data'):
    """Save clustering results"""
    import os
    import json
    
    print(f"\nSaving results to {output_dir}/...")
    
    # Save cluster info as JSON
    with open(f'{output_dir}/cluster_info.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    # Save models
    with open(f'{output_dir}/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open(f'{output_dir}/umap_reducer.pkl', 'wb') as f:
        pickle.dump(reducer, f)
    
    # Save labels
    np.save(f'{output_dir}/cluster_labels.npy', labels)
    
    print("Results saved successfully!")

def main():
    # Load embeddings
    df, embeddings = load_embeddings('data/cast_embeddings_bge.parquet')
    
    # Reduce dimensions for better clustering
    reduced_embeddings, reducer = reduce_dimensions(embeddings, n_components=50)
    
    # Find optimal number of clusters
    # optimal_k, scores = find_optimal_clusters(reduced_embeddings, min_k=15, max_k=30)
    
    # For speed, let's use a fixed number
    n_clusters = 25  # Good balance for content diversity
    
    # Perform clustering
    labels, kmeans, hdbscan_model, cluster_sizes = cluster_embeddings(
        reduced_embeddings, n_clusters=n_clusters
    )
    
    print("\nCluster sizes:")
    print(cluster_sizes)
    
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