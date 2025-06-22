#!/usr/bin/env python3
"""
Cluster high-quality content for improved user onboarding
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm
import base64
from collections import Counter
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import time

# Try to import Gemini API
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install with: pip install google-genai")

def initialize_gemini():
    """Initialize Gemini API client"""
    if not GEMINI_AVAILABLE:
        return None
    
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Falling back to keyword-based naming.")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Warning: Could not initialize Gemini API: {e}")
        return None

def generate_embeddings(df, model_name="BAAI/bge-large-en-v1.5", batch_size=512):
    """Generate embeddings for quality content"""
    
    print(f"Loading embedding model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    
    if torch.cuda.is_available():
        model = model.half()  # Use half precision for speed
    
    texts = df['Text'].tolist()
    embeddings = []
    
    print(f"Generating embeddings for {len(texts):,} texts...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Add instruction prefix for better clustering
        batch_texts = [f"Represent this post for clustering: {text}" for text in batch_texts]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                batch_embeddings = model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=batch_size
                )
            batch_embeddings = batch_embeddings.float().cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def cluster_content(embeddings, n_clusters=50):
    """Cluster content into meaningful topics"""
    
    print(f"\nReducing dimensions for clustering...")
    pca = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    print(f"Clustering into {n_clusters} topics...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=5000,
        n_init=10
    )
    
    labels = kmeans.fit_predict(reduced_embeddings)
    
    return labels, kmeans, reduced_embeddings

def extract_cluster_themes(cluster_texts, n_keywords=30):
    """Extract themes from cluster using TF-IDF"""
    
    # Common English stopwords to exclude
    stopwords = {
        'the', 'and', 'for', 'this', 'that', 'with', 'from', 'have', 
        'will', 'your', 'what', 'when', 'where', 'which', 'their', 
        'would', 'there', 'could', 'should', 'about', 'after', 
        'before', 'because', 'been', 'being', 'just', 'like', 
        'more', 'very', 'much', 'some', 'only', 'into', 'over',
        'than', 'then', 'them', 'these', 'those', 'through',
        'during', 'while', 'such', 'both', 'each', 'other'
    }
    
    # Use TF-IDF to find distinctive keywords for this cluster
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words=list(stopwords),
        ngram_range=(1, 2),  # Include bigrams
        min_df=2
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top TF-IDF scores
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-n_keywords:][::-1]
        
        keywords = [(feature_names[i], scores[i]) for i in top_indices]
    except:
        # Fallback to simple word frequency
        words = []
        for text in cluster_texts[:200]:
            text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words.extend([w for w in text_words if w not in stopwords])
        
        word_freq = Counter(words).most_common(n_keywords)
        keywords = word_freq
    
    return keywords

def generate_cluster_name_with_gemini(client, keywords, representatives):
    """Generate a descriptive name for the cluster using Gemini API"""
    
    # Get sample posts for context
    sample_posts = [rep['text'][:200] for rep in representatives[:3]]
    keywords_str = ', '.join([kw[0] for kw in keywords[:10]])
    
    prompt = f"""Analyze this cluster of social media posts and generate a human-friendly name that people would naturally use to describe this topic.

Keywords: {keywords_str}

Sample posts:
1. {sample_posts[0] if len(sample_posts) > 0 else 'N/A'}
2. {sample_posts[1] if len(sample_posts) > 1 else 'N/A'}
3. {sample_posts[2] if len(sample_posts) > 2 else 'N/A'}

Rules for naming:
- Use 2-4 words maximum
- Make it specific and recognizable (e.g., "Startup Founders", "Food Photography", "Parenting Tips")
- Avoid vague terms like "General Discussion", "Various Topics", "Mixed Content"
- Use terms people would actually search for or identify with
- Focus on the main activity, interest, or community represented
- If it's about a specific topic, name that topic clearly
- Prefer noun phrases that describe either the topic or the people discussing it

Return only the cluster name, nothing else."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=20,
            )
        )
        
        name = response.text.strip()
        # Clean up the name
        name = name.replace('"', '').replace("'", '').strip()
        name = name.split('\n')[0]  # Take first line only
        
        # Ensure it's not too long
        if len(name) > 40:
            name = name[:40].rsplit(' ', 1)[0]
        
        return name
    except Exception as e:
        print(f"Gemini API naming failed: {e}")
        return None

def generate_cluster_name(keywords, representatives):
    """Fallback: Generate a descriptive name for the cluster based on keywords"""
    
    # Get top keywords
    top_words = [kw[0] for kw in keywords[:10]]
    
    # Try to identify common themes from keywords
    theme_mappings = {
        # Food related
        ('food', 'recipe', 'cooking', 'meal', 'restaurant', 'eat', 'delicious', 'taste', 'dinner', 'lunch'): 'Food & Cooking',
        ('coffee', 'tea', 'drink', 'beverage', 'cafe'): 'Coffee & Drinks',
        
        # Tech related
        ('build', 'ship', 'launch', 'product', 'app', 'software', 'code', 'developer'): 'Product Building',
        ('ai', 'ml', 'machine learning', 'artificial intelligence', 'model'): 'AI & Machine Learning',
        ('design', 'ui', 'ux', 'interface', 'user experience'): 'Design & UX',
        
        # Creative
        ('art', 'artist', 'creative', 'paint', 'draw', 'artwork'): 'Visual Arts',
        ('music', 'song', 'album', 'artist', 'band', 'concert'): 'Music',
        ('photo', 'photography', 'picture', 'shot', 'camera'): 'Photography',
        ('write', 'writing', 'author', 'book', 'story', 'novel'): 'Writing & Books',
        
        # Lifestyle
        ('travel', 'trip', 'journey', 'destination', 'explore'): 'Travel',
        ('fitness', 'workout', 'exercise', 'gym', 'health'): 'Fitness & Health',
        ('family', 'kids', 'children', 'parenting', 'parent'): 'Family & Parenting',
        ('fashion', 'style', 'outfit', 'clothes', 'wear'): 'Fashion & Style',
        
        # Professional
        ('startup', 'founder', 'entrepreneur', 'business', 'company'): 'Startups & Business',
        ('job', 'career', 'work', 'hiring', 'recruit'): 'Career & Jobs',
        ('invest', 'investment', 'market', 'trading', 'finance'): 'Finance & Investing',
        
        # Other interests
        ('game', 'gaming', 'play', 'gamer', 'video game'): 'Gaming',
        ('movie', 'film', 'cinema', 'watch', 'series'): 'Movies & TV',
        ('sport', 'sports', 'team', 'player', 'game', 'match'): 'Sports',
        ('nature', 'outdoor', 'hiking', 'mountain', 'forest'): 'Nature & Outdoors',
        ('science', 'research', 'study', 'experiment', 'discovery'): 'Science & Research',
        ('politics', 'political', 'policy', 'government', 'election'): 'Politics',
        ('philosophy', 'think', 'thought', 'idea', 'question'): 'Philosophy & Ideas',
        ('community', 'people', 'friends', 'social', 'together'): 'Community',
        ('news', 'media', 'journalism', 'report', 'story'): 'News & Media',
        ('education', 'learn', 'teach', 'student', 'school'): 'Education',
        ('mental', 'wellness', 'mindfulness', 'meditation', 'therapy'): 'Mental Wellness'
    }
    
    # Check which theme matches best
    lower_keywords = [w.lower() for w in top_words]
    
    for theme_keywords, theme_name in theme_mappings.items():
        matches = sum(1 for kw in lower_keywords if any(theme_kw in kw for theme_kw in theme_keywords))
        if matches >= 2:  # At least 2 matching keywords
            return theme_name
    
    # If no clear theme, try to create a sensible combination
    if len(top_words) >= 2:
        # Look for the most meaningful keywords
        meaningful_words = [w for w in top_words if len(w) > 4 and w.lower() not in ['https', 'com', 'org', 'net', 'post', 'share', 'like']]
        if len(meaningful_words) >= 2:
            return f"{meaningful_words[0].title()} {meaningful_words[1].title()}"
    
    # Last resort - use the most prominent keyword
    if top_words:
        return f"{top_words[0].title()} Topics"
    
    return "Mixed Topics"

def process_single_cluster(args):
    """Process a single cluster (for parallel processing)"""
    cluster_id, cluster_df, embeddings, center, gemini_client = args
    
    if len(cluster_df) == 0:
        return None
    
    # Calculate distances to cluster center
    cluster_embeddings = embeddings[cluster_df.index]
    distances = np.linalg.norm(cluster_embeddings - center, axis=1)
    cluster_df = cluster_df.copy()
    cluster_df['distance_to_center'] = distances
    
    # Get representative posts (closest to center + most engaged)
    closest_posts = cluster_df.nsmallest(10, 'distance_to_center')
    most_engaged = cluster_df.nlargest(10, 'cast_score')
    
    # Combine and deduplicate
    representatives = pd.concat([closest_posts, most_engaged]).drop_duplicates(subset=['Hash']).head(10)
    
    # Extract themes using TF-IDF
    cluster_texts = cluster_df['Text'].tolist()
    keywords = extract_cluster_themes(cluster_texts)
    
    # Prepare representative data for naming
    rep_data = []
    for _, post in representatives.iterrows():
        # Convert hash from base64 to hex if needed
        try:
            hash_hex = base64.b64decode(post['Hash']).hex()
        except:
            hash_hex = post['Hash']
        
        rep_data.append({
            'hash': hash_hex,
            'fid': int(post['Fid']),
            'text': post['Text'],
            'author_quality': float(post['quality_score']),
            'quality_reactions': int(post['quality_reactions']),
            'quality_replies': int(post['quality_replies']),
            'cast_score': float(post['cast_score'])
        })
    
    # Generate cluster name
    if gemini_client:
        cluster_name = generate_cluster_name_with_gemini(gemini_client, keywords, rep_data)
        if not cluster_name:
            cluster_name = generate_cluster_name(keywords, rep_data)
    else:
        cluster_name = generate_cluster_name(keywords, rep_data)
    
    cluster_data = {
        'cluster_id': int(cluster_id),
        'name': cluster_name,
        'size': len(cluster_df),
        'keywords': [(kw[0], float(kw[1])) for kw in keywords[:20]],
        'top_words': [kw[0] for kw in keywords[:10]],
        'avg_quality_score': float(cluster_df['quality_score'].mean()),
        'avg_quality_reactions': float(cluster_df['quality_reactions'].mean()),
        'avg_quality_replies': float(cluster_df['quality_replies'].mean()),
        'avg_cast_score': float(cluster_df['cast_score'].mean()),
        'representatives': rep_data
    }
    
    return cluster_data

def extract_cluster_info_parallel(df, labels, embeddings, kmeans, gemini_client=None):
    """Extract representative posts and themes from each cluster using parallel processing"""
    
    cluster_info = []
    
    # Prepare arguments for parallel processing
    args_list = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_mask = labels == cluster_id
        cluster_df = df[cluster_mask]
        center = kmeans.cluster_centers_[cluster_id]
        
        args_list.append((cluster_id, cluster_df, embeddings, center, gemini_client))
    
    # Use ThreadPoolExecutor for I/O-bound Gemini API calls
    # Limit workers to avoid rate limiting
    max_workers = min(8, mp.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_cluster, args_list),
            total=len(args_list),
            desc="Analyzing clusters"
        ))
    
    # Filter out None results and sort by average cast score
    cluster_info = [r for r in results if r is not None]
    cluster_info.sort(key=lambda x: -x['avg_cast_score'])
    
    return cluster_info


def main():
    import os
    
    # Initialize Gemini API for cluster naming
    gemini_client = initialize_gemini()
    if gemini_client:
        print("Using Gemini API for cluster naming")
    else:
        print("Using keyword-based cluster naming")
    
    # Load quality content
    print("Loading high-quality casts...")
    df = pd.read_parquet('data/high_quality_casts.parquet')
    print(f"Loaded {len(df):,} quality casts")
    
    # Generate embeddings
    embeddings = generate_embeddings(df)
    
    # Cluster content (more clusters for finer granularity)
    n_clusters = 50
    labels, kmeans, reduced_embeddings = cluster_content(embeddings, n_clusters)
    
    # Add cluster labels to dataframe
    df['cluster_id'] = labels
    
    # Extract cluster information with parallel processing
    cluster_info = extract_cluster_info_parallel(df, labels, reduced_embeddings, kmeans, gemini_client)
    
    # Save results
    print("\nSaving results...")
    
    with open('data/quality_clusters.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    # Save models
    with open('models/quality_kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Save dataframe with cluster assignments
    df.to_parquet('data/high_quality_casts_clustered.parquet', index=False)
    
    # Create summary
    summary = {
        'total_clusters': len(cluster_info),
        'total_posts': len(df),
        'cluster_sizes': {
            'mean': np.mean([c['size'] for c in cluster_info]),
            'median': np.median([c['size'] for c in cluster_info]),
            'min': min(c['size'] for c in cluster_info),
            'max': max(c['size'] for c in cluster_info)
        }
    }
    
    with open('data/clustering_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print(f"\nClustered into {len(cluster_info)} topics")
    
    print("\nTop 10 clusters by quality:")
    for i, cluster in enumerate(cluster_info[:10]):
        print(f"\n{i+1}. {cluster['name']} (ID: {cluster['cluster_id']})")
        print(f"   Size: {cluster['size']} posts")
        print(f"   Avg cast score: {cluster['avg_cast_score']:.1f}")
        print(f"   Keywords: {', '.join(cluster['top_words'][:5])}")
        if cluster['representatives']:
            print(f"   Sample: {cluster['representatives'][0]['text'][:100]}...")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    main()
