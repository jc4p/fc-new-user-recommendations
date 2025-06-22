#!/usr/bin/env python3
"""
Extract features for all users to train quality classifier
"""
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime

def extract_text_features(texts):
    """Extract various text quality features from a user's posts"""
    if len(texts) == 0:
        return {}
    
    # Basic stats
    lengths = [len(t) for t in texts]
    
    # Vocabulary richness
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        all_words.extend(words)
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    # Content patterns
    gm_count = sum(1 for t in texts if re.match(r'^(gm|good morning)', t.lower().strip()))
    question_count = sum(1 for t in texts if '?' in t)
    url_count = sum(1 for t in texts if 'http' in t.lower())
    emoji_count = sum(1 for t in texts if any(ord(c) > 127 for c in t))
    
    # Spam indicators
    spam_keywords = ['airdrop', 'claim', 'giveaway', 'free', 'check eligibility']
    spam_count = sum(1 for t in texts if any(kw in t.lower() for kw in spam_keywords))
    
    features = {
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'vocabulary_size': unique_words,
        'vocabulary_richness': unique_words / total_words if total_words > 0 else 0,
        'gm_ratio': gm_count / len(texts),
        'question_ratio': question_count / len(texts),
        'url_ratio': url_count / len(texts),
        'emoji_ratio': emoji_count / len(texts),
        'spam_ratio': spam_count / len(texts),
        'post_count': len(texts)
    }
    
    return features

def extract_temporal_features(timestamps):
    """Extract temporal posting patterns"""
    if len(timestamps) == 0 or len(timestamps) < 2:
        return {
            'posting_regularity': 0,
            'active_hours': 1,
            'active_days': 1,
            'avg_posts_per_day': 1
        }
    
    # Convert to datetime
    dts = [datetime.fromtimestamp(ts + 1609459200) for ts in timestamps]
    
    # Hour distribution
    hours = [dt.hour for dt in dts]
    active_hours = len(set(hours))
    
    # Day distribution
    dates = [dt.date() for dt in dts]
    unique_dates = set(dates)
    active_days = len(unique_dates)
    
    # Posting regularity (std of intervals between posts)
    sorted_ts = sorted(timestamps)
    intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
    regularity = np.std(intervals) if intervals else 0
    
    # Posts per day
    if unique_dates:
        date_range = (max(dates) - min(dates)).days + 1
        avg_posts_per_day = len(timestamps) / date_range if date_range > 0 else len(timestamps)
    else:
        avg_posts_per_day = len(timestamps)
    
    return {
        'posting_regularity': regularity,
        'active_hours': active_hours,
        'active_days': active_days,
        'avg_posts_per_day': avg_posts_per_day
    }

def process_users_batch(conn, user_fids):
    """Process a batch of users and extract features"""
    
    fid_list = ','.join(map(str, user_fids))
    
    # Get posts for users
    query = f"""
    WITH user_posts AS (
        SELECT 
            Fid,
            Text,
            CAST(Timestamp AS BIGINT) as timestamp
        FROM 'data/casts.parquet'
        WHERE Fid IN ({fid_list})
          AND Text IS NOT NULL
          AND LENGTH(Text) > 5
    ),
    user_engagement AS (
        SELECT 
            c.Fid,
            COUNT(DISTINCT r.Fid) as unique_reactors,
            COUNT(r.Hash) as total_reactions
        FROM 'data/casts.parquet' c
        LEFT JOIN 'data/farcaster_reactions.parquet' r
            ON r.TargetCastId = c.Fid || ':' || c.Hash
        WHERE c.Fid IN ({fid_list})
        GROUP BY c.Fid
    )
    SELECT 
        p.Fid,
        ARRAY_AGG(p.Text ORDER BY p.timestamp DESC) as texts,
        ARRAY_AGG(p.timestamp ORDER BY p.timestamp DESC) as timestamps,
        COALESCE(e.unique_reactors, 0) as unique_reactors,
        COALESCE(e.total_reactions, 0) as total_reactions
    FROM user_posts p
    LEFT JOIN user_engagement e ON p.Fid = e.Fid
    GROUP BY p.Fid, e.unique_reactors, e.total_reactions
    """
    
    df = conn.execute(query).fetch_df()
    
    features_list = []
    
    for _, row in df.iterrows():
        fid = row['Fid']
        texts = row['texts'] if row['texts'] is not None else []
        timestamps = row['timestamps'] if row['timestamps'] is not None else []
        
        # Extract text features
        text_features = extract_text_features(texts[:100])  # Use up to 100 most recent
        
        # Extract temporal features
        temporal_features = extract_temporal_features(timestamps[:1000])  # Use up to 1000
        
        # Engagement features
        engagement_features = {
            'unique_reactors': row['unique_reactors'],
            'total_reactions': row['total_reactions'],
            'avg_reactions_per_post': row['total_reactions'] / len(texts) if len(texts) > 0 else 0,
            'reactor_diversity': row['unique_reactors'] / row['total_reactions'] if row['total_reactions'] > 0 else 0
        }
        
        # Combine all features
        features = {
            'fid': fid,
            **text_features,
            **temporal_features,
            **engagement_features
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def main():
    # Connect to DuckDB
    conn = duckdb.connect()
    
    print("Getting list of all users with sufficient activity...")
    
    # Get all users with at least 10 posts
    all_users_query = """
    SELECT DISTINCT Fid
    FROM 'data/casts.parquet'
    WHERE Text IS NOT NULL
    GROUP BY Fid
    HAVING COUNT(*) >= 10
    """
    
    all_fids = conn.execute(all_users_query).fetch_df()['Fid'].tolist()
    print(f"Found {len(all_fids):,} users with 10+ posts")
    
    # Process in batches
    batch_size = 1000
    all_features = []
    
    print("Extracting features for all users...")
    
    for i in tqdm(range(0, len(all_fids), batch_size)):
        batch_fids = all_fids[i:i+batch_size]
        batch_features = process_users_batch(conn, batch_fids)
        all_features.append(batch_features)
    
    # Combine all features
    features_df = pd.concat(all_features, ignore_index=True)
    
    # Fill any NaN values
    features_df = features_df.fillna(0)
    
    print(f"\nExtracted features for {len(features_df):,} users")
    
    # Save to parquet
    features_df.to_parquet('data/user_features.parquet', index=False)
    print("Saved features to data/user_features.parquet")
    
    # Print feature statistics
    print("\nFeature statistics:")
    numeric_cols = [col for col in features_df.columns if col != 'fid']
    for col in numeric_cols[:10]:  # Show first 10 features
        print(f"  {col}: mean={features_df[col].mean():.3f}, std={features_df[col].std():.3f}")

if __name__ == "__main__":
    main()