#!/usr/bin/env python3
"""
Sample diverse users from Farcaster data for quality labeling
"""
import duckdb
import json
import numpy as np
from tqdm import tqdm
import random

def get_user_samples(conn, n_samples=10000):
    """Sample diverse users based on activity patterns"""
    
    print("Analyzing user activity distribution...")
    
    # Get user activity stats
    user_stats_query = """
    WITH user_activity AS (
        SELECT 
            Fid,
            COUNT(*) as post_count,
            AVG(LENGTH(Text)) as avg_text_length,
            MAX(CAST(Timestamp AS BIGINT)) as last_post_time,
            MIN(CAST(Timestamp AS BIGINT)) as first_post_time,
            SUM(CASE WHEN LOWER(Text) LIKE 'gm%' OR LOWER(Text) LIKE 'good morning%' THEN 1 ELSE 0 END) as gm_posts,
            SUM(CASE WHEN Text LIKE '%?%' THEN 1 ELSE 0 END) as question_posts,
            COUNT(DISTINCT DATE_TRUNC('day', TIMESTAMP '2021-01-01' + (CAST(Timestamp AS BIGINT) * INTERVAL '1 second'))) as active_days
        FROM 'data/casts.parquet'
        WHERE Text IS NOT NULL AND LENGTH(Text) > 10
        GROUP BY Fid
        HAVING COUNT(*) >= 10  -- At least 10 posts
    )
    SELECT * FROM user_activity
    ORDER BY post_count DESC
    LIMIT 1000000  -- Get top 1M users by activity
    """
    
    df = conn.execute(user_stats_query).fetch_df()
    print(f"Found {len(df):,} users with 10+ posts")
    
    # Categorize users
    df['posts_per_day'] = df['post_count'] / (df['active_days'] + 1)
    df['gm_ratio'] = df['gm_posts'] / df['post_count']
    df['question_ratio'] = df['question_posts'] / df['post_count']
    
    # Define user segments for diverse sampling
    segments = {
        'high_volume': df[df['post_count'] > df['post_count'].quantile(0.9)],
        'medium_volume': df[(df['post_count'] > df['post_count'].quantile(0.5)) & 
                           (df['post_count'] <= df['post_count'].quantile(0.9))],
        'low_volume': df[(df['post_count'] >= 10) & 
                        (df['post_count'] <= df['post_count'].quantile(0.5))],
        'high_gm': df[df['gm_ratio'] > 0.3],
        'questioners': df[df['question_ratio'] > 0.2],
        'long_form': df[df['avg_text_length'] > 200],
        'short_form': df[df['avg_text_length'] < 50]
    }
    
    # Sample from each segment
    samples_per_segment = max(n_samples // len(segments), 100)
    sampled_fids = set()
    
    print("\nSampling users from different segments:")
    for segment_name, segment_df in segments.items():
        n = min(samples_per_segment, len(segment_df))
        if n > 0:
            segment_sample = segment_df.sample(n=n, replace=False)
            sampled_fids.update(segment_sample['Fid'].tolist())
            print(f"  {segment_name}: {n} users")
    
    # Ensure we have exactly n_samples
    sampled_fids = list(sampled_fids)
    if len(sampled_fids) > n_samples:
        sampled_fids = random.sample(sampled_fids, n_samples)
    elif len(sampled_fids) < n_samples:
        # Add more random users
        remaining_fids = set(df['Fid'].tolist()) - set(sampled_fids)
        additional = random.sample(list(remaining_fids), 
                                 min(n_samples - len(sampled_fids), len(remaining_fids)))
        sampled_fids.extend(additional)
    
    print(f"\nTotal unique users sampled: {len(sampled_fids)}")
    
    # Get posts for sampled users
    print("\nFetching posts for sampled users...")
    users_data = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(sampled_fids), batch_size)):
        batch_fids = sampled_fids[i:i+batch_size]
        fid_list = ','.join(map(str, batch_fids))
        
        posts_query = f"""
        SELECT 
            Fid,
            Text,
            CAST(Timestamp AS BIGINT) as timestamp
        FROM 'data/casts.parquet'
        WHERE Fid IN ({fid_list})
          AND Text IS NOT NULL
          AND LENGTH(Text) > 10
        ORDER BY Fid, timestamp DESC
        """
        
        posts_df = conn.execute(posts_query).fetch_df()
        
        # Group by user
        for fid in batch_fids:
            user_posts = posts_df[posts_df['Fid'] == fid]
            if len(user_posts) > 0:
                # Take up to 50 most recent posts
                recent_posts = user_posts.head(50)['Text'].tolist()
                
                # Get user stats
                user_stat = df[df['Fid'] == fid].iloc[0] if fid in df['Fid'].values else None
                
                user_data = {
                    'fid': int(fid),
                    'posts': recent_posts,
                    'post_count': int(user_stat['post_count']) if user_stat is not None else len(recent_posts),
                    'avg_text_length': float(user_stat['avg_text_length']) if user_stat is not None else np.mean([len(p) for p in recent_posts]),
                    'gm_ratio': float(user_stat['gm_ratio']) if user_stat is not None else 0,
                    'question_ratio': float(user_stat['question_ratio']) if user_stat is not None else 0,
                    'sample_post': recent_posts[0] if recent_posts else ""
                }
                users_data.append(user_data)
    
    return users_data

def main():
    # Connect to DuckDB
    conn = duckdb.connect()
    
    # Sample users
    users_data = get_user_samples(conn, n_samples=10000)
    
    # Save to JSON
    output_file = 'data/users_to_label.json'
    with open(output_file, 'w') as f:
        json.dump(users_data, f, indent=2)
    
    print(f"\nSaved {len(users_data)} users to {output_file}")
    
    # Print summary statistics
    post_counts = [u['post_count'] for u in users_data]
    avg_lengths = [u['avg_text_length'] for u in users_data]
    gm_ratios = [u['gm_ratio'] for u in users_data]
    
    print("\nSampled user statistics:")
    print(f"Post counts: min={min(post_counts)}, max={max(post_counts)}, avg={np.mean(post_counts):.1f}")
    print(f"Avg text lengths: min={min(avg_lengths):.1f}, max={max(avg_lengths):.1f}, avg={np.mean(avg_lengths):.1f}")
    print(f"GM ratios: min={min(gm_ratios):.3f}, max={max(gm_ratios):.3f}, avg={np.mean(gm_ratios):.3f}")
    
    # Show a few examples
    print("\nExample users:")
    for i in range(min(3, len(users_data))):
        user = users_data[i]
        print(f"\nUser {user['fid']}:")
        print(f"  Posts: {user['post_count']}, Avg length: {user['avg_text_length']:.1f}")
        print(f"  GM ratio: {user['gm_ratio']:.2f}, Question ratio: {user['question_ratio']:.2f}")
        print(f"  Sample: {user['sample_post'][:100]}...")

if __name__ == "__main__":
    main()