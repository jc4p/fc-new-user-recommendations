#!/usr/bin/env python3
"""
Filter casts based on author quality and engagement from quality users
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def load_quality_casts():
    """Load pre-exported quality casts data"""
    
    print("Loading pre-exported quality casts data...")
    
    # Check if raw data exists
    if not os.path.exists('data/quality_casts_raw.parquet'):
        print("ERROR: data/quality_casts_raw.parquet not found!")
        print("Please run ./export_quality_data.sh first to export the data")
        exit(1)
    
    # Load the pre-exported data
    df = pd.read_parquet('data/quality_casts_raw.parquet')
    print(f"Loaded {len(df):,} quality casts from pre-exported data")
    
    return df

def add_author_scores(df):
    """Add author quality scores to casts"""
    
    print("\nAdding author quality scores...")
    
    # Load user scores
    user_scores = pd.read_parquet('data/user_quality_scores.parquet')
    
    # Merge with user scores
    df = df.merge(
        user_scores[['fid', 'quality_score', 'quality_category']], 
        left_on='Fid',
        right_on='fid',
        how='left'
    )
    
    # Calculate engagement quality ratio
    df['quality_engagement_ratio'] = (
        (df['quality_reactions'] + df['quality_replies'] * 2) / 
        (df['total_reactions'] + df['total_replies'] * 2 + 1)
    )
    
    # Calculate overall cast score
    df['cast_score'] = (
        df['quality_score'] * 0.3 +  # Author quality
        df['quality_reactions'] * 0.3 +  # Quality reactions
        df['quality_replies'] * 0.4  # Quality replies (weighted higher)
    )
    
    return df

def ensure_diversity_by_author(df, posts_per_author=10):
    """Ensure diversity by limiting posts per author"""
    
    print("\nEnsuring author diversity...")
    
    # Sort by cast score within each author
    df = df.sort_values(['Fid', 'cast_score'], ascending=[True, False])
    
    # Take top N posts per author
    df_diverse = df.groupby('Fid').head(posts_per_author)
    
    # Sort by overall cast score
    df_diverse = df_diverse.sort_values('cast_score', ascending=False)
    
    print(f"Reduced from {len(df):,} to {len(df_diverse):,} posts")
    print(f"Unique authors: {df_diverse['Fid'].nunique():,}")
    
    return df_diverse

def main():
    # Load pre-exported quality casts
    df = load_quality_casts()
    
    # Add author scores
    df = add_author_scores(df)
    
    # Ensure diversity by limiting posts per author
    df = ensure_diversity_by_author(df, posts_per_author=10)
    
    # Take top posts by cast score
    final_size = min(100000, len(df))
    df = df.nlargest(final_size, 'cast_score')
    
    print(f"\nFinal dataset: {len(df):,} high-quality, diverse casts")
    
    # Save results
    df.to_parquet('data/high_quality_casts.parquet', index=False)
    print("Saved to data/high_quality_casts.parquet")
    
    # Print statistics
    print("\nContent statistics:")
    print(f"Unique authors: {df['Fid'].nunique():,}")
    print(f"Average author quality score: {df['quality_score'].mean():.2f}")
    print(f"Average quality reactions: {df['quality_reactions'].mean():.1f}")
    print(f"Average quality replies: {df['quality_replies'].mean():.1f}")
    print(f"Quality engagement ratio: {df['quality_engagement_ratio'].mean():.2%}")
    
    # Show score distribution
    print("\nCast score distribution:")
    print(df['cast_score'].describe())
    
    # Show sample high-quality posts
    print("\nTop 5 posts by cast score:")
    for i, (_, post) in enumerate(df.head(5).iterrows()):
        print(f"\n{i+1}. Author quality: {post['quality_score']:.1f}, Reactions: {post['quality_reactions']}, Replies: {post['quality_replies']}")
        print(f"   Text: {post['Text'][:150]}...")

if __name__ == "__main__":
    main()