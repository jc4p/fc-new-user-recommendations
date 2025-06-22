#!/usr/bin/env python3
"""
Score all users using the trained quality classifier
"""
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import lightgbm as lgb

def load_models():
    """Load trained models"""
    with open('models/user_quality_models.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

def score_users_batch(features_df, models):
    """Score a batch of users"""
    
    regression_model = models['regression_model']
    feature_columns = models['feature_columns']
    
    # Ensure we have all required features
    missing_features = set(feature_columns) - set(features_df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feat in missing_features:
            features_df[feat] = 0
    
    # Select features in correct order
    X = features_df[feature_columns]
    
    # Predict quality scores
    quality_scores = regression_model.predict(X, num_iteration=regression_model.best_iteration)
    
    # Clip scores to valid range
    quality_scores = np.clip(quality_scores, 0, 10)
    
    # Create results dataframe
    results = pd.DataFrame({
        'fid': features_df['fid'],
        'quality_score': quality_scores,
        'is_quality_user': (quality_scores >= 5).astype(int),
        'quality_category': pd.cut(
            quality_scores,
            bins=[-0.1, 2.5, 4.5, 6.5, 8.5, 10.1],
            labels=['Spammer', 'Low', 'Average', 'High', 'Leader']
        )
    })
    
    return results

def main():
    # Load models
    print("Loading trained models...")
    models = load_models()
    
    # Load user features
    print("Loading user features...")
    features_df = pd.read_parquet('data/user_features.parquet')
    print(f"Loaded features for {len(features_df):,} users")
    
    # Score users in batches
    batch_size = 10000
    all_scores = []
    
    print("Scoring all users...")
    for i in tqdm(range(0, len(features_df), batch_size)):
        batch = features_df.iloc[i:i+batch_size]
        batch_scores = score_users_batch(batch, models)
        all_scores.append(batch_scores)
    
    # Combine results
    scores_df = pd.concat(all_scores, ignore_index=True)
    
    # Save results
    scores_df.to_parquet('data/user_quality_scores.parquet', index=False)
    print(f"\nSaved quality scores for {len(scores_df):,} users")
    
    # Print statistics
    print("\nUser Quality Distribution:")
    print(scores_df['quality_category'].value_counts())
    print(f"\nAverage quality score: {scores_df['quality_score'].mean():.2f}")
    print(f"Median quality score: {scores_df['quality_score'].median():.2f}")
    
    # High quality users
    high_quality_users = scores_df[scores_df['quality_score'] >= 7]
    print(f"\nHigh quality users (score >= 7): {len(high_quality_users):,} ({len(high_quality_users)/len(scores_df)*100:.1f}%)")
    
    # Save high quality user list separately
    high_quality_users[['fid', 'quality_score']].to_csv('data/high_quality_users.csv', index=False)
    print(f"Saved high quality user list to data/high_quality_users.csv")

if __name__ == "__main__":
    main()