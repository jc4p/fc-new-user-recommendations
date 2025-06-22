#!/usr/bin/env python3
"""
Train a user quality classifier using labeled data from Gemini
"""
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_data():
    """Load labeled users and merge with features"""
    
    # Try to load local labeled data first, fallback to API labeled data
    try:
        with open('data/labeled_users_local.json', 'r') as f:
            labeled_users = json.load(f)
        print("Using locally labeled data (Gemma)")
    except FileNotFoundError:
        try:
            with open('data/labeled_users.json', 'r') as f:
                labeled_users = json.load(f)
            print("Using API labeled data (Gemini)")
        except FileNotFoundError:
            raise FileNotFoundError("No labeled data found. Please run label_users_locally.py or label_users_with_gemini.py first.")
    
    labeled_df = pd.DataFrame(labeled_users)
    print(f"Loaded {len(labeled_df)} labeled users")
    
    # Load features
    features_df = pd.read_parquet('data/user_features.parquet')
    # Convert fid to int to match labeled data
    features_df['fid'] = features_df['fid'].astype(int)
    print(f"Loaded features for {len(features_df)} users")
    
    # Merge
    merged_df = labeled_df.merge(features_df, on='fid', how='inner')
    print(f"Merged data: {len(merged_df)} users with labels and features")
    
    return merged_df

def prepare_training_data(df):
    """Prepare features and labels for training"""
    
    # Create binary classification: high quality (score >= 5) vs low quality
    df['is_quality'] = (df['quality_score'] >= 5).astype(int)
    
    # Also create multi-class labels
    df['quality_class'] = pd.cut(
        df['quality_score'], 
        bins=[-0.1, 2.5, 4.5, 6.5, 8.5, 10.1],
        labels=['Spammer', 'Low', 'Average', 'High', 'Leader']
    )
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in [
        'fid', 'quality_score', 'category', 'reasoning', 'content_themes',
        'is_quality', 'quality_class'
    ]]
    
    print(f"\nUsing {len(feature_cols)} features:")
    print(feature_cols)
    
    return df[feature_cols], df['is_quality'], df['quality_score'], df['quality_class']

def train_binary_classifier(X_train, y_train, X_test, y_test):
    """Train binary classifier for quality vs non-quality"""
    
    print("\nTraining binary classifier (quality vs non-quality)...")
    
    # Train LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'random_state': 42,
        'n_jobs': -1
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )
    
    # Predictions
    y_pred = (model.predict(X_test, num_iteration=model.best_iteration) > 0.5).astype(int)
    
    print("\nBinary Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_imp.head(10))
    
    return model, feature_imp

def train_regression_model(X_train, y_train, X_test, y_test):
    """Train regression model for quality scores"""
    
    print("\nTraining regression model for quality scores...")
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'random_state': 42,
        'n_jobs': -1
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )
    
    # Predictions
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"\nRegression Metrics:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 10], [0, 10], 'r--', lw=2)
    plt.xlabel('Actual Quality Score')
    plt.ylabel('Predicted Quality Score')
    plt.title('Actual vs Predicted Quality Scores')
    plt.savefig('data/quality_score_predictions.png')
    plt.close()
    
    return model

def main():
    # Load and prepare data
    df = load_and_merge_data()
    
    # Check label distribution
    print("\nLabel distribution:")
    print(df['category'].value_counts())
    
    # Prepare features and labels
    X, y_binary, y_scores, y_multiclass = prepare_training_data(df)
    
    # Split data
    X_train, X_test, y_binary_train, y_binary_test, y_scores_train, y_scores_test = train_test_split(
        X, y_binary, y_scores, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print(f"\nTraining set: {len(X_train)} users")
    print(f"Test set: {len(X_test)} users")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train binary classifier
    binary_model, feature_importance = train_binary_classifier(
        X_train, y_binary_train, X_test, y_binary_test
    )
    
    # Train regression model
    regression_model = train_regression_model(
        X_train, y_scores_train, X_test, y_scores_test
    )
    
    # Save models
    models = {
        'binary_model': binary_model,
        'regression_model': regression_model,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'feature_importance': feature_importance
    }
    
    with open('models/user_quality_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("\nModels saved to models/user_quality_models.pkl")
    
    # Create a simple visualization of feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Most Important Features for User Quality')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()
    
    print("Feature importance plot saved to data/feature_importance.png")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    main()