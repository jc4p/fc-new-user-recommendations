# Farcaster Quality Content Clustering Pipeline

A machine learning pipeline for identifying high-quality content on Farcaster and clustering it into meaningful topics for improved user onboarding and content discovery.

## Goals

1. **Identify Quality Users**: Use ML to score users based on their engagement patterns and content quality
2. **Filter High-Quality Content**: Select posts from quality authors with meaningful engagement
3. **Discover Content Categories**: Use unsupervised clustering to automatically discover content themes without predefined categories
4. **Enable Better Onboarding**: Provide new users with diverse, high-quality content recommendations based on their interests

## How It Works

### Phase 1: User Quality Scoring
- Samples diverse users and labels them using LLM-based quality assessment
- Extracts behavioral features (post frequency, engagement ratios, spam indicators)
- Trains ML models to predict user quality scores
- Applies the model to score all users in the network

### Phase 2: Content Filtering
- Exports engagement data efficiently using DuckDB
- Filters content based on:
  - Author quality scores (≥4 to exclude spammers)
  - Quality user engagement (reactions/replies from users with score ≥7)
  - Content length and substance requirements
  - Only top-level posts (no replies)
  - At least 1 reply required
  - Extensive spam filtering (crypto, eggs/hens, gambling, engagement farming)
- Ensures diversity by limiting posts per author

### Phase 3: Content Clustering
- Generates semantic embeddings using BAAI/bge-large-en-v1.5
- Clusters content into 50 fine-grained topics
- Uses Gemini API (gemini-2.5-flash-lite-preview-06-17) for intelligent cluster naming
- Extracts keywords using TF-IDF analysis
- Parallel processing for faster analysis
- Human-friendly cluster names that people can identify with
- All categories emerge naturally from the data - no hardcoded categories

## Data Sources

- **Casts Dataset**: https://huggingface.co/datasets/jc4p/farcaster-casts
- **Reactions Dataset**: https://huggingface.co/datasets/jc4p/farcaster-reactions

Download these datasets and place them in the `data/` directory as:
- `data/casts.parquet`
- `data/farcaster_reactions.parquet`

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models

# Set environment variables
export GEMINI_API_KEY="your-gemini-api-key"  # Required for cluster naming
```

## Order of Operations

### 1. Sample Users for Labeling
```bash
python sample_users_for_labeling.py
```
Samples 10,000 diverse users across different activity levels for labeling.

### 2. Label Users with Quality Scores
```bash
python label_users_with_gemini.py
# OR for local labeling:
python label_users_locally.py
```
Uses LLM to assess user quality based on their posts and engagement patterns.

### 3. Extract User Features
```bash
python extract_user_features.py
```
Extracts behavioral features for all users (post frequency, engagement ratios, etc.).

### 4. Train User Quality Classifier
```bash
python train_user_classifier.py
```
Trains XGBoost and Random Forest models on labeled data to predict user quality.

### 5. Score All Users
```bash
python score_all_users.py
```
Applies the trained model to score all users in the network.

### 6. Export Quality Cast Data
```bash
./export_quality_data.sh
```
**NEW STEP**: Uses DuckDB CLI to efficiently export filtered casts with engagement metrics.
This step optimizes RAM usage and reduces CPU bottleneck during data retrieval.

### 7. Filter Quality Content
```bash
python filter_quality_content.py
```
Loads pre-exported data and filters for high-quality, diverse content.

### 8. Cluster Quality Content
```bash
python cluster_quality_content.py
```
Clusters filtered content into topics using semantic embeddings and discovers categories from the data.

## Output Files

### User Quality Data
- `data/user_quality_scores.parquet` - Quality scores for all users
- `data/labeled_users.parquet` - Manually labeled training data
- `models/user_quality_model.pkl` - Trained classifier model

### Content Data
- `data/quality_casts_raw.parquet` - Pre-exported casts with engagement metrics
- `data/high_quality_casts.parquet` - Filtered high-quality content
- `data/high_quality_casts_clustered.parquet` - Content with cluster assignments

### Clustering Results
- `data/quality_clusters.json` - Detailed information about each cluster with Gemini-generated names
- `data/clustering_summary.json` - Summary statistics
- `models/quality_kmeans.pkl` - Trained clustering model
- `recommendations_output.json` - User recommendations including content and users to follow

## Key Features

- **No Hardcoded Categories**: All content categories emerge naturally from clustering
- **Intelligent Cluster Naming**: Uses Gemini API to generate human-friendly cluster names
- **Quality-Weighted Engagement**: Prioritizes engagement from high-quality users
- **Comprehensive Spam Filtering**: Removes crypto spam, engagement farming, and low-quality content
- **Top-Level Posts Only**: Focuses on original content, not replies
- **Author Diversity**: Limits posts per author to ensure diverse perspectives
- **User Recommendations**: Suggests both content and users to follow based on preferences
- **Efficient Data Processing**: Uses DuckDB for optimized data retrieval
- **Parallel Processing**: Multi-threaded cluster analysis for faster performance
- **Semantic Understanding**: Uses state-of-the-art embeddings for content similarity

## Performance Considerations

- The export step (`export_quality_data.sh`) handles large-scale data aggregation efficiently
- Embedding generation uses GPU acceleration when available
- Clustering uses MiniBatchKMeans for memory efficiency
- All data is stored in Parquet format for optimal performance

## Running the Recommendation System

After clustering, you can run the interactive recommendation system:

```bash
python recommend_content.py
```

This will:
1. Show you 3-5 posts to gauge your interests
2. Learn your preferences using binary search through clusters
3. Recommend 10 personalized posts
4. Suggest 10 users to follow based on your interests

## Filtered Content

The pipeline filters out:
- Crypto/token/NFT/DeFi spam
- Gambling content (spins, slots)
- Egg/hen/cock spam patterns
- Engagement farming (follow for follow, etc.)
- Platform meta-discussions (Farcaster, Warpcast, Neynar, FarCon)
- Low-effort greetings (gm, gn)
- Posts with no replies
- Reply posts (only shows top-level content)

## Next Steps

The output can be used for:
- Personalized content recommendations for new users
- Topic-based content discovery
- Understanding community interests and trends
- Building curated feeds based on cluster preferences
- Identifying quality authors in specific niches