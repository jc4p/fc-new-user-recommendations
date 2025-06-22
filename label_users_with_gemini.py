#!/usr/bin/env python3
"""
Label Farcaster users for quality using Gemini with structured outputs
"""
import json
import os
from typing import List
from pydantic import BaseModel, Field
from google import genai
from tqdm import tqdm
import time

class UserQualityRating(BaseModel):
    fid: int = Field(description="Farcaster user ID")
    quality_score: float = Field(description="Quality score from 0-10", ge=0, le=10)
    category: str = Field(description="Category: Spammer, Low, Average, High, Leader")
    reasoning: str = Field(description="Brief explanation for the rating")
    content_themes: List[str] = Field(description="Main topics this user posts about")

class BatchUserRatings(BaseModel):
    ratings: List[UserQualityRating]

def create_batch_prompt(users_batch):
    """Create prompt for rating multiple users at once"""
    
    users_text = []
    for user in users_batch:
        posts_preview = "\n".join([f"- {post[:150]}..." if len(post) > 150 else f"- {post}" 
                                  for post in user['posts'][:30]])  # Show up to 30 posts
        
        user_section = f"""
USER FID: {user['fid']}
Total posts: {user['post_count']}
Average post length: {user['avg_text_length']:.0f} characters
GM/GN ratio: {user['gm_ratio']:.2%}
Question ratio: {user['question_ratio']:.2%}

Recent posts:
{posts_preview}
"""
        users_text.append(user_section)
    
    prompt = f"""Analyze these Farcaster users and rate their content quality.

For each user, provide:
1. A quality score from 0-10:
   - 0-2: Spammer (mostly GM/GN, airdrops, repetitive low-effort posts)
   - 3-4: Low quality (minimal substance, very short posts, little engagement value)
   - 5-6: Average (some interesting content mixed with low-effort posts)
   - 7-8: High quality (consistently thoughtful, informative, or entertaining)
   - 9-10: Thought leader (exceptionally insightful, original content, drives discussions)

2. A category based on the score: "Spammer", "Low", "Average", "High", or "Leader"

3. Brief reasoning for your rating

4. Main content themes they post about (e.g., "crypto", "food", "sports", "technology", "personal", etc.)

Consider:
- Post substance and originality
- Engagement value (do posts invite discussion?)
- Variety vs repetition
- Effort level
- Value to the community

{'='*80}
{('='*80).join(users_text)}
{'='*80}

Rate each user based on their overall posting pattern, not just individual posts."""
    
    return prompt

def process_users_batch(client, users_batch, model="gemini-2.5-flash-lite-preview-06-17"):
    """Process a batch of users with Gemini"""
    
    prompt = create_batch_prompt(users_batch)
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": BatchUserRatings,
                "temperature": 0.3,  # Lower temperature for more consistent ratings
                "max_output_tokens": 16384,  # Increase output limit (default is often lower)
            },
        )
        
        if response.parsed:
            return response.parsed.ratings
        else:
            print(f"Failed to parse response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

def main():
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Load users to label
    with open('data/users_to_label.json', 'r') as f:
        users_data = json.load(f)
    
    print(f"Loaded {len(users_data)} users to label")
    
    # Process in batches to maximize context window usage
    batch_size = 25
    all_ratings = []
    
    print(f"\nProcessing users in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(users_data), batch_size)):
        batch = users_data[i:i+batch_size]
        
        # Process batch
        ratings = process_users_batch(client, batch)
        
        if ratings:
            all_ratings.extend(ratings)
            print(f"  Batch {i//batch_size + 1}: Labeled {len(ratings)} users")
        else:
            print(f"  Batch {i//batch_size + 1}: Failed")
        
        # Rate limiting - Gemini has generous limits but let's be respectful
        if i + batch_size < len(users_data):
            time.sleep(1)
    
    print(f"\nTotal users labeled: {len(all_ratings)}")
    
    # Convert to dictionary format and save
    labeled_data = []
    for rating in all_ratings:
        labeled_data.append({
            'fid': rating.fid,
            'quality_score': rating.quality_score,
            'category': rating.category,
            'reasoning': rating.reasoning,
            'content_themes': rating.content_themes
        })
    
    # Save labeled data
    with open('data/labeled_users.json', 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    print(f"Saved labeled data to data/labeled_users.json")
    
    # Print summary statistics
    if labeled_data:
        scores = [u['quality_score'] for u in labeled_data]
        categories = [u['category'] for u in labeled_data]
        
        print("\nLabeling summary:")
        print(f"Average quality score: {sum(scores)/len(scores):.2f}")
        print("\nCategory distribution:")
        for cat in ["Spammer", "Low", "Average", "High", "Leader"]:
            count = categories.count(cat)
            print(f"  {cat}: {count} ({count/len(categories)*100:.1f}%)")
        
        # Show examples from each category
        print("\nExample users from each category:")
        for cat in ["Spammer", "Low", "Average", "High", "Leader"]:
            examples = [u for u in labeled_data if u['category'] == cat]
            if examples:
                example = examples[0]
                print(f"\n{cat} (score: {example['quality_score']}):")
                print(f"  FID: {example['fid']}")
                print(f"  Themes: {', '.join(example['content_themes'])}")
                print(f"  Reasoning: {example['reasoning']}")

if __name__ == "__main__":
    main()
