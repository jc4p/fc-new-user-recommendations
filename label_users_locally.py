#!/usr/bin/env python3
"""
Label Farcaster users for quality using local Gemma-3-12B-IT model
"""
import json
import os
from typing import List
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import time

def create_user_prompt(user):
    """Create prompt for rating a single user"""
    
    posts_preview = "\n".join([f"- {post[:150]}..." if len(post) > 150 else f"- {post}" 
                              for post in user['posts'][:30]])  # Show up to 30 posts
    
    prompt = f"""Analyze this Farcaster user and rate their content quality.

USER FID: {user['fid']}
Total posts: {user['post_count']}
Average post length: {user['avg_text_length']:.0f} characters
GM/GN ratio: {user['gm_ratio']:.2%}
Question ratio: {user['question_ratio']:.2%}

Recent posts:
{posts_preview}

Rate this user's content quality from 0-10:
- 0-2: Spammer (mostly GM/GN, airdrops, repetitive low-effort posts)
- 3-4: Low quality (minimal substance, very short posts, little engagement value)
- 5-6: Average (some interesting content mixed with low-effort posts)
- 7-8: High quality (consistently thoughtful, informative, or entertaining)
- 9-10: Thought leader (exceptionally insightful, original content, drives discussions)

Provide your response in this exact format:
SCORE: [number]
CATEGORY: [Spammer|Low|Average|High|Leader]
THEMES: [comma-separated list of content themes]
REASONING: [1-2 sentence explanation]"""
    
    return prompt

def parse_gemma_response(response_text):
    """Parse the structured response from Gemma"""
    try:
        lines = response_text.strip().split('\n')
        
        score = None
        category = None
        themes = []
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('SCORE:'):
                score_text = line.replace('SCORE:', '').strip()
                # Extract number from text like "7" or "7/10"
                score = float(score_text.split('/')[0].split()[0])
            elif line.startswith('CATEGORY:'):
                category = line.replace('CATEGORY:', '').strip()
            elif line.startswith('THEMES:'):
                themes_text = line.replace('THEMES:', '').strip()
                themes = [t.strip() for t in themes_text.split(',')]
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        # Validate and fix category based on score
        if score is not None:
            if score <= 2.5:
                category = "Spammer"
            elif score <= 4.5:
                category = "Low"
            elif score <= 6.5:
                category = "Average"
            elif score <= 8.5:
                category = "High"
            else:
                category = "Leader"
        
        return {
            'quality_score': score if score is not None else 5.0,
            'category': category if category else "Average",
            'content_themes': themes if themes else ["general"],
            'reasoning': reasoning if reasoning else "Unable to parse response"
        }
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response was: {response_text[:200]}...")
        return {
            'quality_score': 5.0,
            'category': "Average",
            'content_themes': ["general"],
            'reasoning': "Error parsing model response"
        }

def process_users_with_gemma(model, processor, users_batch, device, progress_callback=None):
    """Process a batch of users with local Gemma model - now with true batching!"""
    
    results = []
    
    # Process multiple users at once
    batch_size = 24  # Process 16 users simultaneously - doubled for H100
    
    for i in range(0, len(users_batch), batch_size):
        batch_users = users_batch[i:i+batch_size]
        
        # Prepare all prompts
        all_messages = []
        for user in batch_users:
            prompt = create_user_prompt(user)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert at analyzing social media content quality."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            all_messages.append(messages)
        
        try:
            # Tokenize all inputs
            all_inputs = []
            input_lengths = []
            
            for messages in all_messages:
                inputs = processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True,
                    return_dict=True, 
                    return_tensors="pt"
                )
                all_inputs.append(inputs)
                input_lengths.append(inputs["input_ids"].shape[-1])
            
            # Pad inputs to same length for batching
            max_length = max(inp["input_ids"].shape[-1] for inp in all_inputs)
            
            # Create batched tensors
            input_ids_list = []
            attention_mask_list = []
            
            for inp in all_inputs:
                # Pad input_ids
                pad_length = max_length - inp["input_ids"].shape[-1]
                if pad_length > 0:
                    padded_ids = torch.nn.functional.pad(
                        inp["input_ids"], 
                        (0, pad_length), 
                        value=processor.tokenizer.pad_token_id
                    )
                    padded_mask = torch.nn.functional.pad(
                        inp["attention_mask"], 
                        (0, pad_length), 
                        value=0
                    )
                else:
                    padded_ids = inp["input_ids"]
                    padded_mask = inp["attention_mask"]
                
                input_ids_list.append(padded_ids)
                attention_mask_list.append(padded_mask)
            
            # Stack into batch
            batched_input_ids = torch.cat(input_ids_list, dim=0).to(device, dtype=torch.long)
            batched_attention_mask = torch.cat(attention_mask_list, dim=0).to(device, dtype=torch.long)
            
            # Generate responses for all users in batch
            with torch.inference_mode():
                generations = model.generate(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode each response
            for idx, (user, input_len) in enumerate(zip(batch_users, input_lengths)):
                generation = generations[idx][input_len:]
                response = processor.decode(generation, skip_special_tokens=True)
                
                # Parse response
                parsed = parse_gemma_response(response)
                parsed['fid'] = user['fid']
                results.append(parsed)
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(len(batch_users))
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Fallback to individual processing
            for user in batch_users:
                results.append({
                    'fid': user['fid'],
                    'quality_score': 5.0,
                    'category': "Average",
                    'content_themes': ["general"],
                    'reasoning': f"Error during batch processing: {str(e)}"
                })
        
        # Clear cache periodically
        if len(results) % 100 == 0:
            torch.cuda.empty_cache()
    
    return results

def main():
    # Check for HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Please set HF_TOKEN environment variable")
        return
    
    print("Loading Gemma-3-12B-IT model...")
    model_id = "google/gemma-3-12b-it"
    
    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    
    print("Model loaded successfully!")
    
    # Load users to label
    with open('data/users_to_label.json', 'r') as f:
        users_data = json.load(f)
    
    print(f"Loaded {len(users_data)} users to label")
    
    # Process all users with single progress bar
    all_ratings = []
    
    print(f"\nProcessing {len(users_data)} users...")
    start_time = time.time()
    
    # Single progress bar for all users
    with tqdm(total=len(users_data), desc="Users processed") as pbar:
        # Process in chunks for memory management
        chunk_size = 200
        for i in range(0, len(users_data), chunk_size):
            chunk = users_data[i:i+chunk_size]
            
            # Process chunk with progress callback
            def update_progress(n):
                pbar.update(n)
                # Update stats
                elapsed = time.time() - start_time
                current_total = len(all_ratings) + n
                users_per_second = current_total / elapsed if elapsed > 0 else 0
                eta = (len(users_data) - current_total) / users_per_second if users_per_second > 0 else 0
                
                pbar.set_postfix({
                    'speed': f'{users_per_second:.1f} u/s',
                    'ETA': f'{eta/60:.1f}m',
                    'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
            
            chunk_results = process_users_with_gemma(model, processor, chunk, device, update_progress)
            all_ratings.extend(chunk_results)
            
            # Show stats
            elapsed = time.time() - start_time
            users_per_second = len(all_ratings) / elapsed if elapsed > 0 else 0
            eta = (len(users_data) - len(all_ratings)) / users_per_second if users_per_second > 0 else 0
            
            pbar.set_postfix({
                'speed': f'{users_per_second:.1f} u/s',
                'ETA': f'{eta/60:.1f}m',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
    
    print(f"\nTotal users labeled: {len(all_ratings)}")
    
    # Save labeled data
    with open('data/labeled_users_local.json', 'w') as f:
        json.dump(all_ratings, f, indent=2)
    
    print(f"Saved labeled data to data/labeled_users_local.json")
    
    # Print summary statistics
    if all_ratings:
        scores = [u['quality_score'] for u in all_ratings]
        categories = [u['category'] for u in all_ratings]
        
        print("\nLabeling summary:")
        print(f"Average quality score: {sum(scores)/len(scores):.2f}")
        print("\nCategory distribution:")
        for cat in ["Spammer", "Low", "Average", "High", "Leader"]:
            count = categories.count(cat)
            print(f"  {cat}: {count} ({count/len(categories)*100:.1f}%)")
        
        # Show examples from each category
        print("\nExample users from each category:")
        for cat in ["Spammer", "Low", "Average", "High", "Leader"]:
            examples = [u for u in all_ratings if u['category'] == cat]
            if examples:
                example = examples[0]
                print(f"\n{cat} (score: {example['quality_score']}):")
                print(f"  FID: {example['fid']}")
                print(f"  Themes: {', '.join(example['content_themes'])}")
                print(f"  Reasoning: {example['reasoning']}")
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")
    print(f"Average speed: {len(all_ratings)/total_time:.1f} users/second")

if __name__ == "__main__":
    main()
