#!/usr/bin/env python3
"""
Generate high-quality embeddings using BGE-large-en-v1.5 for crypto social media clustering
Optimized for H100 GPU with mixed precision and large batch sizes
"""
import time
import base64
import torch
import duckdb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

def main():
    # Configuration
    input_file = "data/filtered_casts.parquet"
    output_file = "data/cast_embeddings_bge.parquet"
    model_name = "BAAI/bge-large-en-v1.5"
    batch_size = 1024  # Large batch for H100
    chunk_size = 20000  # Larger chunks since we have more data
    
    print("="*80)
    print("BGE-Large Embeddings Generation for Crypto Social Media")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Enable TF32 for better performance on H100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load model on GPU with mixed precision
    print(f"\nLoading model: {model_name}")
    print("This is a 335M parameter model optimized for clustering...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    
    # Enable half precision for faster inference
    if torch.cuda.is_available():
        model = model.half()
    
    # Connect to DuckDB and count total rows
    conn = duckdb.connect()
    total_rows = conn.execute(f'SELECT COUNT(*) FROM "{input_file}"').fetchone()[0]
    print(f"\nTotal casts to process: {total_rows:,}")
    
    # Prepare output data
    all_data = {
        'hash_hex': [],
        'fid': [],
        'text': [],
        'embedding': [],
        'reaction_count': [],
        'reply_count': []
    }
    
    # Process data in chunks
    start_time = time.time()
    processed = 0
    
    with tqdm(total=total_rows, desc="Generating BGE embeddings") as pbar:
        # Read data in chunks
        offset = 0
        while offset < total_rows:
            # Query chunk with engagement metrics
            query = f"""
            SELECT Hash, Fid, Text, reaction_count, reply_count
            FROM "{input_file}" 
            WHERE Text IS NOT NULL AND Text != ''
            LIMIT {chunk_size} OFFSET {offset}
            """
            chunk_df = conn.execute(query).fetch_df()
            
            if len(chunk_df) == 0:
                break
            
            # Convert base64 hashes to hex
            hex_hashes = []
            for hash_b64 in chunk_df['Hash']:
                try:
                    decoded = base64.b64decode(hash_b64)
                    hex_hash = decoded.hex()
                    hex_hashes.append(hex_hash)
                except Exception as e:
                    hex_hashes.append(hash_b64)  # Keep original if conversion fails
            
            # Extract texts and prepare for BGE (add instruction prefix)
            texts = chunk_df['Text'].tolist()
            # BGE works better with instruction prefix for retrieval/clustering
            texts_with_instruction = [f"Represent this social media post for clustering: {text}" for text in texts]
            
            # Generate embeddings in batches
            chunk_embeddings = []
            for i in range(0, len(texts_with_instruction), batch_size):
                batch_texts = texts_with_instruction[i:i+batch_size]
                
                # Generate embeddings with GPU acceleration
                with torch.no_grad():
                    with torch.cuda.amp.autocast():  # Mixed precision
                        batch_embeddings = model.encode(
                            batch_texts,
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            normalize_embeddings=True,  # Important for clustering
                            batch_size=batch_size
                        )
                    # Convert to numpy and move to CPU
                    batch_embeddings = batch_embeddings.float().cpu().numpy()
                    chunk_embeddings.extend(batch_embeddings)
            
            # Convert FIDs to int
            fids = []
            for fid in chunk_df['Fid'].tolist():
                try:
                    fids.append(int(fid))
                except (ValueError, TypeError):
                    fids.append(0)
            
            # Add to results
            all_data['hash_hex'].extend(hex_hashes)
            all_data['fid'].extend(fids)
            all_data['text'].extend(texts)
            all_data['embedding'].extend(chunk_embeddings)
            all_data['reaction_count'].extend(chunk_df['reaction_count'].tolist())
            all_data['reply_count'].extend(chunk_df['reply_count'].tolist())
            
            # Update progress
            processed += len(chunk_df)
            pbar.update(len(chunk_df))
            
            # Print stats
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            gpu_util = torch.cuda.utilization() if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'rate': f'{rate:.0f} casts/s',
                'GPU mem': f'{gpu_memory:.1f}GB',
                'GPU util': f'{gpu_util}%'
            })
            
            offset += chunk_size
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and processed % 50000 == 0:
                torch.cuda.empty_cache()
    
    # Create DataFrame and save to parquet
    print("\nCreating output DataFrame...")
    result_df = pd.DataFrame({
        'hash_hex': all_data['hash_hex'],
        'fid': all_data['fid'],
        'text': all_data['text'],
        'embedding': all_data['embedding'],
        'reaction_count': all_data['reaction_count'],
        'reply_count': all_data['reply_count']
    })
    
    # Save to parquet with appropriate schema
    print(f"Saving to {output_file}...")
    # BGE-large produces 1024-dim embeddings
    embedding_dim = len(all_data['embedding'][0])
    print(f"Embedding dimension: {embedding_dim}")
    
    schema = pa.schema([
        ('hash_hex', pa.string()),
        ('fid', pa.int64()),
        ('text', pa.string()),
        ('embedding', pa.list_(pa.float32(), list_size=embedding_dim)),
        ('reaction_count', pa.int64()),
        ('reply_count', pa.int64())
    ])
    
    table = pa.Table.from_pandas(result_df, schema=schema)
    pq.write_table(table, output_file, compression='snappy')
    
    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nCompleted!")
    print(f"Total casts processed: {processed:,}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average rate: {processed/total_time:.0f} casts/second")
    print(f"Output saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024**2:.1f} MB")
    
    # Quick quality check
    print("\nQuick embedding quality check:")
    sample_embeddings = np.array(all_data['embedding'][:100])
    print(f"Embedding stats - Mean: {np.mean(sample_embeddings):.4f}, Std: {np.std(sample_embeddings):.4f}")

if __name__ == "__main__":
    import os
    main()