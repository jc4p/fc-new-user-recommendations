#!/usr/bin/env python3
"""
Generate embeddings for all cast texts using sentence-transformers on GPU
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

def main():
    # Configuration
    input_file = "data/filtered_casts.parquet"
    output_file = "data/cast_embeddings.parquet"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 2048  # Large batch size for H100
    chunk_size = 10000  # Read data in chunks
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load model on GPU
    print(f"\nLoading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    
    # Connect to DuckDB and count total rows
    conn = duckdb.connect()
    total_rows = conn.execute(f'SELECT COUNT(*) FROM "{input_file}"').fetchone()[0]
    print(f"\nTotal casts to process: {total_rows:,}")
    
    # Prepare output data
    all_data = {
        'hash_hex': [],
        'fid': [],
        'text': [],
        'embedding': []
    }
    
    # Process data in chunks
    start_time = time.time()
    processed = 0
    
    with tqdm(total=total_rows, desc="Generating embeddings") as pbar:
        # Read data in chunks
        offset = 0
        while offset < total_rows:
            # Query chunk
            query = f"""
            SELECT Hash, Fid, Text 
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
            
            # Extract texts
            texts = chunk_df['Text'].tolist()
            
            # Generate embeddings in batches
            chunk_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Generate embeddings with GPU acceleration
                with torch.no_grad():
                    batch_embeddings = model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        normalize_embeddings=True  # Normalize for better similarity search
                    )
                    # Convert to numpy and move to CPU
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    chunk_embeddings.extend(batch_embeddings)
            
            # Add to results
            all_data['hash_hex'].extend(hex_hashes)
            # Convert FID to int, handling potential string values
            fids = []
            for fid in chunk_df['Fid'].tolist():
                try:
                    fids.append(int(fid))
                except (ValueError, TypeError):
                    fids.append(0)  # Default value for invalid FIDs
            all_data['fid'].extend(fids)
            all_data['text'].extend(texts)
            all_data['embedding'].extend(chunk_embeddings)
            
            # Update progress
            processed += len(chunk_df)
            pbar.update(len(chunk_df))
            
            # Print stats
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'rate': f'{rate:.0f} casts/s',
                'GPU mem': f'{gpu_memory:.1f}GB'
            })
            
            offset += chunk_size
    
    # Create DataFrame and save to parquet
    print("\nCreating output DataFrame...")
    result_df = pd.DataFrame({
        'hash_hex': all_data['hash_hex'],
        'fid': all_data['fid'],
        'text': all_data['text'],
        'embedding': all_data['embedding']
    })
    
    # Save to parquet with appropriate schema
    print(f"Saving to {output_file}...")
    # Convert embeddings to fixed-size array for better parquet compatibility
    embedding_dim = len(all_data['embedding'][0])
    schema = pa.schema([
        ('hash_hex', pa.string()),
        ('fid', pa.int64()),
        ('text', pa.string()),
        ('embedding', pa.list_(pa.float32(), list_size=embedding_dim))
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

if __name__ == "__main__":
    import os
    main()