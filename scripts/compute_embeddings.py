import os
import argparse
import pandas as pd
import torch
import rasterio
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import duckdb
import numpy as np

import json
import fsspec

# Using DINOv2 as a placeholder/proxy for DINOv3 API calls, 
# assuming similar transformers/timm usage.
MODEL_NAME = "facebook/dinov2-small" 

def compute_embeddings(tiles_dir, output_dir, fs_args_str=None):
    """
    Reads tiles based on index.parquet in tiles_dir.
    Computes embeddings.
    Saves to parquet in output_dir.
    """
    # Configure filesystem
    storage_options = json.loads(fs_args_str) if fs_args_str else {}
    if 'AWS_ENDPOINT_URL' in storage_options:
        storage_options['endpoint_url'] = storage_options['AWS_ENDPOINT_URL']

    # Read Index
    index_path = os.path.join(tiles_dir, "index.parquet")
    print(f"Reading index from {index_path}")
    
    try:
        df = pd.read_parquet(index_path, storage_options=storage_options)
    except Exception as e:
        print(f"Error reading index: {e}")
        return

    print(f"Processing {len(df)} tiles...")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    embeddings = []
    
    fs = fsspec.filesystem("s3", **storage_options) if tiles_dir.startswith("s3://") else None

    for _, row in df.iterrows():
        tile_filename = row['tile_path']
        tile_path = os.path.join(tiles_dir, tile_filename)
        
        # Download tile if S3
        local_tile_path = tile_filename
        if tiles_dir.startswith("s3://"):
            local_tile_path = f"/tmp/{tile_filename}"
            # ensure dir exists
            os.makedirs(os.path.dirname(local_tile_path), exist_ok=True)
            if not os.path.exists(local_tile_path):
                 fs.get(tile_path, local_tile_path)
        else:
            local_tile_path = tile_path

        # Open raster with rasterio
        with rasterio.open(local_tile_path) as src:
            # Assume 3 bands RGB for now, or take first 3
            # If single band, repeat?
            img_array = src.read()
            # Normalize/Transpose for simple PIL creation: (C, H, W) -> (H, W, C)
            img_array = np.moveaxis(img_array, 0, -1)
            
            # Handle non-uint8 or normalization if needed.
            # Simple assumption: Input is valid image-like
            if img_array.dtype != np.uint8:
                 # simplistic normalization to 0-255 uint8 for vision model input
                 img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Create PIL
            image = Image.fromarray(img_array)
            
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding or mean pooling
            # DINOv2 typically uses the first token for global representation
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states[0, 0, :].cpu().numpy().tolist()
            
        embeddings.append(embedding)
        
    df['embedding'] = embeddings
    
    # Save Output
    out_path = os.path.join(output_dir, "embeddings.parquet")
    print(f"Saving to {out_path}...")
    df.to_parquet(out_path, storage_options=storage_options)
    print(f"Saved embeddings to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    args = parser.parse_args()
    
    compute_embeddings(args.tiles_dir, args.output_dir, args.fs_args)
