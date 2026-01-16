import argparse
import os
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import fsspec
import json
import shutil
import duckdb

def _sanitize_storage_options(fs_args_str: str | None) -> dict:
    storage_options = json.loads(fs_args_str) if fs_args_str else {}
    if "AWS_ENDPOINT_URL" in storage_options:
        client_kwargs = storage_options.get("client_kwargs") or {}
        if isinstance(client_kwargs, dict) and "endpoint_url" not in client_kwargs:
            client_kwargs["endpoint_url"] = storage_options["AWS_ENDPOINT_URL"]
            storage_options["client_kwargs"] = client_kwargs
    if "AWS_ACCESS_KEY_ID" in storage_options and "key" not in storage_options:
        storage_options["key"] = storage_options["AWS_ACCESS_KEY_ID"]
    if "AWS_SECRET_ACCESS_KEY" in storage_options and "secret" not in storage_options:
        storage_options["secret"] = storage_options["AWS_SECRET_ACCESS_KEY"]
    if "AWS_DEFAULT_REGION" in storage_options:
        client_kwargs = storage_options.get("client_kwargs") or {}
        if isinstance(client_kwargs, dict) and "region_name" not in client_kwargs:
            client_kwargs["region_name"] = storage_options["AWS_DEFAULT_REGION"]
            storage_options["client_kwargs"] = client_kwargs

    allowed = {
        "key",
        "secret",
        "token",
        "client_kwargs",
        "config_kwargs",
        "use_ssl",
        "anon",
    }
    return {k: v for k, v in storage_options.items() if k in allowed}


def tile_raster(input_file, output_dir, tile_size=256, overlap=0, fs_args_str=None):
    """
    Splits a raster into tiles of given size and saves them as separate files.
    Calculates vector footprints for index.parquet.
    Supports S3 via fsspec (simple download/upload strategy).
    """
    dataset_name = os.path.basename(input_file).split('.')[0]
    
    # Configure filesystem
    storage_options = _sanitize_storage_options(fs_args_str)
    
    # Handle Input (Download if S3)
    local_input_path = input_file
    if input_file.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
        local_input_path = "/tmp/input_raster.tif"
        print(f"Downloading {input_file} to {local_input_path}...")
        fs.get(input_file, local_input_path)
    
    # Handle Output (Local temp dir)
    dataset_name = os.path.basename(local_input_path).split('.')[0] # Update name based on local file
    local_output_dir = f"/tmp/tiles_{dataset_name}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    tile_records = []
    
    with rasterio.open(local_input_path) as src:
        width = src.width
        height = src.height
        
        # Simple loop for tiling without overlap logic for now
        for col in range(0, width, tile_size):
            for row in range(0, height, tile_size):
                window = Window(col, row, tile_size, tile_size)
                transform = src.window_transform(window)
                
                # Check bounds
                if col + tile_size > width or row + tile_size > height:
                    # Logic for partial tiles: skip or pad? Let's skip small edge tiles for now or user can config
                    continue

                tile_data = src.read(window=window)
                
                # Create output filename
                tile_id = f"{dataset_name}_x{col}_y{row}"
                tile_filename = f"{tile_id}.tif"
                # Save to local temp dir first
                tile_path = os.path.join(local_output_dir, tile_filename)
                
                # Save tile
                kwargs = src.meta.copy()
                kwargs.update({
                    'driver': 'GTiff',
                    'height': window.height,
                    'width': window.width,
                    'transform': transform
                })
                
                with rasterio.open(tile_path, 'w', **kwargs) as dst:
                    dst.write(tile_data)
                
                # Metadata for index
                bounds = rasterio.windows.bounds(window, src.transform)
                tile_records.append({
                    'dataset_id': dataset_name,
                    'tile_id': tile_id,
                    'min_x': bounds[0],
                    'min_y': bounds[1],
                    'max_x': bounds[2],
                    'max_y': bounds[3],
                    'crs': str(src.crs),
                    'tile_path': tile_filename # Relative to output_dir
                })
                
    # Save index
    df = pd.DataFrame(tile_records)
    index_path = os.path.join(local_output_dir, 'index.parquet')
    try:
        df.to_parquet(index_path, index=False)
    except Exception as e:
        # analytics-embeddings image may not have a parquet engine (pyarrow/fastparquet).
        # DuckDB can always write parquet without extra deps.
        print(f"pandas.to_parquet failed ({type(e).__name__}: {e}); falling back to DuckDB COPY...")
        con = duckdb.connect()
        con.register("tiles_index", df)
        safe_path = index_path.replace("'", "''")
        con.execute(f"COPY tiles_index TO '{safe_path}' (FORMAT PARQUET)")
        con.close()
    print(f"Created {len(tile_records)} tiles and index at {index_path}")

    # Upload to S3 if needed
    if output_dir.startswith("s3://"):
        print(f"Uploading tiles to {output_dir}...")
        fs = fsspec.filesystem("s3", **storage_options)
        fs.put(local_output_dir, output_dir, recursive=True)
        print("Upload complete.")
    else:
        # Move keys if strictly local
        if output_dir != local_output_dir:
            shutil.copytree(local_output_dir, output_dir, dirs_exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    args = parser.parse_args()
    
    tile_raster(args.input, args.output_dir, tile_size=args.tile_size, fs_args_str=args.fs_args)
