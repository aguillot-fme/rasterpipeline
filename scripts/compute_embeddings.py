import os
import argparse
import pandas as pd

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None
import numpy as np

import json
import fsspec
import posixpath
from urllib.parse import urlparse

# Keep duckdb optional; it's only needed for parquet fallbacks.
try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

# Keep transformers optional so the script can run offline with local DINOv3 weights.
try:
    from transformers import AutoImageProcessor, AutoModel  # type: ignore
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None

# Using DINOv2 as a placeholder/proxy for DINOv3 API calls, 
# assuming similar transformers/timm usage.
MODEL_NAME = "facebook/dinov2-small" 

def _join_uri(base: str, *parts: str) -> str:
    if base.startswith("s3://"):
        parsed = urlparse(base)
        prefix = parsed.path.lstrip("/")
        joined = posixpath.join(prefix, *[p.strip("/") for p in parts if p])
        return f"s3://{parsed.netloc}/{joined}"
    return os.path.join(base, *parts)


def _read_parquet(path: str, storage_options: dict) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, storage_options=storage_options)
    except Exception as e:
        # analytics-embeddings image may not have pyarrow/fastparquet installed.
        # DuckDB can read parquet without extra deps.
        print(f"pandas.read_parquet failed ({type(e).__name__}: {e}); falling back to DuckDB read_parquet...")
        return _read_parquet_via_duckdb(path)


def _read_parquet_via_duckdb(path: str) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("duckdb is required for parquet fallback reads")
    con = duckdb.connect()
    safe_path = path.replace("'", "''")
    df = con.execute(f"SELECT * FROM read_parquet('{safe_path}')").df()
    con.close()
    return df


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
        return
    except Exception as e:
        print(f"pandas.to_parquet failed ({type(e).__name__}: {e}); falling back to DuckDB COPY...")
        if duckdb is None:
            raise RuntimeError("duckdb is required for parquet fallback writes") from e
        con = duckdb.connect()
        con.register("embeddings_df", df)
        safe_path = path.replace("'", "''")
        con.execute(f"COPY embeddings_df TO '{safe_path}' (FORMAT PARQUET)")
        con.close()


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    if torch is None:
        raise RuntimeError("torch is required to load local model checkpoints")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    # Common prefixes from training frameworks.
    cleaned = {}
    for k, v in ckpt.items():
        if not isinstance(k, str):
            continue
        for prefix in ("module.", "model.", "backbone.", "teacher.", "student."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def _infer_vit_img_size_from_pos_embed(state_dict: dict, patch_size: int = 16, default: int = 224) -> int:
    pos = state_dict.get("pos_embed")
    if pos is None or not hasattr(pos, "shape") or len(pos.shape) < 2:
        return default
    n_tokens = int(pos.shape[1])
    if n_tokens <= 1:
        return default
    grid = int(round((n_tokens - 1) ** 0.5))
    if grid * grid != (n_tokens - 1):
        return default
    return grid * patch_size


def _load_dinov3_timm_model(model_path: str, device: torch.device):
    # Imported lazily so unit tests that only mock the HF path don't need these deps installed.
    import timm  # type: ignore

    state_dict = _load_checkpoint_state_dict(model_path)
    img_size = _infer_vit_img_size_from_pos_embed(state_dict, patch_size=16, default=224)

    model = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=img_size, num_classes=0)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[dinov3] Missing keys (showing first 20): {missing[:20]}")
    if unexpected:
        print(f"[dinov3] Unexpected keys (showing first 20): {unexpected[:20]}")

    model.eval().to(device)
    return model, img_size


def _make_vit_preprocess(img_size: int):
    from torchvision import transforms  # type: ignore

    # Standard ImageNet normalization used by ViT/DINO-style models.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _extract_embedding_vit(model, pixel_values: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            feats = model.forward_features(pixel_values)
            if isinstance(feats, dict) and "x" in feats:
                feats = feats["x"]
        else:
            feats = model(pixel_values)

        # timm ViT typically returns either (B, tokens, dim) or (B, dim)
        if hasattr(feats, "ndim") and feats.ndim == 3:
            emb = feats[:, 0, :]
        else:
            emb = feats

        return emb[0].detach().cpu().numpy()


def compute_embeddings(tiles_dir, output_dir, fs_args_str=None, model_path: str | None = None):
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
    index_path = _join_uri(tiles_dir, "index.parquet")
    print(f"Reading index from {index_path}")
    
    try:
        df = pd.read_parquet(index_path, storage_options=storage_options)
    except Exception as e:
        # Preserve the initial pandas.read_parquet attempt so unit tests can assert the call;
        # then fall back for environments without parquet engines or with limited fsspec support.
        print(f"Error reading index via pandas: {e}")
        if tiles_dir.startswith("s3://"):
            fs = fsspec.filesystem("s3", **storage_options)
            local_index_path = "/tmp/index.parquet"
            fs.get(index_path, local_index_path)
            # Avoid a second pandas.read_parquet call (tests assert on the first call's storage_options).
            df = _read_parquet_via_duckdb(local_index_path)
        else:
            df = _read_parquet(index_path, storage_options=storage_options)

    print(f"Processing {len(df)} tiles...")
    
    if torch is None:
        raise RuntimeError("torch is required to compute embeddings")
    if rasterio is None:
        raise RuntimeError("rasterio is required to read GeoTIFF tiles")
    if Image is None:
        raise RuntimeError("Pillow is required to prepare images for embedding")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model: prefer local DINOv3 weights if provided to keep runs offline.
    vit_model = None
    vit_preprocess = None
    if model_path:
        print(f"Loading local DINOv3 weights from {model_path}")
        vit_model, vit_img_size = _load_dinov3_timm_model(model_path, device=device)
        vit_preprocess = _make_vit_preprocess(vit_img_size)
    else:
        if AutoImageProcessor is None or AutoModel is None:
            raise RuntimeError("transformers is not installed and no --model_path was provided")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    embeddings = []
    
    fs = fsspec.filesystem("s3", **storage_options) if tiles_dir.startswith("s3://") else None

    for _, row in df.iterrows():
        tile_filename = row['tile_path']
        tile_path = _join_uri(tiles_dir, tile_filename)
        
        # Download tile if S3
        local_tile_path = tile_filename
        if tiles_dir.startswith("s3://"):
            local_tile_basename = os.path.basename(tile_filename)
            local_tile_path = f"/tmp/{local_tile_basename}"
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
            if img_array.ndim == 3 and img_array.shape[0] > 3:
                img_array = img_array[:3, :, :]
            if img_array.ndim == 3 and img_array.shape[0] == 1:
                img_array = np.repeat(img_array, 3, axis=0)
            # Normalize/Transpose for simple PIL creation: (C, H, W) -> (H, W, C)
            img_array = np.moveaxis(img_array, 0, -1)
            
            # Handle non-uint8 or normalization if needed.
            # Simple assumption: Input is valid image-like
            if img_array.dtype != np.uint8:
                # simplistic normalization to 0-255 uint8 for vision model input
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Create PIL
            image = Image.fromarray(img_array)
            
        if vit_model is not None:
            pixel_values = vit_preprocess(image).unsqueeze(0).to(device)
            embedding = _extract_embedding_vit(vit_model, pixel_values).tolist()
        else:
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
    if output_dir.startswith("s3://"):
        local_out_dir = "/tmp/embeddings_out"
        os.makedirs(local_out_dir, exist_ok=True)
        local_out_path = os.path.join(local_out_dir, "embeddings.parquet")
        _write_parquet(df, local_out_path)

        out_path = _join_uri(output_dir, "embeddings.parquet")
        print(f"Uploading to {out_path}...")
        fs_out = fsspec.filesystem("s3", **storage_options)
        fs_out.put(local_out_path, out_path)
        print(f"Saved embeddings to {out_path}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "embeddings.parquet")
        print(f"Saving to {out_path}...")
        _write_parquet(df, out_path)
        print(f"Saved embeddings to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    parser.add_argument("--model_path", default=None, help="Local .pth weights path (e.g. DINOv3) mounted into the container")
    args = parser.parse_args()
    
    compute_embeddings(args.tiles_dir, args.output_dir, args.fs_args, model_path=args.model_path)
