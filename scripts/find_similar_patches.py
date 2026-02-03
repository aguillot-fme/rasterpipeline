import argparse
import json
import os
import posixpath
from typing import Callable, Iterable, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import fsspec

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

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

try:
    from transformers import AutoImageProcessor, AutoModel  # type: ignore
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None

from scripts.compute_embeddings import (
    MODEL_NAME,
    _extract_embedding_vit,
    _load_dinov3_timm_model,
    _make_vit_preprocess,
)


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
        print(f"pandas.read_parquet failed ({type(e).__name__}: {e}); falling back to DuckDB read_parquet...")
        if duckdb is None:
            raise RuntimeError("duckdb is required for parquet fallback reads") from e
        if path.startswith("s3://"):
            fs = fsspec.filesystem("s3", **storage_options)
            local_path = "/tmp/duckdb_parquet.parquet"
            fs.get(path, local_path)
            path = local_path
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
        con.register("results_df", df)
        safe_path = path.replace("'", "''")
        con.execute(f"COPY results_df TO '{safe_path}' (FORMAT PARQUET)")
        con.close()


def _parse_query_paths(raw: Optional[str], list_file: Optional[str], query_dir: Optional[str], storage_options: dict) -> List[str]:
    paths: List[str] = []

    if raw:
        raw = raw.strip()
        if raw:
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        paths.extend([str(p) for p in parsed])
                    else:
                        paths.append(str(parsed))
                except json.JSONDecodeError:
                    paths.extend([p.strip() for p in raw.split(",") if p.strip()])
            else:
                paths.extend([p.strip() for p in raw.split(",") if p.strip()])

    if list_file:
        fs = fsspec.filesystem("s3", **storage_options) if list_file.startswith("s3://") else None
        if fs:
            with fs.open(list_file, "rb") as f:
                content = f.read().decode("utf-8")
        else:
            with open(list_file, "r", encoding="utf-8") as f:
                content = f.read()
        content = content.strip()
        if content.startswith("["):
            parsed = json.loads(content)
            if isinstance(parsed, list):
                paths.extend([str(p) for p in parsed])
        else:
            paths.extend([line.strip() for line in content.splitlines() if line.strip()])

    if query_dir:
        if query_dir.startswith("s3://"):
            fs = fsspec.filesystem("s3", **storage_options)
            pattern = query_dir.rstrip("/") + "/**/*.tif*"
            paths.extend(sorted(fs.glob(pattern)))
        else:
            import glob

            pattern = os.path.join(query_dir, "**", "*.tif*")
            paths.extend(sorted(glob.glob(pattern, recursive=True)))

    return paths


def _parse_query_coords(raw: Optional[str]) -> List[dict]:
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            coords = []
            for item in parsed:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    coords.append({"x": float(item["x"]), "y": float(item["y"])})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    coords.append({"x": float(item[0]), "y": float(item[1])})
            return coords
    coords = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        xy = part.split()
        if len(xy) == 2:
            coords.append({"x": float(xy[0]), "y": float(xy[1])})
    return coords


def _default_output_path(output_path: str, embeddings_dir: str) -> str:
    if output_path:
        if output_path.endswith("/") or os.path.splitext(output_path)[1] == "":
            return _join_uri(output_path, "similarity_results.parquet")
        return output_path
    return _join_uri(embeddings_dir, "similarity_results.parquet")


def _csv_path_from_output(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    if not ext:
        return _join_uri(output_path, "similarity_results.csv")
    return f"{root}.csv"


def _build_embedder(model_path: Optional[str]):
    if torch is None or rasterio is None or Image is None:
        raise RuntimeError("torch, rasterio, and Pillow are required for similarity search")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path:
        print(f"Loading local DINOv3 weights from {model_path}")
        vit_model, vit_img_size = _load_dinov3_timm_model(model_path, device=device)
        vit_preprocess = _make_vit_preprocess(vit_img_size)

        def embed_fn(image):
            pixel_values = vit_preprocess(image).unsqueeze(0).to(device)
            return _extract_embedding_vit(vit_model, pixel_values)

        return embed_fn

    if AutoImageProcessor is None or AutoModel is None:
        raise RuntimeError("transformers is not installed and no --model_path was provided")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    def embed_fn(image):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            return last_hidden_states[0, 0, :].detach().cpu().numpy()

    return embed_fn


def _load_patch_image(path: str, storage_options: dict) -> Image.Image:
    local_path = path
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
        local_path = os.path.join("/tmp", os.path.basename(path))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        fs.get(path, local_path)

    with rasterio.open(local_path) as src:
        img_array = src.read()
        if img_array.ndim == 3 and img_array.shape[0] > 3:
            img_array = img_array[:3, :, :]
        if img_array.ndim == 3 and img_array.shape[0] == 1:
            img_array = np.repeat(img_array, 3, axis=0)
        img_array = np.moveaxis(img_array, 0, -1)
        if img_array.dtype != np.uint8:
            denom = img_array.max() - img_array.min()
            if denom == 0:
                img_array = np.zeros_like(img_array, dtype=np.uint8)
            else:
                img_array = ((img_array - img_array.min()) / denom * 255).astype(np.uint8)
        return Image.fromarray(img_array)


def find_similar_patches(
    embeddings_dir: str,
    query_paths: Iterable[str],
    output_path: str,
    fs_args_str: Optional[str] = None,
    model_path: Optional[str] = None,
    top_k: int = 5,
    embedder: Optional[Callable[[Image.Image], np.ndarray]] = None,
    tiles_dir: Optional[str] = None,
    query_coords: Optional[List[dict]] = None,
) -> pd.DataFrame:
    storage_options = _sanitize_storage_options(fs_args_str)
    embeddings_path = _join_uri(embeddings_dir, "embeddings.parquet")
    print(f"Reading embeddings from {embeddings_path}")
    emb_df = _read_parquet(embeddings_path, storage_options=storage_options)

    if "embedding" not in emb_df.columns:
        raise ValueError("Embeddings parquet is missing 'embedding' column")

    emb_df = emb_df[emb_df["embedding"].notna()].reset_index(drop=True)
    if emb_df.empty:
        raise ValueError("Embeddings parquet has no embeddings to search")

    tiles_df = None
    tile_bounds = {}
    if tiles_dir:
        index_path = _join_uri(tiles_dir, "index.parquet")
        print(f"Reading tiles index from {index_path}")
        tiles_df = _read_parquet(index_path, storage_options=storage_options)
        required_cols = {"tile_id", "min_x", "min_y", "max_x", "max_y"}
        if not required_cols.issubset(set(tiles_df.columns)):
            missing = required_cols - set(tiles_df.columns)
            raise ValueError(f"Tiles index missing columns: {sorted(missing)}")
        else:
            for _, row in tiles_df.iterrows():
                tile_bounds[row["tile_id"]] = (
                    float(row["min_x"]),
                    float(row["min_y"]),
                    float(row["max_x"]),
                    float(row["max_y"]),
                )

    has_patch_bounds = {"patch_min_x", "patch_min_y", "patch_max_x", "patch_max_y"}.issubset(
        set(emb_df.columns)
    )

    vectors = np.vstack([np.asarray(v, dtype=np.float32) for v in emb_df["embedding"]])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    results = []

    embed_fn = embedder or _build_embedder(model_path) if query_paths else None

    for qpath in query_paths:
        if not qpath:
            continue
        image = _load_patch_image(qpath, storage_options)
        if embed_fn is None:
            raise RuntimeError("Embedding model unavailable for query_paths input")
        qvec = embed_fn(image)
        qvec = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
        qnorm = np.linalg.norm(qvec, axis=1, keepdims=True)
        qvec = qvec / (qnorm if qnorm[0, 0] != 0 else 1.0)

        scores = (vectors @ qvec.T).reshape(-1)
        top_idx = np.argsort(scores)[::-1][:top_k]
        for rank, idx in enumerate(top_idx, start=1):
            row = emb_df.iloc[int(idx)]
            if has_patch_bounds:
                bounds = (
                    row.get("patch_min_x"),
                    row.get("patch_min_y"),
                    row.get("patch_max_x"),
                    row.get("patch_max_y"),
                )
            else:
                bounds = tile_bounds.get(row.get("tile_id")) if tile_bounds else None
            results.append(
                {
                    "query_path": qpath,
                    "query_x": None,
                    "query_y": None,
                    "match_tile_id": row.get("tile_id"),
                    "match_tile_path": row.get("tile_path"),
                    "match_patch_id": row.get("patch_id"),
                    "score": float(scores[int(idx)]),
                    "rank": int(rank),
                    "match_min_x": bounds[0] if bounds else None,
                    "match_min_y": bounds[1] if bounds else None,
                    "match_max_x": bounds[2] if bounds else None,
                    "match_max_y": bounds[3] if bounds else None,
                }
            )

    if query_coords:
        for coord in query_coords:
            x = float(coord["x"])
            y = float(coord["y"])
            if has_patch_bounds:
                match = emb_df[
                    (emb_df["patch_min_x"] <= x)
                    & (emb_df["patch_max_x"] > x)
                    & (emb_df["patch_min_y"] <= y)
                    & (emb_df["patch_max_y"] > y)
                ]
                if match.empty:
                    raise ValueError(f"No patch contains point x={x}, y={y}")
                emb_row = match.iloc[0]
            else:
                if not tiles_dir or tiles_df is None:
                    raise ValueError("tiles_dir is required when using query_coords")
                required_cols = {"tile_id", "min_x", "min_y", "max_x", "max_y"}
                if tiles_df is None or not required_cols.issubset(set(tiles_df.columns)):
                    missing = required_cols - set(tiles_df.columns)
                    raise ValueError(f"Tiles index missing columns: {sorted(missing)}")
                match = tiles_df[
                    (tiles_df["min_x"] <= x)
                    & (tiles_df["max_x"] > x)
                    & (tiles_df["min_y"] <= y)
                    & (tiles_df["max_y"] > y)
                ]
                if match.empty:
                    raise ValueError(f"No tile contains point x={x}, y={y}")
                tile_id = match.iloc[0]["tile_id"]
                emb_row = emb_df[emb_df["tile_id"] == tile_id]
                if emb_row.empty:
                    raise ValueError(f"No embedding found for tile_id {tile_id}")
                emb_row = emb_row.iloc[0]

            qvec = np.asarray(emb_row["embedding"], dtype=np.float32).reshape(1, -1)
            qnorm = np.linalg.norm(qvec, axis=1, keepdims=True)
            qvec = qvec / (qnorm if qnorm[0, 0] != 0 else 1.0)

            scores = (vectors @ qvec.T).reshape(-1)
            top_idx = np.argsort(scores)[::-1][:top_k]
            for rank, idx in enumerate(top_idx, start=1):
                row = emb_df.iloc[int(idx)]
                if has_patch_bounds:
                    bounds = (
                        row.get("patch_min_x"),
                        row.get("patch_min_y"),
                        row.get("patch_max_x"),
                        row.get("patch_max_y"),
                    )
                else:
                    bounds = tile_bounds.get(row.get("tile_id")) if tile_bounds else None
                results.append(
                    {
                        "query_path": None,
                        "query_x": x,
                        "query_y": y,
                        "match_tile_id": row.get("tile_id"),
                        "match_tile_path": row.get("tile_path"),
                        "match_patch_id": row.get("patch_id"),
                        "score": float(scores[int(idx)]),
                        "rank": int(rank),
                        "match_min_x": bounds[0] if bounds else None,
                        "match_min_y": bounds[1] if bounds else None,
                        "match_max_x": bounds[2] if bounds else None,
                        "match_max_y": bounds[3] if bounds else None,
                    }
                )

    out_df = pd.DataFrame(results)
    output_path = _default_output_path(output_path, embeddings_dir)
    csv_path = _csv_path_from_output(output_path)

    if output_path.startswith("s3://"):
        local_out = "/tmp/similarity_results.parquet"
        local_csv = "/tmp/similarity_results.csv"
        _write_parquet(out_df, local_out)
        out_df.to_csv(local_csv, index=False)
        fs = fsspec.filesystem("s3", **storage_options)
        fs.put(local_out, output_path)
        fs.put(local_csv, csv_path)
        print(f"Saved results to {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _write_parquet(out_df, output_path)
        out_df.to_csv(csv_path, index=False)
        print(f"Saved results to {output_path}")

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--tiles_dir", default=None, help="Tiles directory containing index.parquet")
    parser.add_argument("--query_paths", default="", help="Comma-separated list or JSON list string")
    parser.add_argument("--query_paths_file", default=None, help="Path to .txt/.json with query paths")
    parser.add_argument("--query_dir", default=None, help="Directory (local or s3://) containing query patches")
    parser.add_argument("--query_coords", default="", help="JSON list of {x,y} or list of [x,y] pairs")
    parser.add_argument("--output_path", default="", help="Output path (file or dir). Defaults under embeddings_dir.")
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    parser.add_argument("--model_path", default=None, help="Local .pth weights path mounted into the container")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    storage_options = _sanitize_storage_options(args.fs_args)
    query_paths = _parse_query_paths(args.query_paths, args.query_paths_file, args.query_dir, storage_options)
    query_coords = _parse_query_coords(args.query_coords)
    if not query_paths and not query_coords:
        raise SystemExit(
            "No query inputs provided. Use --query_paths, --query_paths_file, --query_dir, or --query_coords."
        )

    find_similar_patches(
        embeddings_dir=args.embeddings_dir,
        query_paths=query_paths,
        output_path=args.output_path,
        fs_args_str=args.fs_args,
        model_path=args.model_path,
        top_k=args.top_k,
        tiles_dir=args.tiles_dir,
        query_coords=query_coords,
    )
