import argparse
import json
import os
import posixpath
from typing import Any, Dict
from urllib.parse import urlparse

import duckdb
import fsspec
import pandas as pd


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
        con = duckdb.connect()
        safe_path = path.replace("'", "''")
        df = con.execute(f"SELECT * FROM read_parquet('{safe_path}')").df()
        con.close()
        return df


def _write_json_to_dest(payload: Dict[str, Any], dest: str, storage_options: dict) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    if dest.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
        with fs.open(dest, "wb") as f:
            f.write(data)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)


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


def qa_metrics(tiles_dir: str, embeddings_dir: str, fs_args_str: str = "{}") -> Dict[str, Any]:
    storage_options = _sanitize_storage_options(fs_args_str)

    index_path = _join_uri(tiles_dir, "index.parquet")
    embeddings_path = _join_uri(embeddings_dir, "embeddings.parquet")

    print(f"Reading tiles index from {index_path}")
    tiles_df = _read_parquet(index_path, storage_options=storage_options)

    print(f"Reading embeddings from {embeddings_path}")
    emb_df = _read_parquet(embeddings_path, storage_options=storage_options)

    tile_count = len(tiles_df)
    emb_count = len(emb_df)
    missing_embeddings = 0
    embedding_dim = None
    if "embedding" in emb_df.columns and len(emb_df) > 0:
        first = emb_df.iloc[0]["embedding"]
        try:
            embedding_dim = len(first)
        except Exception:
            embedding_dim = None
        missing_embeddings = int(emb_df["embedding"].isna().sum()) if "embedding" in emb_df.columns else 0

    summary = {
        "tile_count": int(tile_count),
        "embedding_count": int(emb_count),
        "missing_embeddings": int(missing_embeddings),
        "embedding_dim": embedding_dim,
        "tiles_dir": tiles_dir,
        "embeddings_dir": embeddings_dir,
    }

    qa_path = _join_uri(embeddings_dir, "qa_metrics.json")
    print(f"Writing QA summary to {qa_path}")
    _write_json_to_dest(summary, qa_path, storage_options=storage_options)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_dir", required=True)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    args = parser.parse_args()

    qa_metrics(args.tiles_dir, args.embeddings_dir, args.fs_args)
