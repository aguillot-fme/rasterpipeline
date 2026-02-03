import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import duckdb
import fsspec


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


def _signature_from_info(path: str, info: Dict[str, Any]) -> str:
    size = info.get("size")
    mtime = info.get("mtime") or info.get("LastModified") or info.get("last_modified")
    if isinstance(mtime, datetime):
        mtime = mtime.isoformat()
    payload = f"{path}|{size}|{mtime}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_signature(path: str, storage_options: dict) -> str:
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
        info = fs.info(path)
        return _signature_from_info(path, info)
    if os.path.exists(path):
        stat = os.stat(path)
        info = {"size": stat.st_size, "mtime": stat.st_mtime}
        return _signature_from_info(path, info)
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _resolve_input_path(path: str, storage_options: dict) -> str:
    if not path.startswith("s3://"):
        return path
    fs = fsspec.filesystem("s3", **storage_options)
    basename = os.path.basename(urlparse(path).path)
    if not basename:
        raise ValueError("Input path must include a filename for S3 downloads.")
    local_dir = os.path.join("/tmp", "llm_reporting")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, basename)
    fs.get(path, local_path)
    return local_path


def _duckdb_reader(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".parq"}:
        return "read_parquet"
    if ext in {".csv", ".tsv", ".txt"}:
        return "read_csv_auto"
    raise ValueError(f"Unsupported file extension: {ext}")


def analyze_dataset(
    input_path: str,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    fs_args_str: Optional[str] = None,
    sample_rows: int = 3,
) -> Dict[str, Any]:
    if sample_rows < 1:
        raise ValueError("sample_rows must be >= 1")

    storage_options = _sanitize_storage_options(fs_args_str)
    local_path = _resolve_input_path(input_path, storage_options)
    dataset_signature = _dataset_signature(input_path, storage_options)

    reader = _duckdb_reader(local_path)
    safe_path = local_path.replace("'", "''")

    con = duckdb.connect()
    table_name = "profile_table"
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {reader}('{safe_path}')"
    )

    try:
        schema_sql = con.execute(f"SHOW CREATE TABLE {table_name}").fetchone()[0]
    except duckdb.Error:
        schema_sql = (
            con.execute(
                "SELECT sql FROM duckdb_tables() WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
        )
    summary_df = con.execute(f"SUMMARIZE {table_name}").df()
    sample_df = con.execute(f"SELECT * FROM {table_name} LIMIT {sample_rows}").df()
    con.close()

    output = {
        "input_path": input_path,
        "dataset_signature": dataset_signature,
        "schema_sql": schema_sql,
        "summary": summary_df.to_dict(orient="records"),
        "sample_rows": sample_df.to_dict(orient="records"),
    }

    if output_path:
        out_path = output_path
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "profile.json")
    else:
        out_path = None

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    parser.add_argument("--sample_rows", type=int, default=3)
    args = parser.parse_args()

    analyze_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir or None,
        output_path=args.output_path or None,
        fs_args_str=args.fs_args,
        sample_rows=args.sample_rows,
    )
