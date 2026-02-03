import json
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import duckdb
import fsspec

from scripts.llm_reporting.contracts import loads_json, validate_queries


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


def _ensure_select(sql: str) -> None:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")


def run_queries(
    queries_json: str,
    input_path: str,
    output_dir: str,
    fs_args_str: Optional[str] = None,
    output_format: str = "json",
) -> Dict[str, Any]:
    storage_options = _sanitize_storage_options(fs_args_str)
    local_input = _resolve_input_path(input_path, storage_options)
    reader = _duckdb_reader(local_input)

    upload_fs = None
    s3_output_dir = None
    local_output_dir = output_dir
    if output_dir.startswith("s3://"):
        upload_fs = fsspec.filesystem("s3", **storage_options)
        s3_output_dir = output_dir.rstrip("/")
        local_output_dir = os.path.join("/tmp", "llm_reporting_results")
    os.makedirs(local_output_dir, exist_ok=True)
    output_format = output_format.lower()
    if output_format not in {"json", "csv", "parquet"}:
        raise ValueError("output_format must be json, csv, or parquet")

    if queries_json.strip().startswith("{"):
        payload = validate_queries(loads_json(queries_json))
    else:
        with open(queries_json, "r", encoding="utf-8") as f:
            payload = validate_queries(loads_json(f.read()))
    queries = payload["queries"]

    con = duckdb.connect()
    table_name = "reporting_table"
    safe_path = local_input.replace("'", "''")
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {reader}('{safe_path}')")

    results_index = []
    for idx, item in enumerate(queries, start=1):
        sql = item["sql"]
        _ensure_select(sql)
        started = time.time()
        status = "success"
        error = ""
        rows = 0
        output_name = f"query_{idx}.{output_format}"
        out_path = os.path.join(local_output_dir, output_name)
        try:
            df = con.execute(sql).df()
            rows = len(df.index)
            if output_format == "json":
                df.to_json(out_path, orient="records")
            elif output_format == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_parquet(out_path, index=False)
        except Exception as exc:
            status = "failed"
            error = str(exc)
        duration = time.time() - started
        results_index.append(
            {
                "query_id": idx,
                "question": item.get("question"),
                "sql": sql,
                "status": status,
                "row_count": rows,
                "duration_sec": round(duration, 4),
                "output_file": output_name if status == "success" else None,
                "error": error or None,
            }
        )

    con.close()

    index_path = os.path.join(local_output_dir, "results_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"results": results_index}, f, indent=2)

    if upload_fs and s3_output_dir:
        upload_fs.put(local_output_dir, s3_output_dir, recursive=True)

    return {"results": results_index, "output_dir": s3_output_dir or local_output_dir}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fs_args", default="{}", help="JSON string of fs args")
    parser.add_argument("--output_format", default="json")
    args = parser.parse_args()

    run_queries(
        queries_json=args.queries_json,
        input_path=args.input_path,
        output_dir=args.output_dir,
        fs_args_str=args.fs_args,
        output_format=args.output_format,
    )
