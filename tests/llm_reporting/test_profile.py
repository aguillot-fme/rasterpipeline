import json
import os
import tempfile

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.llm_reporting.profile import analyze_dataset


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        con = duckdb.connect()
        con.register("df", df)
        safe_path = path.replace("'", "''")
        con.execute(f"COPY df TO '{safe_path}' (FORMAT PARQUET)")
        con.close()


def test_analyze_dataset_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "count": [1, 2, 3],
        }
    )
    parquet_path = tmp_path / "sample.parquet"
    _write_parquet(df, str(parquet_path))

    out_dir = tmp_path / "out"
    result = analyze_dataset(str(parquet_path), output_dir=str(out_dir))

    out_path = out_dir / "profile.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["dataset_signature"]
    assert "CREATE TABLE" in payload["schema_sql"].upper()
    assert isinstance(payload["summary"], list)
    assert isinstance(payload["sample_rows"], list)
    assert len(payload["sample_rows"]) == 3
    assert result["dataset_signature"] == payload["dataset_signature"]


def test_analyze_dataset_csv(tmp_path):
    df = pd.DataFrame(
        {
            "city": ["Helsinki", "Espoo"],
            "value": [10.5, 3.2],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_dataset(str(csv_path), output_dir=str(tmp_path))

    assert result["dataset_signature"]
    assert isinstance(result["summary"], list)
    assert isinstance(result["sample_rows"], list)


@pytest.mark.integration
def test_analyze_dataset_minio_real():
    endpoint = os.getenv("MINIO_ENDPOINT")
    bucket = os.getenv("MINIO_BUCKET")
    key = os.getenv("MINIO_KEY")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not all([endpoint, bucket, key, access_key, secret_key]):
        pytest.skip("MINIO_* env vars not set for integration test")

    s3_path = f"s3://{bucket}/{key}"
    fs_args = json.dumps(
        {
            "AWS_ENDPOINT_URL": endpoint,
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = analyze_dataset(s3_path, output_dir=tmpdir, fs_args_str=fs_args)

    assert result["dataset_signature"]
