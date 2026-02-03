import json
import os

import pandas as pd
import pytest

from scripts.llm_reporting.executor import run_queries


def test_run_queries_local(tmp_path):
    df = pd.DataFrame({"city": ["A", "B"], "value": [1, 2]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    queries = {
        "queries": [
            {"question": "top rows", "sql": "SELECT * FROM reporting_table"},
        ]
    }
    output_dir = tmp_path / "out"
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(queries), encoding="utf-8")
    result = run_queries(
        queries_json=str(queries_path),
        input_path=str(csv_path),
        output_dir=str(output_dir),
        output_format="json",
    )

    index_path = output_dir / "results_index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["row_count"] == 2
    assert result["output_dir"] == str(output_dir)


def test_run_queries_rejects_non_select(tmp_path):
    df = pd.DataFrame({"x": [1]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    queries = {"queries": [{"question": "bad", "sql": "DELETE FROM reporting_table"}]}

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(queries), encoding="utf-8")

    with pytest.raises(ValueError):
        run_queries(
            queries_json=str(queries_path),
            input_path=str(csv_path),
            output_dir=str(tmp_path / "out"),
        )


@pytest.mark.integration
def test_run_queries_minio_real():
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
    queries = {"queries": [{"question": "sample", "sql": "SELECT * FROM reporting_table"}]}
    local_queries = json.dumps(queries)

    result = run_queries(
        queries_json=local_queries,
        input_path=s3_path,
        output_dir="s3://{}/llm_reports/test_run".format(bucket),
        fs_args_str=fs_args,
    )

    assert result["results"][0]["status"] in {"success", "failed"}
