import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest
import requests


DAG_ID = "raster_ingest"
DEFAULT_SOURCE = "D:/rasterpipeline/data/raw/HEL_0_0.tif"
TERMINAL_STATES = {"success", "failed"}
DEFAULT_WAIT_SECONDS = int(os.getenv("TEST_DAG_WAIT_SECONDS", "60"))


def _get_access_token(base_http: str, username: str, password: str) -> str:
    token_url = os.getenv("AIRFLOW_TOKEN_URL", f"{base_http}/auth/token")
    payload = {"username": username, "password": password}
    try:
        resp = requests.post(token_url, json=payload, timeout=10)
    except Exception as exc:
        pytest.fail(f"Cannot reach Airflow token endpoint: {exc}")

    if resp.status_code not in (200, 201):
        pytest.fail(f"Failed to obtain token ({resp.status_code}): {resp.text}")

    data = resp.json()
    token = data.get("access_token")
    if not token:
        pytest.fail(f"No access_token in token response: {data}")
    return token


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upload_local_raster(local_path: Path) -> str:
    bucket = os.getenv("TEST_RASTER_BUCKET", "raster-data")
    key_prefix = os.getenv("TEST_RASTER_UPLOAD_PREFIX", "raw/rest_trigger")
    endpoint = os.getenv("TEST_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    access_key = os.getenv("TEST_S3_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    secret_key = os.getenv("TEST_S3_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    key = f"{key_prefix.rstrip('/')}/{uuid.uuid4()}/{local_path.name}"
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)

    client.put_object(Bucket=bucket, Key=key, Body=local_path.read_bytes())
    remote_uri = f"s3://{bucket}/{key}"
    print(f"Uploaded {local_path} to {remote_uri}")
    return remote_uri


def _prepare_source_path(source_path: str) -> str:
    parsed = urlparse(source_path)
    is_local = parsed.scheme in ("", "file") or bool(os.path.splitdrive(source_path)[0])
    if is_local:
        local_path = Path(parsed.path if parsed.scheme == "file" else source_path)
        if not local_path.exists():
            pytest.fail(f"Raster not found at {local_path}")
        if os.getenv("TEST_RASTER_UPLOAD", "1").lower() not in ("0", "false", "no"):
            return _upload_local_raster(local_path)
    return source_path


def test_trigger_dag_via_rest():
    """
    List existing DAG runs and trigger raster_ingest via Airflow's REST API (v2) using JWT bearer auth.
    """
    if os.getenv("RUN_AIRFLOW_REST_TESTS", "").strip() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_AIRFLOW_REST_TESTS=1 to run Airflow REST integration test")
    # Base API URL (v2 for DAG operations); caller can override via env
    base_http = os.getenv("AIRFLOW_REST_URL", "http://localhost:8080")
    base_api = os.getenv("AIRFLOW_API_BASE", f"{base_http}/api/v2")
    user = os.getenv("AIRFLOW_REST_USER", "airflow")
    password = os.getenv("AIRFLOW_REST_PASSWORD", "airflow")
    source_path = _prepare_source_path(os.getenv("TEST_RASTER_PATH", DEFAULT_SOURCE))

    token = _get_access_token(base_http, user, password)
    auth_headers = {"Authorization": f"Bearer {token}"}

    dag_run_id = f"test-{uuid.uuid4()}"
    list_url = f"{base_api}/dags/{DAG_ID}/dagRuns"
    trigger_url = f"{base_api}/dags/{DAG_ID}/dagRuns"
    detail_url = f"{base_api}/dags/{DAG_ID}/dagRuns/{dag_run_id}"
    task_url = f"{base_api}/dags/{DAG_ID}/dagRuns/{dag_run_id}/taskInstances"

    list_params = {"order_by": "-logical_date", "limit": 5}
    try:
        list_resp = requests.get(list_url, headers=auth_headers, params=list_params, timeout=10)
    except Exception as exc:
        pytest.fail(f"Cannot list DAG runs: {exc}")

    if list_resp.status_code != 200:
        pytest.fail(f"Failed to list DAG runs ({list_resp.status_code}): {list_resp.text}")

    existing_runs = list_resp.json().get("dag_runs", [])
    print("Existing runs:")
    for run in existing_runs:
        print(f" - {run.get('dag_run_id')} @ {run.get('logical_date')} -> {run.get('state')}")

    payload = {
        "dag_run_id": dag_run_id,
        "logical_date": _iso_now(),
        "conf": {"source_path": source_path},
    }

    try:
        resp = requests.post(trigger_url, json=payload, headers=auth_headers, timeout=10)
    except Exception as exc:
        pytest.fail(f"Cannot reach Airflow API: {exc}")

    if resp.status_code in (401, 403):
        pytest.fail(f"Airflow API auth failed ({resp.status_code}); check AIRFLOW_REST_USER/PASSWORD")

    if resp.status_code not in (200, 201):
        pytest.fail(f"Failed to trigger DAG ({resp.status_code}): {resp.text}")

    # Poll for a terminal state; allow extra time for image pulls/startup but keep bounded
    deadline = time.time() + DEFAULT_WAIT_SECONDS
    last_state = None
    while time.time() < deadline:
        try:
            detail = requests.get(detail_url, headers=auth_headers, timeout=10)
        except Exception:
            break
        if detail.status_code != 200:
            break
        data = detail.json()
        last_state = data.get("state")
        state_lower = (last_state or "").lower()

        # Fallback: if the DAG run still reports "running" but all tasks are success, treat as success
        if state_lower == "running":
            try:
                tasks_resp = requests.get(task_url, headers=auth_headers, timeout=10)
                if tasks_resp.status_code == 200:
                    tasks = tasks_resp.json().get("task_instances", [])
                    if tasks and all(t.get("state") == "success" for t in tasks):
                        last_state = "success"
                        state_lower = "success"
            except Exception:
                pass

        if state_lower in TERMINAL_STATES:
            break
        time.sleep(3)

    if (last_state or "").lower() not in TERMINAL_STATES:
        pytest.fail(f"DAG run did not reach a terminal state (last_state={last_state})")
    assert last_state.lower() == "success", f"DAG run ended in {last_state}"
