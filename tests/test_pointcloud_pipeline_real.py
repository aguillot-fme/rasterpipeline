import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage import get_storage_backend, S3Storage


def _run_compose_command(args: list[str]) -> str:
    proc = subprocess.run(
        ["docker", "compose", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed:\n{stdout}\n{stderr}")
    # `docker compose` can emit warnings to stdout; return last non-empty line as "result".
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


@pytest.mark.skipif(os.name != "nt", reason="Host-side docker compose test intended for Windows runner")
def test_pointcloud_pipeline_via_docker_and_minio(monkeypatch):
    """
    Host-side integration test:
    - Runs pointcloud ingest/enrich inside `pointcloud-processing-image` container
    - Verifies MinIO contains both raw and enriched outputs via S3Storage
    """
    # Ensure docker is available
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True)
    except Exception as exc:
        pytest.skip(f"Docker not available: {exc}")

    # Configure host-side S3/MinIO settings (host uses localhost; containers use minio)
    monkeypatch.setenv("STORAGE_TYPE", "s3")
    monkeypatch.setenv("S3_BUCKET", os.getenv("S3_BUCKET", "raster-data"))
    monkeypatch.setenv("S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

    try:
        storage = get_storage_backend()
    except Exception as exc:
        pytest.skip(f"MinIO not reachable: {exc}")

    if not isinstance(storage, S3Storage):
        pytest.skip("STORAGE_TYPE did not resolve to S3Storage")

    # Ingest synthetic LAS
    file_id_str = _run_compose_command(
        [
            "run",
            "--rm",
            "pointcloud-processing-image",
            "python",
            "-m",
            "pointcloud_processing.ingest",
            "--source-path",
            "synthetic://sample.las",
        ]
    )
    # Basic sanity check for UUID output
    try:
        file_id = str(uuid.UUID(file_id_str))
    except Exception as exc:
        raise AssertionError(f"Unexpected ingest output (expected UUID): {file_id_str}") from exc

    raw_matches = storage.list_files("pointcloud/raw", pattern=f"*{file_id}*/*.las")
    assert raw_matches, f"Ingest did not write a raw LAS for file_id={file_id}"
    raw_key = raw_matches[0]

    # Enrich
    enriched_path = _run_compose_command(
        [
            "run",
            "--rm",
            "pointcloud-processing-image",
            "python",
            "-m",
            "pointcloud_processing.enrich",
            "--file-id",
            file_id,
        ]
    )
    assert re.match(r"^pointcloud/enriched/.+\.las$", enriched_path), enriched_path
    assert storage.exists(enriched_path)

    # Cleanup to keep bucket tidy
    if not os.getenv("KEEP_TEST_POINTCLOUD"):
        storage.delete_file(raw_key)
        storage.delete_file(enriched_path)
