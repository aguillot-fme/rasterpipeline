import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from storage import get_storage_backend, S3Storage, LocalStorage
from processing.ingest import ingest_raster
from processing.validate import validate_raster
from processing.transform import calculate_ndvi
from processing.metrics import compute_raster_metrics


def _make_test_raster_bytes(width: int = 4, height: int = 4) -> bytes:
    """Create a small single-band GeoTIFF in memory."""
    data = np.arange(width * height, dtype=np.uint8).reshape(1, height, width)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1),
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile.read()


def test_local_pipeline_with_real_raster(monkeypatch):
    """
    End-to-end smoke test of the ingest/validate/transform/metrics flow using
    the local storage backend rooted in the project data folder.
    """
    # Allow using a provided raster for the test to exercise real data
    override = os.getenv("TEST_RASTER_PATH")
    if override:
        print(f"[test] Using provided raster: {override}")
        raster_bytes = Path(override).read_bytes()
        width = height = None  # unknown until read back
    else:
        raster_bytes = _make_test_raster_bytes()
        width = height = 4

    # Choose band indices for NDVI (default 1/1; override via env)
    red_idx = int(os.getenv("TEST_RED_BAND", "1"))
    nir_idx = int(os.getenv("TEST_NIR_BAND", str(red_idx)))

    # Isolate under data/local_tests/<uuid>
    base_path = Path("/opt/airflow/dags/repo/data/local_tests") / str(uuid.uuid4())
    base_path.mkdir(parents=True, exist_ok=True)

    # Configure storage to local
    monkeypatch.setenv("STORAGE_TYPE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(base_path))
    storage = get_storage_backend()
    assert isinstance(storage, LocalStorage)

    source_path = base_path / "source.tif"
    source_path.write_bytes(raster_bytes)

    # Run the pipeline pieces
    file_id = ingest_raster(storage, str(source_path), destination_dir="raw")
    ingested_files = storage.list_files("raw", pattern=f"*{file_id}*/*.tif")
    print(f"[test] Ingested file_id={file_id}, local dest={ingested_files}, red_idx={red_idx}, nir_idx={nir_idx}")
    assert ingested_files, "Ingest did not write to local storage"

    file_path = ingested_files[0]
    assert validate_raster(storage, file_path) is True

    output_path = file_path.replace("raw", "processed").replace(".tif", "_ndvi.tif")
    calculate_ndvi(storage, file_path, file_path, output_path, red_band_index=red_idx, nir_band_index=nir_idx)
    print(f"[test] NDVI written to {output_path}")
    assert storage.exists(output_path)

    metrics = compute_raster_metrics(storage, output_path)
    if width and height:
        assert metrics["width"] == width
        assert metrics["height"] == height
    assert metrics["count"] == 1

    # Cleanup the isolated test directory to keep the host volume tidy
    if not os.getenv("KEEP_TEST_RASTER"):
        shutil.rmtree(base_path, ignore_errors=True)


def test_minio_pipeline_with_real_raster(monkeypatch):
    """
    Smoke test the transform/metrics path using MinIO via S3 REST (boto3).
    Skips if MinIO is unreachable.
    """
    # Configure S3/MinIO settings
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    bucket = os.getenv("S3_BUCKET", "raster-data")
    monkeypatch.setenv("STORAGE_TYPE", "s3")
    monkeypatch.setenv("S3_BUCKET", bucket)
    monkeypatch.setenv("S3_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

    # Initialize storage; if MinIO is not reachable, skip the test
    try:
        storage = get_storage_backend()
    except Exception as exc:
        pytest.skip(f"MinIO not reachable: {exc}")

    if not isinstance(storage, S3Storage):
        pytest.skip("STORAGE_TYPE did not resolve to S3Storage")

    override = os.getenv("TEST_RASTER_PATH")
    if override:
        print(f"[test] Using provided raster: {override}")
        raster_bytes = Path(override).read_bytes()
        width = height = None
    else:
        raster_bytes = _make_test_raster_bytes()
        width = height = 4

    # Choose band indices for NDVI (default 1/1; override via env)
    red_idx = int(os.getenv("TEST_RED_BAND", "1"))
    nir_idx = int(os.getenv("TEST_NIR_BAND", str(red_idx)))
    file_id = str(uuid.uuid4())
    src_key = f"raw/{file_id}/source.tif"

    # Upload the raster via the S3 REST client (boto3 under the hood)
    storage.write_file(src_key, raster_bytes)
    print(f"[test] Uploaded raster to s3://{storage.bucket_name}/{src_key}, red_idx={red_idx}, nir_idx={nir_idx}")
    assert storage.exists(src_key)

    # Validate and transform using the S3 backend
    assert validate_raster(storage, src_key) is True

    out_key = src_key.replace("raw", "processed").replace(".tif", "_ndvi.tif")
    calculate_ndvi(storage, src_key, src_key, out_key, red_band_index=red_idx, nir_band_index=nir_idx)
    print(f"[test] NDVI written to s3://{storage.bucket_name}/{out_key}")
    assert storage.exists(out_key)

    metrics = compute_raster_metrics(storage, out_key)
    if width and height:
        assert metrics["width"] == width
        assert metrics["height"] == height
    assert metrics["count"] == 1

    # Cleanup objects to keep the bucket tidy
    if not os.getenv("KEEP_TEST_RASTER"):
        storage.delete_file(src_key)
        storage.delete_file(out_key)
