import importlib.util
import os
import sys

import pytest

pytest.importorskip("airflow")


def _load_dag_module(monkeypatch, **env_overrides):
    for key, value in env_overrides.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    path = os.path.join("airflow", "dags", "dag_raster_embeddings.py")
    spec = importlib.util.spec_from_file_location("dag_raster_embeddings", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dag_raster_embeddings"] = module
    spec.loader.exec_module(module)
    return module


def test_output_dir_s3(monkeypatch):
    module = _load_dag_module(
        monkeypatch,
        STORAGE_TYPE="s3",
        S3_BUCKET="test-bucket",
        S3_PREFIX="prefix",
        S3_ENDPOINT_URL="http://minio:9000",
        LOCAL_STORAGE_PATH="/tmp/local",
    )

    assert (
        module._output_dir("tiles", "abc123")
        == "s3://test-bucket/prefix/tiles/abc123"
    )
    assert module.FS_ARGS_JSON == '{"AWS_ENDPOINT_URL": "http://minio:9000"}'


def test_output_dir_local(monkeypatch, tmp_path):
    module = _load_dag_module(
        monkeypatch,
        STORAGE_TYPE="local",
        LOCAL_STORAGE_PATH=str(tmp_path),
    )

    assert module._output_dir("embeddings", "abc123") == os.path.join(
        str(tmp_path), "embeddings", "abc123"
    )
    assert module.FS_ARGS_JSON == "{}"


def test_resolve_storage_path_uses_storage(monkeypatch):
    module = _load_dag_module(monkeypatch, STORAGE_TYPE="s3")

    class DummyStorage:
        def list_files(self, _path, pattern=None):
            return ["s3://bucket/raw/a.tif", "s3://bucket/raw/b.tif"]

    module.get_storage_backend = lambda: DummyStorage()

    class DummyTI:
        def xcom_pull(self, task_ids):
            assert task_ids == "ingest_task"
            return "file-id"

    resolved = module.resolve_storage_path(ti=DummyTI())
    assert resolved == "s3://bucket/raw/b.tif"


def test_raster_embeddings_dag_structure():
    try:
        from airflow.models import DagBag
    except Exception:
        pytest.skip("Apache Airflow not installed in this environment")

    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert dag_bag.import_errors == {}
    assert "raster_embeddings" in dag_bag.dags

    dag = dag_bag.dags["raster_embeddings"]
    assert len(dag.tasks) == 5
    assert dag.params.get("default_raster_path")

    ingest = dag.get_task("ingest_task")
    resolve = dag.get_task("resolve_path_task")
    tile = dag.get_task("tile_task")
    embed = dag.get_task("embed_task")
    qa = dag.get_task("qa_task")

    assert resolve.upstream_task_ids == {ingest.task_id}
    assert tile.upstream_task_ids == {resolve.task_id}
    assert embed.upstream_task_ids == {tile.task_id}
    assert qa.upstream_task_ids == {embed.task_id}
