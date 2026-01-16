import json
import os
import posixpath
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import PythonOperator
from docker.types import Mount

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from storage import get_storage_backend

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Configuration
DEFAULT_RASTER_PATH = os.getenv(
    "DEFAULT_RASTER_PATH",
    "s3://raster-data/raw/20251210_121155_704c36a2-73ae-4794-86ca-ebcc5bd6d5f3/HEL_0_0.tif",
)

# Docker Config
PROCESSING_IMAGE = os.getenv("PROCESSING_IMAGE", "raster-processing:latest")
ANALYTICS_IMAGE = os.getenv("ANALYTICS_IMAGE", "analytics-embeddings:latest")
PROCESSING_NETWORK = os.getenv("DOCKER_NETWORK", "rasterpipeline_default")
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET") or os.getenv("DOCKER_HOST") or "unix://var/run/docker.sock"
DEV_CODE_MOUNT = os.getenv("DEV_CODE_MOUNT", "").lower() in {"1", "true", "yes"}
HOST_REPO_PATH = os.getenv("HOST_REPO_PATH")

# Shared Environment
COMMON_ENV = {
    "STORAGE_TYPE": os.getenv("STORAGE_TYPE", "local"),
    "S3_BUCKET": os.getenv("S3_BUCKET", "raster-data"),
    "S3_PREFIX": os.getenv("S3_PREFIX", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL"),
}
COMMON_ENV = {k: v for k, v in COMMON_ENV.items() if v is not None}

STORAGE_TYPE = COMMON_ENV.get("STORAGE_TYPE", "local").lower()
S3_BUCKET = COMMON_ENV.get("S3_BUCKET", "raster-data")
S3_PREFIX = COMMON_ENV.get("S3_PREFIX", "")
S3_ENDPOINT_URL = COMMON_ENV.get("S3_ENDPOINT_URL", "http://minio:9000")
LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "/opt/airflow/dags/repo")
FS_ARGS_JSON = json.dumps({"AWS_ENDPOINT_URL": S3_ENDPOINT_URL}) if STORAGE_TYPE == "s3" else "{}"


def _join_s3(prefix: str, *parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p]
    if prefix:
        cleaned = [prefix.strip("/")] + cleaned
    return f"s3://{S3_BUCKET}/{posixpath.join(*cleaned)}"


def _output_dir(base_dir: str, file_id: str) -> str:
    if STORAGE_TYPE == "s3":
        return _join_s3(S3_PREFIX, base_dir, file_id)
    return os.path.join(LOCAL_STORAGE_PATH, base_dir, file_id)

dag = DAG(
    "raster_embeddings",
    default_args=default_args,
    description="Pipeline to tile rasters and compute DINOv3 embeddings",
    schedule=None,
    catchup=False,
    params={"default_raster_path": DEFAULT_RASTER_PATH},
)


def get_docker_operator(task_id, image, command, environment=None, mounts=None, **kwargs):
    merged_env = {**COMMON_ENV, **(environment or {})}
    final_mounts = []
    if DEV_CODE_MOUNT and HOST_REPO_PATH:
        final_mounts.append(Mount(target="/app", source=HOST_REPO_PATH, type="bind", read_only=False))
    if mounts:
        final_mounts.extend(mounts)

    return DockerOperator(
        task_id=task_id,
        image=image,
        api_version="auto",
        auto_remove="success",
        mount_tmp_dir=False,
        xcom_all=False,
        command=command,
        docker_url=DOCKER_SOCKET,
        network_mode=PROCESSING_NETWORK,
        environment=merged_env,
        mounts=final_mounts or None,
        mem_limit="4g",
        cpus=2.0,
        **kwargs,
    )


def resolve_storage_path(**kwargs):
    ti = kwargs["ti"]
    file_id = ti.xcom_pull(task_ids="ingest_task")
    if not file_id:
        raise ValueError("No file_id returned from ingest_task")

    file_id = str(file_id).strip()
    storage = get_storage_backend()
    pattern = f"*_{file_id}/*.tif*"
    matches = storage.list_files("raw", pattern=pattern)
    if not matches:
        raise FileNotFoundError(f"No raster found under raw/ for file_id {file_id}")

    matches = sorted(matches)
    if len(matches) > 1:
        print(f"Multiple rasters found for {file_id}; using {matches[-1]}")
    return matches[-1]


with dag:
    ingest_task = get_docker_operator(
        task_id="ingest_task",
        image=PROCESSING_IMAGE,
        command=[
            "python",
            "-m",
            "processing.ingest",
            "--source-path",
            "{{ (dag_run.conf.get('source_path') if dag_run and dag_run.conf else params.default_raster_path) or '"
            + DEFAULT_RASTER_PATH
            + "' }}",
            "--destination-dir",
            "raw",
        ],
        do_xcom_push=True,
    )

    resolve_path_task = PythonOperator(
        task_id="resolve_path_task",
        python_callable=resolve_storage_path,
    )

    tile_task = get_docker_operator(
        task_id="tile_task",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/tile_raster.py",
            "--input",
            "{{ ti.xcom_pull(task_ids='resolve_path_task') }}",
            "--output_dir",
            _output_dir("tiles", "{{ ti.xcom_pull(task_ids='ingest_task') }}"),
            "--fs_args",
            FS_ARGS_JSON,
        ],
    )

    embed_task = get_docker_operator(
        task_id="embed_task",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/compute_embeddings.py",
            "--tiles_dir",
            _output_dir("tiles", "{{ ti.xcom_pull(task_ids='ingest_task') }}"),
            "--output_dir",
            _output_dir("embeddings", "{{ ti.xcom_pull(task_ids='ingest_task') }}"),
            "--fs_args",
            FS_ARGS_JSON,
        ],
    )

    qa_task = get_docker_operator(
        task_id="qa_task",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/qa_metrics.py",
            "--tiles_dir",
            _output_dir("tiles", "{{ ti.xcom_pull(task_ids='ingest_task') }}"),
            "--embeddings_dir",
            _output_dir("embeddings", "{{ ti.xcom_pull(task_ids='ingest_task') }}"),
            "--fs_args",
            FS_ARGS_JSON,
        ],
    )

    ingest_task >> resolve_path_task >> tile_task >> embed_task >> qa_task
