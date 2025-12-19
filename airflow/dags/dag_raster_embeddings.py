import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import PythonOperator
from docker.types import Mount

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from processing.ingest import ingest_raster_with_dest_path
from storage import get_storage_backend, S3Storage

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

DEFAULT_RASTER_PATH = os.getenv("DEFAULT_RASTER_PATH", "data/raw/HEL_0_0.tif")
ANALYTICS_IMAGE = os.getenv("ANALYTICS_IMAGE", "analytics-embeddings:latest")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "rasterpipeline_default")
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET") or os.getenv("DOCKER_HOST") or "unix://var/run/docker.sock"
DEV_CODE_MOUNT = os.getenv("DEV_CODE_MOUNT", "").lower() in {"1", "true", "yes"}
HOST_REPO_PATH = os.getenv("HOST_REPO_PATH")
REPO_ROOT = os.getenv("REPO_ROOT", "/opt/airflow/dags/repo")

PROCESSING_ENV = {
    "STORAGE_TYPE": os.getenv("STORAGE_TYPE", "s3"),
    "S3_BUCKET": os.getenv("S3_BUCKET", "raster-data"),
    "S3_PREFIX": os.getenv("S3_PREFIX", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
}
PROCESSING_ENV = {k: v for k, v in PROCESSING_ENV.items() if v is not None}
PROCESSING_ENV.setdefault("AWS_ENDPOINT_URL", PROCESSING_ENV.get("S3_ENDPOINT_URL"))

DINO_MODEL_HOST_PATH = os.getenv("DINO_MODEL_HOST_PATH", "").strip()
DINO_MODEL_CONTAINER_DIR = os.getenv("DINO_MODEL_CONTAINER_DIR", "/models/dinov3")


def get_embeddings_operator(task_id, command, environment=None, mounts=None, **kwargs):
    merged_env = {**PROCESSING_ENV, **(environment or {})}
    final_mounts = list(mounts or [])

    if DEV_CODE_MOUNT and HOST_REPO_PATH:
        # Match other DAGs and also let the container use live scripts without rebuilding.
        final_mounts.append(
            Mount(target="/opt/airflow/dags/repo", source=HOST_REPO_PATH, type="bind", read_only=False)
        )
        final_mounts.append(
            Mount(
                target="/app/scripts",
                source=os.path.join(HOST_REPO_PATH, "scripts").replace("\\", "/"),
                type="bind",
                read_only=False,
            )
        )
        final_mounts.append(
            Mount(
                target="/app/storage",
                source=os.path.join(HOST_REPO_PATH, "storage").replace("\\", "/"),
                type="bind",
                read_only=False,
            )
        )

    return DockerOperator(
        task_id=task_id,
        image=ANALYTICS_IMAGE,
        api_version="auto",
        auto_remove="success",
        mount_tmp_dir=False,
        xcom_all=False,
        command=command,
        environment=merged_env,
        docker_url=DOCKER_SOCKET,
        network_mode=DOCKER_NETWORK,
        mounts=final_mounts or None,
        mem_limit="4g",
        cpus=2.0,
        **kwargs,
    )


def register_dataset_func(**kwargs):
    dag_run = kwargs.get("dag_run")
    conf = (dag_run.conf if dag_run and getattr(dag_run, "conf", None) else {}) or {}
    dataset_id = conf.get("dataset_id") or "sample_raster"
    source_path = conf.get("source_path") or (kwargs.get("params") or {}).get("default_raster_path") or DEFAULT_RASTER_PATH
    print(f"Registering dataset {dataset_id} from {source_path}")
    return dataset_id


def ingest_raster_to_storage(**kwargs):
    dag_run = kwargs.get("dag_run")
    conf = (dag_run.conf if dag_run and getattr(dag_run, "conf", None) else {}) or {}
    dataset_id = conf.get("dataset_id") or "sample_raster"
    source_path = conf.get("source_path") or (kwargs.get("params") or {}).get("default_raster_path") or DEFAULT_RASTER_PATH

    # Allow local paths relative to the mounted repo root.
    local_path = Path(source_path)
    if not local_path.is_absolute():
        local_path = Path(REPO_ROOT) / source_path

    storage = get_storage_backend()
    file_id, dest_path = ingest_raster_with_dest_path(storage, str(local_path), destination_dir=f"raw/{dataset_id}")

    if isinstance(storage, S3Storage):
        uri = f"s3://{storage.bucket_name}/{dest_path}"
    else:
        uri = dest_path

    print(f"Ingested raster file_id={file_id} -> {uri}")
    return uri

dag = DAG(
    "raster_embeddings",
    default_args=default_args,
    description="Ingest, tile, embed, and QA rasters into MinIO/S3.",
    schedule=None,
    catchup=False,
    tags=["raster", "embeddings"],
    params={"default_raster_path": DEFAULT_RASTER_PATH},
)

with dag:
    # Params are read from dag_run.conf if provided, else from params defaults.
    dataset_id_tmpl = "{{ (dag_run.conf.get('dataset_id') if dag_run and dag_run.conf else 'sample_raster') or 'sample_raster' }}"
    source_path_tmpl = "{{ (dag_run.conf.get('source_path') if dag_run and dag_run.conf else params.default_raster_path) or params.default_raster_path }}"

    register = PythonOperator(
        task_id="register_dataset",
        python_callable=register_dataset_func
    )

    ingest_task = PythonOperator(
        task_id="ingest_raster",
        python_callable=ingest_raster_to_storage,
    )
    
    # We will use the analytics image for all tasks for simplicity, 
    # or use raster-processing for tiling if it's lighter. 
    # But analytics image has all deps (rasterio etc), so let's stick to it.
    
    # 1. Tile Raster
    # Re-using the analytics image because it has rasterio + s3fs
    tile_task = get_embeddings_operator(
        task_id="tile_raster",
        command=[
            "python",
            "/app/scripts/tile_raster.py",
            "--input",
            "{{ ti.xcom_pull(task_ids='ingest_raster') | string | trim }}",
            "--output_dir",
            f"s3://{PROCESSING_ENV.get('S3_BUCKET', 'raster-data')}/tiles/{dataset_id_tmpl}",
            "--fs_args",
            json.dumps(PROCESSING_ENV),
        ],
    )

    # 2. Compute Embeddings
    embed_cmd = [
        "python",
        "/app/scripts/compute_embeddings.py",
        "--tiles_dir",
        f"s3://{PROCESSING_ENV.get('S3_BUCKET', 'raster-data')}/tiles/{dataset_id_tmpl}",
        "--output_dir",
        f"s3://{PROCESSING_ENV.get('S3_BUCKET', 'raster-data')}/embeddings/{dataset_id_tmpl}",
        "--fs_args",
        json.dumps(PROCESSING_ENV),
    ]
    embed_mounts = []
    if DINO_MODEL_HOST_PATH:
        host_dir = os.path.dirname(DINO_MODEL_HOST_PATH).replace("\\", "/")
        model_name = os.path.basename(DINO_MODEL_HOST_PATH)
        model_path_in_container = f"{DINO_MODEL_CONTAINER_DIR.rstrip('/')}/{model_name}"
        embed_cmd += ["--model_path", model_path_in_container]
        embed_mounts.append(
            Mount(
                source=host_dir,
                target=DINO_MODEL_CONTAINER_DIR,
                type="bind",
                read_only=True,
            )
        )

    embed_task = get_embeddings_operator(
        task_id="compute_embeddings",
        command=embed_cmd,
        mounts=embed_mounts or None,
        # GPU support if needed: device_requests=[...]
    )

    qa_task = get_embeddings_operator(
        task_id="qa_metrics",
        command=[
            "python", "/app/scripts/qa_metrics.py",
            "--tiles_dir",
            f"s3://{PROCESSING_ENV.get('S3_BUCKET', 'raster-data')}/tiles/{dataset_id_tmpl}",
            "--embeddings_dir",
            f"s3://{PROCESSING_ENV.get('S3_BUCKET', 'raster-data')}/embeddings/{dataset_id_tmpl}",
            "--fs_args",
            json.dumps(PROCESSING_ENV),
        ],
    )

    register >> ingest_task >> tile_task >> embed_task >> qa_task
