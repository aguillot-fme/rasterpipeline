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

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Docker Config
ANALYTICS_IMAGE = os.getenv("ANALYTICS_IMAGE", "analytics-embeddings:latest")
PROCESSING_NETWORK = os.getenv("DOCKER_NETWORK", "rasterpipeline_default")
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET") or os.getenv("DOCKER_HOST") or "unix://var/run/docker.sock"
DEV_CODE_MOUNT = os.getenv("DEV_CODE_MOUNT", "").lower() in {"1", "true", "yes"}
HOST_REPO_PATH = os.getenv("HOST_REPO_PATH")

# Shared Environment
COMMON_ENV = {
    "STORAGE_TYPE": os.getenv("STORAGE_TYPE", "s3"),
    "S3_BUCKET": os.getenv("S3_BUCKET", "raster-data"),
    "S3_PREFIX": os.getenv("S3_PREFIX", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
}
COMMON_ENV = {k: v for k, v in COMMON_ENV.items() if v is not None}

STORAGE_TYPE = COMMON_ENV.get("STORAGE_TYPE", "s3").lower()
S3_BUCKET = COMMON_ENV.get("S3_BUCKET", "raster-data")
S3_PREFIX = COMMON_ENV.get("S3_PREFIX", "")
S3_ENDPOINT_URL = COMMON_ENV.get("S3_ENDPOINT_URL", "http://minio:9000")
LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "/opt/airflow/dags/repo")
if STORAGE_TYPE in {"s3", "minio"}:
    fs_args = {
        "AWS_ENDPOINT_URL": S3_ENDPOINT_URL,
        "AWS_ACCESS_KEY_ID": COMMON_ENV.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": COMMON_ENV.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_DEFAULT_REGION": COMMON_ENV.get("AWS_DEFAULT_REGION"),
    }
    fs_args = {k: v for k, v in fs_args.items() if v}
    FS_ARGS_JSON = json.dumps(fs_args)
else:
    FS_ARGS_JSON = "{}"


def _join_s3(prefix: str, *parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p]
    if prefix:
        cleaned = [prefix.strip("/")] + cleaned
    return f"s3://{S3_BUCKET}/{posixpath.join(*cleaned)}"


def _output_dir(base_dir: str, dataset_id: str) -> str:
    if STORAGE_TYPE in {"s3", "minio"}:
        return _join_s3(S3_PREFIX, base_dir, dataset_id)
    return os.path.join(LOCAL_STORAGE_PATH, base_dir, dataset_id)

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


def prepare_inputs(**kwargs):
    dag_run = kwargs.get("dag_run")
    conf = (dag_run.conf if dag_run and getattr(dag_run, "conf", None) else {}) or {}
    dataset_id = conf.get("dataset_id") or "sample_raster"
    run_id = getattr(dag_run, "run_id", None) or ""

    embeddings_dir = conf.get("embeddings_dir") or _output_dir("embeddings", dataset_id)
    output_path = conf.get("output_path")
    if not output_path and run_id:
        output_path = _join_uri(embeddings_dir, run_id, "similarity_results.parquet")
    if output_path is None:
        output_path = ""

    query_paths = conf.get("query_paths")
    if isinstance(query_paths, list):
        query_paths = json.dumps(query_paths)
    elif query_paths is None and conf.get("query_path"):
        query_paths = json.dumps([conf.get("query_path")])
    elif query_paths is None:
        query_paths = ""

    query_coords_raw = conf.get("query_coords")
    query_coords = json.dumps(query_coords_raw) if query_coords_raw else ""
    tiles_dir = conf.get("tiles_dir") or (_output_dir("tiles", dataset_id) if query_coords else "")

    return {
        "dataset_id": dataset_id,
        "embeddings_dir": embeddings_dir,
        "tiles_dir": tiles_dir,
        "output_path": output_path,
        "query_paths": query_paths,
        "query_paths_file": conf.get("query_paths_file") or "",
        "query_dir": conf.get("query_dir") or "",
        "query_coords": query_coords,
        "model_path": conf.get("model_path") or "",
        "top_k": int(conf.get("top_k", 5)),
    }


dag = DAG(
    "raster_patch_similarity",
    default_args=default_args,
    description="Find similar raster patches using DINO embeddings.",
    schedule=None,
    catchup=False,
    tags=["raster", "embeddings", "similarity"],
)


with dag:
    prepare_task = PythonOperator(
        task_id="prepare_inputs",
        python_callable=prepare_inputs,
    )

    similarity_task = get_docker_operator(
        task_id="find_similar_patches",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/find_similar_patches.py",
            "--embeddings_dir",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['embeddings_dir'] }}",
            "--tiles_dir",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['tiles_dir'] }}",
            "--query_paths",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['query_paths'] }}",
            "--query_paths_file",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['query_paths_file'] }}",
            "--query_dir",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['query_dir'] }}",
            "--query_coords",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['query_coords'] }}",
            "--output_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['output_path'] }}",
            "--fs_args",
            FS_ARGS_JSON,
            "--model_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['model_path'] }}",
            "--top_k",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['top_k'] }}",
        ],
    )

    prepare_task >> similarity_task
