import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
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

DEFAULT_POINTCLOUD_PATH = os.getenv("DEFAULT_POINTCLOUD_PATH", "synthetic://sample.las")
POINTCLOUD_IMAGE = os.getenv("POINTCLOUD_IMAGE", "pointcloud-processing:latest")
PROCESSING_NETWORK = os.getenv("DOCKER_NETWORK", "rasterpipeline_default")
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET") or os.getenv("DOCKER_HOST") or "unix://var/run/docker.sock"
DEV_CODE_MOUNT = os.getenv("DEV_CODE_MOUNT", "").lower() in {"1", "true", "yes"}
HOST_REPO_PATH = os.getenv("HOST_REPO_PATH")

PROCESSING_ENV = {
    "STORAGE_TYPE": os.getenv("STORAGE_TYPE", "local"),
    "S3_BUCKET": os.getenv("S3_BUCKET", "raster-data"),
    "S3_PREFIX": os.getenv("S3_PREFIX", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL"),
}
PROCESSING_ENV = {k: v for k, v in PROCESSING_ENV.items() if v is not None}


def get_pointcloud_operator(task_id, command, environment=None, **kwargs):
    merged_env = {**PROCESSING_ENV, **(environment or {})}
    mounts = []
    if DEV_CODE_MOUNT and HOST_REPO_PATH:
        mounts.append(Mount(target="/opt/airflow/dags/repo", source=HOST_REPO_PATH, type="bind", read_only=False))
    return DockerOperator(
        task_id=task_id,
        image=POINTCLOUD_IMAGE,
        api_version="auto",
        auto_remove="success",
        mount_tmp_dir=False,
        xcom_all=False,
        command=command,
        docker_url=DOCKER_SOCKET,
        network_mode=PROCESSING_NETWORK,
        environment=merged_env,
        mounts=mounts or None,
        mem_limit="2g",
        cpus=1.0,
        **kwargs,
    )


dag = DAG(
    "pointcloud_enrich",
    default_args=default_args,
    description="Ingest and enrich a point cloud (features/attributes) into MinIO/S3.",
    schedule=None,
    catchup=False,
    params={"default_pointcloud_path": DEFAULT_POINTCLOUD_PATH},
)


with dag:
    ingest_task = get_pointcloud_operator(
        task_id="ingest_task",
        command=[
            "python",
            "-m",
            "pointcloud_processing.ingest",
            "--source-path",
            "{{ (dag_run.conf.get('source_path') if dag_run and dag_run.conf else params.default_pointcloud_path) or params.default_pointcloud_path }}",
            "--destination-dir",
            "pointcloud/raw",
        ],
        params={"default_pointcloud_path": DEFAULT_POINTCLOUD_PATH},
        do_xcom_push=True,
    )

    enrich_task = get_pointcloud_operator(
        task_id="enrich_task",
        command=[
            "python",
            "-m",
            "pointcloud_processing.enrich",
            "--file-id",
            "{{ ti.xcom_pull(task_ids='ingest_task') | string | trim }}",
            "--raw-base-dir",
            "pointcloud/raw",
            "--enriched-base-dir",
            "pointcloud/enriched",
        ],
        do_xcom_push=True,
    )

    ingest_task >> enrich_task
