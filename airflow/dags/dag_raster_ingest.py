import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from iceberg.writer import IcebergWriter

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

DEFAULT_RASTER_PATH = os.getenv(
    "DEFAULT_RASTER_PATH",
    "s3://raster-data/raw/20251210_121155_704c36a2-73ae-4794-86ca-ebcc5bd6d5f3/HEL_0_0.tif",
)
PROCESSING_IMAGE = os.getenv("PROCESSING_IMAGE", "raster-processing:latest")
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

dag = DAG(
    'raster_ingest',
    default_args=default_args,
    description='A flexible raster processing pipeline with branching',
    schedule=None,
    catchup=False,
    params={"default_raster_path": DEFAULT_RASTER_PATH},
)


def get_processing_operator(task_id, command, environment=None, **kwargs):
    merged_env = {**PROCESSING_ENV, **(environment or {})}
    mounts = []
    if DEV_CODE_MOUNT and HOST_REPO_PATH:
        mounts.append(Mount(target="/opt/airflow/dags/repo", source=HOST_REPO_PATH, type="bind", read_only=False))
    return DockerOperator(
        task_id=task_id,
        image=PROCESSING_IMAGE,
        api_version='auto',
        auto_remove="success",
        mount_tmp_dir=False,
        xcom_all=False,
        command=command,
        docker_url=DOCKER_SOCKET,
        network_mode=PROCESSING_NETWORK,
        environment=merged_env,
        mounts=mounts or None,
        mem_limit="512m",
        cpus=1.0,
        **kwargs,
    )

def decide_path(**kwargs):
    conf = kwargs['dag_run'].conf or {}
    skip_validation = conf.get('skip_validation', False)
    
    if skip_validation:
        return 'transform_task'
    else:
        return 'validate_task'

def run_write_iceberg(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='metrics_task')
    file_id = ti.xcom_pull(task_ids='ingest_task')
    file_id_str = str(file_id).strip()

    if isinstance(metrics, str):
        metrics = metrics.strip()
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            print(f"Unable to decode metrics payload: {metrics}")
            metrics = {"raw": {}, "processed": {}}
    metrics_raw = metrics.get('raw', {}) if isinstance(metrics, dict) else {}
    
    writer = IcebergWriter()
    
    record = {
        "file_id": file_id_str,
        "filename": "demo.tif", 
        "ingested_at": datetime.now(),
        "width": metrics_raw.get('width'),
        "height": metrics_raw.get('height'),
        "crs": metrics_raw.get('crs'),
        "min_val": metrics_raw.get('min'),
        "max_val": metrics_raw.get('max'),
        "mean_val": metrics_raw.get('mean'),
        "std_val": metrics_raw.get('std'),
        "s3_path": f"raw/{file_id_str}/demo.tif"
    }
    
    writer.write_raster_record(record)

with dag:
    ingest_task = get_processing_operator(
        task_id='ingest_task',
        command=[
            "python",
            "-m",
            "processing.ingest",
            "--source-path",
            "{{ (dag_run.conf.get('source_path') if dag_run and dag_run.conf else params.default_raster_path) or '" + DEFAULT_RASTER_PATH + "' }}",
            "--destination-dir",
            "raw",
        ],
        params={"default_raster_path": DEFAULT_RASTER_PATH},
        do_xcom_push=True,
    )

    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=decide_path,
    )

    validate_task = get_processing_operator(
        task_id='validate_task',
        command=[
            "python",
            "-m",
            "processing.validate",
            "--file-id",
            "{{ ti.xcom_pull(task_ids='ingest_task') | string | trim }}",
        ],
        do_xcom_push=True,
    )

    transform_task = get_processing_operator(
        task_id='transform_task',
        command=[
            "python",
            "-m",
            "processing.transform",
            "--file-id",
            "{{ ti.xcom_pull(task_ids='ingest_task') | string | trim }}",
        ],
        do_xcom_push=True,
        trigger_rule='none_failed_min_one_success', # Runs if branch skipped validate OR validate passed
    )

    metrics_task = get_processing_operator(
        task_id='metrics_task',
        command=[
            "python",
            "-m",
            "processing.metrics",
            "--file-id",
            "{{ ti.xcom_pull(task_ids='ingest_task') | string | trim }}",
            "--processed-path",
            "{{ ti.xcom_pull(task_ids='transform_task') | string | trim }}",
        ],
        do_xcom_push=True,
    )
    
    write_task = PythonOperator(
        task_id='write_task',
        python_callable=run_write_iceberg,
    )

    # Topology
    ingest_task >> branch_task
    branch_task >> validate_task >> transform_task
    branch_task >> transform_task # The skip path
    
    transform_task >> metrics_task >> write_task
