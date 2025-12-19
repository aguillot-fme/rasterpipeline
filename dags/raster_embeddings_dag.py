from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
import json
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

# Environment variables for containers (targeting MinIO)
# These should ideally come from Airflow Connections/Variables
ENV_VARS = {
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin",
    "AWS_ENDPOINT_URL": "http://minio:9000",
    "AWS_DEFAULT_REGION": "us-east-1",
    "S3_ENDPOINT_URL": "http://minio:9000" # For some libs
}

def register_dataset_func(**kwargs):
    dataset_id = kwargs['params'].get('dataset_id')
    source_path = kwargs['params'].get('source_path')
    print(f"Registering dataset {dataset_id} from {source_path}")
    # TODO: Write to MinIO manifest or DB
    return dataset_id

with DAG(
    dag_id="raster_embeddings",
    default_args=default_args,
    description="Tile rasters and compute embeddings",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["raster", "embeddings"],
    params={
        "dataset_id": "sample_raster",
        "source_path": "data/raw/sample.tif"  # Local host path or URL
    }
) as dag:

    register = PythonOperator(
        task_id="register_dataset",
        python_callable=register_dataset_func
    )

    # Ingest: For now, assume data is mounted or we copy it. 
    # Let's assume the source is reachable via mounted /opt/airflow connection
    # or we use a simple script to 'upload' to MinIO if it's external.
    # For MVP, let's skip complex ingest and assume source_path is accessible.
    
    # We will use the analytics image for all tasks for simplicity, 
    # or use raster-processing for tiling if it's lighter. 
    # But analytics image has all deps (rasterio etc), so let's stick to it.
    
    # 1. Tile Raster
    # Re-using the analytics image because it has rasterio + s3fs
    tile_task = DockerOperator(
        task_id="tile_raster",
        image="analytics-embeddings:latest",
        api_version="auto",
        auto_remove=True,
        command=[
            "python", "/app/scripts/tile_raster.py",
            "--input", "{{ params.source_path }}", 
            "--output_dir", "s3://raster-data/tiles/{{ params.dataset_id }}",
            "--fs_args", json.dumps(ENV_VARS) # Pass env vars as json if script needs explicit config
        ],
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            # Mount scripts so we don't have to rebuild image for script changes
            Mount(source="d:/rasterpipeline/scripts", target="/app/scripts", type="bind"),
            # Mount data for local testing (optional, if source_path is local)
            Mount(source="d:/rasterpipeline/data", target="/app/data", type="bind"),
        ],
    )

    # 2. Compute Embeddings
    embed_task = DockerOperator(
        task_id="compute_embeddings",
        image="analytics-embeddings:latest",
        api_version="auto",
        auto_remove=True,
        command=[
            "python", "/app/scripts/compute_embeddings.py",
            "--tiles_dir", "s3://raster-data/tiles/{{ params.dataset_id }}",
            "--output_dir", "s3://raster-data/embeddings/{{ params.dataset_id }}",
             "--fs_args", json.dumps(ENV_VARS)
        ],
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(source="d:/rasterpipeline/scripts", target="/app/scripts", type="bind"),
        ],
        # GPU support if needed: device_requests=[...]
    )

    register >> tile_task >> embed_task
