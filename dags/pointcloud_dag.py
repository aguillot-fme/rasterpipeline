from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
import json
import os

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

# Default Pipeline JSON (Normal + DBSCAN)
DEFAULT_PIPELINE = {
    "pipeline": [
        {"type": "readers.las", "filename": "__INPUT_FILE__"},
        {"type": "filters.normal", "knn": 8},
        {"type": "filters.dbscan", "min_points": 10, "eps": 2.0, "dimensions": "X,Y,Z"},
        {"type": "writers.las", "filename": "__OUTPUT_FILE__", "extra_dims": "all", "minor_version": "4"}
    ]
}

with DAG(
    dag_id="pointcloud_processing",
    default_args=default_args,
    description="Process pointcloud data using PDAL",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["pointcloud", "pdal"],
    # Define parameters with defaults
    params={
        "input_file": "data/raw/sample.las",
        "output_file": "data/processed/sample_processed.las",
        "pipeline": json.dumps(DEFAULT_PIPELINE)
    }
) as dag:

    # Define paths using Template variables (Jinja)
    # The DockerOperator resolves these at runtime.
    input_path = "/opt/airflow/{{ params.input_file }}"
    output_path = "/opt/airflow/{{ params.output_file }}"
    pipeline_json_str = "{{ params.pipeline }}"

    run_pdal = DockerOperator(
        task_id="run_pdal_pipeline",
        image="pointcloud-worker:latest",
        api_version="auto",
        auto_remove=True,
        command=[
            "conda", "run", "-n", "pdal", "python",
            "/opt/airflow/scripts/pdal_processor.py",
            "--input", input_path,
            "--output", output_path,
            "--pipeline", pipeline_json_str
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            # Mount the project root to /opt/airflow
            # Update 'source' to match the actual host path if different
            Mount(source="d:/rasterpipeline", target="/opt/airflow", type="bind"),
        ],
        working_dir="/opt/airflow",
        # Ensure user permissions match if needed, though usually root in container is fine for temp jobs
        # user="airflow", 
    )

    run_pdal
