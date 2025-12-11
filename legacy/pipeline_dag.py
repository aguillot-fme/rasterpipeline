from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from storage import get_storage_backend
from processing.ingest import ingest_raster
from processing.validate import validate_raster
from processing.transform import calculate_ndvi
from processing.metrics import compute_raster_metrics
from iceberg.writer import IcebergWriter

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'raster_pipeline',
    default_args=default_args,
    description='A simple raster processing pipeline',
    schedule=None,
)

def get_storage():
    return get_storage_backend()

def run_ingest(**kwargs):
    storage = get_storage()
    # In a real scenario, source_path comes from dag_run conf
    # For Docker, we might need to download from a URL or assume it's already in a landing bucket
    # Let's assume the input is a URL or a path in a landing bucket
    source_path = kwargs['dag_run'].conf.get('source_path')
    if not source_path:
        raise ValueError("source_path is required")
        
    # If source_path is local (e.g. mounted volume), S3Storage.write_file expects bytes
    # If it's a URL, we'd fetch it.
    # For this demo, let's assume we are simulating ingestion from a local file 
    # that we uploaded to the container or a mounted volume.
    
    file_id = ingest_raster(storage, source_path, destination_dir="raw")
    return file_id

def run_validate(**kwargs):
    ti = kwargs['ti']
    file_id = ti.xcom_pull(task_ids='ingest_task')
    storage = get_storage()
    
    # Find the file
    files = storage.list_files("raw", pattern=f"*{file_id}*/*.tif")
    if not files:
         raise FileNotFoundError(f"File for ID {file_id} not found")
    file_path = files[0]
    
    validate_raster(storage, file_path)
    return file_path

def run_transform(**kwargs):
    ti = kwargs['ti']
    file_path = ti.xcom_pull(task_ids='validate_task')
    storage = get_storage()
    
    output_path = file_path.replace("raw", "processed").replace(".tif", "_ndvi.tif")
    
    calculate_ndvi(storage, file_path, file_path, output_path)
    return output_path

def run_metrics(**kwargs):
    ti = kwargs['ti']
    file_path = ti.xcom_pull(task_ids='validate_task') # Metrics on raw
    processed_path = ti.xcom_pull(task_ids='transform_task') # Metrics on processed
    
    storage = get_storage()
    
    raw_metrics = compute_raster_metrics(storage, file_path)
    processed_metrics = compute_raster_metrics(storage, processed_path)
    
    return {'raw': raw_metrics, 'processed': processed_metrics}

def run_write_iceberg(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='metrics_task')
    file_id = ti.xcom_pull(task_ids='ingest_task')
    
    writer = IcebergWriter()
    
    # Write record
    record = {
        "file_id": file_id,
        "filename": "demo.tif", # Simplified
        "ingested_at": datetime.now(),
        "width": metrics['raw']['width'],
        "height": metrics['raw']['height'],
        "crs": metrics['raw']['crs'],
        "min_val": metrics['raw']['min'],
        "max_val": metrics['raw']['max'],
        "mean_val": metrics['raw']['mean'],
        "std_val": metrics['raw']['std'],
        "s3_path": f"raw/{file_id}/demo.tif" # Simplified path
    }
    
    writer.write_raster_record(record)

with dag:
    ingest_task = PythonOperator(
        task_id='ingest_task',
        python_callable=run_ingest,
    )

    validate_task = PythonOperator(
        task_id='validate_task',
        python_callable=run_validate,
    )

    transform_task = PythonOperator(
        task_id='transform_task',
        python_callable=run_transform,
    )

    metrics_task = PythonOperator(
        task_id='metrics_task',
        python_callable=run_metrics,
    )
    
    write_task = PythonOperator(
        task_id='write_task',
        python_callable=run_write_iceberg,
    )

    ingest_task >> validate_task >> transform_task >> metrics_task >> write_task
