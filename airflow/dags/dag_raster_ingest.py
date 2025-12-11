from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
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
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'raster_ingest',
    default_args=default_args,
    description='A flexible raster processing pipeline with branching',
    schedule=None,
    catchup=False,
)

DEFAULT_RASTER_PATH = os.getenv(
    "DEFAULT_RASTER_PATH",
    "s3://raster-data/raw/20251210_121155_704c36a2-73ae-4794-86ca-ebcc5bd6d5f3/HEL_0_0.tif",
)

def get_storage():
    return get_storage_backend()

def run_ingest(**kwargs):
    storage = get_storage()
    # Flexible input: expecting conf['source_path']
    conf = kwargs['dag_run'].conf or {}
    source_path = conf.get('source_path')
    
    if not source_path:
        # Fallback for manual trigger without conf for testing
        source_path = DEFAULT_RASTER_PATH
        print(f"Warning: No source_path in conf, using default: {source_path}")

    # In a real app, we might download from a URL here if source_path is http://...
    
    file_id = ingest_raster(storage, source_path, destination_dir="raw")
    return file_id

def decide_path(**kwargs):
    conf = kwargs['dag_run'].conf or {}
    skip_validation = conf.get('skip_validation', False)
    
    if skip_validation:
        return 'transform_task'
    else:
        return 'validate_task'

def run_validate(**kwargs):
    ti = kwargs['ti']
    file_id = ti.xcom_pull(task_ids='ingest_task')
    storage = get_storage()
    
    files = storage.list_files("raw", pattern=f"*{file_id}*/*.tif")
    if not files:
         raise FileNotFoundError(f"File for ID {file_id} not found")
    file_path = files[0]
    
    validate_raster(storage, file_path)
    return file_path

def run_transform(**kwargs):
    ti = kwargs['ti']
    # If we skipped validation, we pull from ingest. If we validated, pull from validate.
    # XComArg would handle this cleaner, but for explicit logic:
    file_id = ti.xcom_pull(task_ids='ingest_task')
    
    # Re-resolve file path since we might have skipped validation step which passed it along
    storage = get_storage()
    files = storage.list_files("raw", pattern=f"*{file_id}*/*.tif")
    if not files:
         raise FileNotFoundError(f"File for ID {file_id} not found")
    file_path = files[0]

    output_path = file_path.replace("raw", "processed").replace(".tif", "_ndvi.tif")
    calculate_ndvi(storage, file_path, file_path, output_path)
    return output_path

def run_metrics(**kwargs):
    ti = kwargs['ti']
    file_id = ti.xcom_pull(task_ids='ingest_task')
    # Transform is always run (it's the join point or sequential)
    processed_path = ti.xcom_pull(task_ids='transform_task') 
    
    storage = get_storage()
    files = storage.list_files("raw", pattern=f"*{file_id}*/*.tif")
    file_path = files[0]
    
    raw_metrics = compute_raster_metrics(storage, file_path)
    processed_metrics = compute_raster_metrics(storage, processed_path)
    
    return {'raw': raw_metrics, 'processed': processed_metrics}

def run_write_iceberg(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='metrics_task')
    file_id = ti.xcom_pull(task_ids='ingest_task')
    
    writer = IcebergWriter()
    
    record = {
        "file_id": file_id,
        "filename": "demo.tif", 
        "ingested_at": datetime.now(),
        "width": metrics['raw']['width'],
        "height": metrics['raw']['height'],
        "crs": metrics['raw']['crs'],
        "min_val": metrics['raw']['min'],
        "max_val": metrics['raw']['max'],
        "mean_val": metrics['raw']['mean'],
        "std_val": metrics['raw']['std'],
        "s3_path": f"raw/{file_id}/demo.tif"
    }
    
    writer.write_raster_record(record)

with dag:
    ingest_task = PythonOperator(
        task_id='ingest_task',
        python_callable=run_ingest,
    )

    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=decide_path,
    )

    validate_task = PythonOperator(
        task_id='validate_task',
        python_callable=run_validate,
    )

    transform_task = PythonOperator(
        task_id='transform_task',
        python_callable=run_transform,
        trigger_rule='none_failed_min_one_success', # Runs if branch skipped validate OR validate passed
    )

    metrics_task = PythonOperator(
        task_id='metrics_task',
        python_callable=run_metrics,
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
