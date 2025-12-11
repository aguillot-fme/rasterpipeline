from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# A minimalist DAG that exercises a three-step workflow to validate
# that the Airflow environment is configured correctly.

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 0,
}


def _start_task() -> str:
    """Return a simple status message."""
    return "started"


def _process_task(ti) -> str:
    """Append an extra marker to the start message."""
    start_message = ti.xcom_pull(task_ids="start_task")
    return f"{start_message} -> processed"


def _finish_task(ti) -> None:
    """Log the processed message; returning None is fine for tests."""
    processed_message = ti.xcom_pull(task_ids="process_task")
    print(f"Simple DAG finished with payload: {processed_message}")


with DAG(
    dag_id="simple_example",
    default_args=default_args,
    description="Simple example DAG for unit tests",
    schedule=None,
    catchup=False,
) as dag:
    start_task = PythonOperator(
        task_id="start_task",
        python_callable=_start_task,
    )

    process_task = PythonOperator(
        task_id="process_task",
        python_callable=_process_task,
    )

    finish_task = PythonOperator(
        task_id="finish_task",
        python_callable=_finish_task,
    )

    start_task >> process_task >> finish_task
