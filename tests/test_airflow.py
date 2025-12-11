import pytest
from airflow.models import DagBag
import os

def test_dag_import():
    """
    Verify that the DAG can be imported without errors.
    """
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"DAG import failures: {dag_bag.import_errors}"
    assert "raster_ingest" in dag_bag.dags

    dag = dag_bag.dags["raster_ingest"]
    # ingest, branch, validate, transform, metrics, write
    assert len(dag.tasks) == 6
