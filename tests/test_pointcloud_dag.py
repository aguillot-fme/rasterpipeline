import pytest


def test_pointcloud_enrich_dag_structure():
    try:
        from airflow.models import DagBag
    except Exception:
        pytest.skip("Apache Airflow not installed in this environment")

    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert dag_bag.import_errors == {}
    assert "pointcloud_enrich" in dag_bag.dags

    dag = dag_bag.dags["pointcloud_enrich"]
    assert len(dag.tasks) == 2
    assert dag.params.get("default_pointcloud_path")

    ingest = dag.get_task("ingest_task")
    enrich = dag.get_task("enrich_task")
    assert enrich.upstream_task_ids == {ingest.task_id}
