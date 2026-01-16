import pytest

pytest.importorskip("airflow.models")
from airflow.models import DagBag


def test_simple_example_dag_structure():
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert dag_bag.import_errors == {}
    assert "simple_example" in dag_bag.dags

    dag = dag_bag.dags["simple_example"]
    assert len(dag.tasks) == 3

    start = dag.get_task("start_task")
    process = dag.get_task("process_task")
    finish = dag.get_task("finish_task")

    assert process.upstream_task_ids == {start.task_id}
    assert finish.upstream_task_ids == {process.task_id}
