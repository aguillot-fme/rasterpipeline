import pytest


def test_raster_patch_similarity_dag_structure():
    try:
        from airflow.models import DagBag
    except Exception:
        pytest.skip("Apache Airflow not installed in this environment")

    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert dag_bag.import_errors == {}
    assert "raster_patch_similarity" in dag_bag.dags

    dag = dag_bag.dags["raster_patch_similarity"]
    assert len(dag.tasks) == 2

    prepare = dag.get_task("prepare_inputs")
    similarity = dag.get_task("find_similar_patches")
    assert similarity.upstream_task_ids == {prepare.task_id}
