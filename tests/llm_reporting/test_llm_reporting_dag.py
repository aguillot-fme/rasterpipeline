import importlib.util
import os
import sys

import pytest

pytest.importorskip("airflow")


def _load_dag_module(monkeypatch, **env_overrides):
    for key, value in env_overrides.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    path = os.path.join("airflow", "dags", "llm_reporting_dag.py")
    spec = importlib.util.spec_from_file_location("llm_reporting_dag", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_reporting_dag"] = module
    spec.loader.exec_module(module)
    return module


def test_llm_reporting_dag_structure(monkeypatch):
    module = _load_dag_module(monkeypatch, STORAGE_TYPE="s3")
    dag = module.dag
    assert dag.dag_id == "duckdb_llm_reporting"
    assert dag.get_task("prepare_inputs")
    assert dag.get_task("profile_dataset")
    assert dag.get_task("plan_questions")
    assert dag.get_task("generate_sql")
    assert dag.get_task("execute_queries")
    assert dag.get_task("generate_report")
    assert dag.get_task("write_manifest")
