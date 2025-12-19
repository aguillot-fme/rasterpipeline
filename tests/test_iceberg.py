import pytest

pytest.importorskip("airflow.providers.apache.iceberg")
from airflow.providers.apache.iceberg.hooks.iceberg import IcebergHook


def test_iceberg_hook_importable():
    """
    Smoke test that the Airflow Iceberg hook is present and can be instantiated
    without touching an external catalog.
    """
    hook = IcebergHook()
    assert hook.conn_id == hook.default_conn_name
    assert hook.conn_type == "iceberg"
