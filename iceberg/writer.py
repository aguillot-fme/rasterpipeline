import logging
from typing import Dict, Any

from airflow.providers.apache.iceberg.hooks.iceberg import IcebergHook

logger = logging.getLogger(__name__)


class IcebergWriter:
    """
    Thin wrapper around Airflow's IcebergHook to keep DAG import-compatible with
    the Airflow SQLAlchemy constraint.
    """

    def __init__(self, iceberg_conn_id: str = "iceberg_default"):
        self.hook = IcebergHook(iceberg_conn_id=iceberg_conn_id)

    def write_raster_record(self, record: Dict[str, Any]):
        """
        Placeholder write using the provider hook. This assumes the connection
        points to a REST-compatible Iceberg catalog. In this dev setup we just
        log the intent to avoid hard dependency on an external client library.
        """
        logger.info("Would write raster record to Iceberg via hook %s: %s", self.hook.conn_id, record)

    def write_metadata_record(self, record: Dict[str, Any]):
        """
        Placeholder metadata write; mirrors write_raster_record for symmetry.
        """
        logger.info("Would write metadata record to Iceberg via hook %s: %s", self.hook.conn_id, record)
