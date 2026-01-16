import os
from .base import StorageBackend
from .local import LocalStorage
from .s3 import S3Storage

__all__ = ["StorageBackend", "LocalStorage", "S3Storage", "get_storage_backend"]

def get_storage_backend() -> StorageBackend:
    """
    Get the configured storage backend.
    Defaults to LocalStorage if STORAGE_TYPE is not 's3'.
    """
    storage_type = os.getenv("STORAGE_TYPE", "local").lower()
    
    if storage_type in {"s3", "minio"}:
        return S3Storage(
            bucket_name=os.getenv("S3_BUCKET", "raster-data"),
            prefix=os.getenv("S3_PREFIX", "")
        )
    else:
        # Default to local
        base_path = os.getenv("LOCAL_STORAGE_PATH", "/opt/airflow/dags/repo")
        return LocalStorage(base_path=base_path)
