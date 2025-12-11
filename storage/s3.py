from typing import List, Optional
from urllib.parse import urlparse

import boto3
import fnmatch
import os
from .base import StorageBackend


class S3Storage(StorageBackend):
    """
    Storage backend implementation for AWS S3 (or MinIO).
    """

    def __init__(self, bucket_name: str, prefix: str = "", endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")

        # Initialize boto3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

        # Ensure bucket exists
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except Exception:
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
            except Exception as e:
                print(f"Could not create bucket {self.bucket_name}: {e}")

    def _full_key(self, path: str) -> str:
        clean = path.lstrip("/")
        if self.prefix:
            prefix = self.prefix.rstrip("/")
            if not clean.startswith(prefix):
                return f"{prefix}/{clean}"
        return clean

    def _bucket_and_key(self, path: str) -> (str, str):
        if path.startswith("s3://"):
            parsed = urlparse(path)
            bucket = parsed.netloc or self.bucket_name
            key = parsed.path.lstrip("/")
            return bucket, key
        return self.bucket_name, self._full_key(path)

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        bucket, prefix = self._bucket_and_key(path)
        prefix = prefix.rstrip("/")
        paginator = self.s3.get_paginator("list_objects_v2")
        files: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                full = f"s3://{bucket}/{key}"
                if pattern:
                    if fnmatch.fnmatch(key, pattern):
                        files.append(full)
                else:
                    files.append(full)
        return files

    def read_file(self, path: str) -> bytes:
        bucket, key = self._bucket_and_key(path)
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def write_file(self, path: str, data: bytes) -> None:
        bucket, key = self._bucket_and_key(path)
        self.s3.put_object(Bucket=bucket, Key=key, Body=data)

    def exists(self, path: str) -> bool:
        bucket, key = self._bucket_and_key(path)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def delete_file(self, path: str) -> None:
        bucket, key = self._bucket_and_key(path)
        self.s3.delete_object(Bucket=bucket, Key=key)

    def get_local_path(self, path: str) -> str:
        import tempfile

        bucket, key = self._bucket_and_key(path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        self.s3.download_file(bucket, key, tmp.name)
        tmp.close()
        return tmp.name
