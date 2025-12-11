import boto3
import os
import pytest
from storage.s3 import S3Storage

@pytest.mark.skipif(not os.getenv("S3_ENDPOINT_URL"), reason="MinIO not available")
def test_s3_connection():
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    )
    
    # List buckets
    response = s3.list_buckets()
    assert response['ResponseMetadata']['HTTPStatusCode'] == 200

def test_s3_storage_backend():
    # This requires MinIO to be running
    if not os.getenv("S3_ENDPOINT_URL"):
        pytest.skip("MinIO not configured")
        
    storage = S3Storage(bucket_name="test-bucket")
    storage.write_file("test.txt", b"hello world")
    
    assert storage.exists("test.txt")
    assert storage.read_file("test.txt") == b"hello world"
    
    files = storage.list_files("")
    assert "s3://test-bucket/test.txt" in files
    
    storage.delete_file("test.txt")
    assert not storage.exists("test.txt")
