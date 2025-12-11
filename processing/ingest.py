import argparse
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse

from storage import StorageBackend, get_storage_backend
from .helpers.raster_utils import read_raster_metadata

def ingest_raster(
    storage: StorageBackend,
    source_path: str,
    destination_dir: str = "raw"
) -> str:
    """
    Ingest a raster file into the system.
    
    Args:
        storage: The storage backend to use.
        source_path: Path to the source file (could be local path or temp path).
        destination_dir: Directory in storage to save the file.
        
    Returns:
        The ID of the ingested file.
    """
    # Generate a unique ID
    file_id = str(uuid.uuid4())
    
    parsed = urlparse(source_path)
    if parsed.scheme in ("", "file"):
        local_path = source_path if parsed.scheme else os.path.abspath(source_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Source raster not found: {source_path}")
        with open(local_path, "rb") as f:
            data = f.read()
        filename = os.path.basename(local_path)
    else:
        # Delegate remote reads to the configured storage backend (e.g., S3/MinIO)
        data = storage.read_file(source_path)
        filename = os.path.basename(parsed.path.rstrip("/"))
        if not filename:
            raise ValueError(f"Cannot derive filename from source_path: {source_path}")
        
    # Validate it's a raster by trying to read metadata
    try:
        _ = read_raster_metadata(data)
    except Exception as e:
        raise ValueError(f"Invalid raster file: {e}")
        
    # Define destination path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = f"{destination_dir}/{timestamp}_{file_id}/{filename}"
    
    # Write to storage
    storage.write_file(dest_path, data)
    
    return file_id


def main():
    parser = argparse.ArgumentParser(description="Ingest a raster into storage.")
    parser.add_argument("--source-path", required=True, help="Local path or URI for the raster to ingest.")
    parser.add_argument(
        "--destination-dir",
        default="raw",
        help="Destination directory/prefix within the storage backend.",
    )
    args = parser.parse_args()

    storage = get_storage_backend()
    file_id = ingest_raster(storage, args.source_path, args.destination_dir)
    print(file_id)


if __name__ == "__main__":
    main()
