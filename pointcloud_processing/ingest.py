import argparse
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse

from storage import StorageBackend, get_storage_backend


SUPPORTED_EXTENSIONS = {".las", ".csv"}


def _read_source(storage: StorageBackend, source_path: str) -> tuple[bytes, str]:
    parsed = urlparse(source_path)
    if parsed.scheme in ("", "file"):
        local_path = source_path if parsed.scheme else os.path.abspath(source_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Source pointcloud not found: {source_path}")
        with open(local_path, "rb") as f:
            data = f.read()
        filename = os.path.basename(local_path)
        return data, filename

    data = storage.read_file(source_path)
    filename = os.path.basename(parsed.path.rstrip("/"))
    if not filename:
        raise ValueError(f"Cannot derive filename from source_path: {source_path}")
    return data, filename


def ingest_pointcloud(
    storage: StorageBackend,
    source_path: str,
    destination_dir: str = "pointcloud/raw",
) -> str:
    file_id = str(uuid.uuid4())

    data, filename = _read_source(storage, source_path)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported pointcloud extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = f"{destination_dir}/{timestamp}_{file_id}/{filename}"
    storage.write_file(dest_path, data)
    return file_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a point cloud into storage.")
    parser.add_argument("--source-path", required=True, help="Local path or URI for the point cloud to ingest.")
    parser.add_argument(
        "--destination-dir",
        default="pointcloud/raw",
        help="Destination directory/prefix within the storage backend.",
    )
    args = parser.parse_args()

    storage = get_storage_backend()
    file_id = ingest_pointcloud(storage, args.source_path, args.destination_dir)
    print(file_id)


if __name__ == "__main__":
    main()

