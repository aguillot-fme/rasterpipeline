import argparse

from storage import StorageBackend, get_storage_backend
from .helpers.raster_utils import read_raster_metadata
from .helpers import find_raster_by_id

def validate_raster(storage: StorageBackend, file_path: str) -> bool:
    """
    Validate a raster file in storage.
    
    Args:
        storage: Storage backend.
        file_path: Path to the file in storage.
        
    Returns:
        True if valid, raises ValueError otherwise.
    """
    if not storage.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    data = storage.read_file(file_path)
    
    try:
        profile = read_raster_metadata(data)
        
        # Add specific validation logic here
        if profile['width'] <= 0 or profile['height'] <= 0:
            raise ValueError("Invalid raster dimensions")
            
        if not profile['crs']:
             # Warning or error depending on strictness
             pass
             
        return True
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate a raster already stored in the backend.")
    parser.add_argument("--file-path", help="Direct path to the raster inside storage.")
    parser.add_argument("--file-id", help="Ingestion file id to resolve raster path automatically.")
    args = parser.parse_args()

    if not args.file_path and not args.file_id:
        raise SystemExit("Provide either --file-path or --file-id.")

    storage = get_storage_backend()
    target_path = args.file_path or find_raster_by_id(storage, args.file_id, base_dir="raw", pattern="*.tif")
    validate_raster(storage, target_path)
    print(target_path)


if __name__ == "__main__":
    main()
