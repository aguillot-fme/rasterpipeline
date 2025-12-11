from storage import StorageBackend
from .helpers.raster_utils import read_raster_metadata

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
