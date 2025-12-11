from storage import StorageBackend
from .helpers.raster_utils import read_raster_data
import numpy as np
from typing import Dict, Any

def compute_raster_metrics(storage: StorageBackend, file_path: str) -> Dict[str, Any]:
    """
    Compute statistics for a raster file.
    
    Args:
        storage: Storage backend.
        file_path: Path to the raster file.
        
    Returns:
        Dictionary containing metrics (min, max, mean, std).
    """
    data, profile = read_raster_data(storage.read_file(file_path))
    
    # Mask nodata values if present
    if profile.get('nodata') is not None:
        masked_data = np.ma.masked_equal(data, profile['nodata'])
    else:
        masked_data = data
        
    metrics = {
        'min': float(np.min(masked_data)),
        'max': float(np.max(masked_data)),
        'mean': float(np.mean(masked_data)),
        'std': float(np.std(masked_data)),
        'width': profile['width'],
        'height': profile['height'],
        'crs': str(profile['crs']),
        'count': profile['count']
    }
    
    return metrics
