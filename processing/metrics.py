import argparse
import json
from typing import Dict, Any

import numpy as np

from storage import StorageBackend, get_storage_backend
from .helpers.raster_utils import read_raster_data
from .helpers import find_raster_by_id

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


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for raw and processed rasters.")
    parser.add_argument("--file-path", help="Path to the raw raster inside storage.")
    parser.add_argument("--processed-path", help="Path to the processed raster (e.g., NDVI) inside storage.")
    parser.add_argument("--file-id", help="Ingestion file id to resolve raster paths automatically.")
    args = parser.parse_args()

    if not args.file_path and not args.file_id:
        raise SystemExit("Provide either --file-id or --file-path.")

    storage = get_storage_backend()
    raw_path = args.file_path or find_raster_by_id(storage, args.file_id, base_dir="raw", pattern="*.tif")
    processed_path = args.processed_path or find_raster_by_id(storage, args.file_id, base_dir="processed", pattern="*_ndvi.tif")

    metrics = {
        "raw": compute_raster_metrics(storage, raw_path),
        "processed": compute_raster_metrics(storage, processed_path),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
