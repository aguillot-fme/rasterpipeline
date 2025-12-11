import rasterio
from rasterio.io import MemoryFile
import numpy as np
from typing import Tuple, Dict, Any

def read_raster_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """
    Read metadata from a raster file (bytes).
    """
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as dataset:
            return dataset.profile

def read_raster_data(file_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read raster data and profile from bytes.
    """
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as dataset:
            data = dataset.read()
            profile = dataset.profile
            return data, profile

def get_raster_bounds(file_bytes: bytes) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of the raster.
    Returns (left, bottom, right, top).
    """
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as dataset:
            return dataset.bounds
