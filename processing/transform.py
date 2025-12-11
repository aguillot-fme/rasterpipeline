from storage import StorageBackend
from .helpers.raster_utils import read_raster_data
import numpy as np
import rasterio
from rasterio.io import MemoryFile


def _select_band(data: np.ndarray, index: int) -> np.ndarray:
    """
    Return a 2D array for the requested band index (1-based).
    If the data is already single-band, return it as-is.
    """
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if index < 1 or index > data.shape[0]:
            raise ValueError(f"Band index {index} out of range for raster with {data.shape[0]} bands")
        return data[index - 1]
    raise ValueError(f"Unexpected raster dimensions: {data.shape}")


def calculate_ndvi(
    storage: StorageBackend,
    red_band_path: str,
    nir_band_path: str,
    output_path: str,
    red_band_index: int = 1,
    nir_band_index: int = 1,
) -> str:
    """
    Calculate NDVI from Red and NIR bands.
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        storage: Storage backend.
        red_band_path: Path to Red band raster.
        nir_band_path: Path to NIR band raster.
        output_path: Path to save the NDVI raster.
        red_band_index: 1-based band index for the red band (if multi-band input).
        nir_band_index: 1-based band index for the NIR band (if multi-band input).
        
    Returns:
        Path to the output file.
    """
    # Read Red band
    red_data_raw, red_profile = read_raster_data(storage.read_file(red_band_path))
    red_data = _select_band(red_data_raw, red_band_index)
    
    # Read NIR band
    nir_data_raw, nir_profile = read_raster_data(storage.read_file(nir_band_path))
    nir_data = _select_band(nir_data_raw, nir_band_index)
    
    # Ensure dimensions match
    if red_data.shape != nir_data.shape:
        raise ValueError("Red and NIR bands must have the same dimensions")
        
    # Calculate NDVI
    # Allow division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_data.astype(float) - red_data.astype(float)) / (nir_data + red_data)
        
    # Handle NaNs
    ndvi[np.isnan(ndvi)] = 0
    
    # Update profile for float output
    profile = red_profile.copy()
    profile.update(dtype=rasterio.float32, count=1, driver='GTiff')
    
    # Write output
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(ndvi.astype(rasterio.float32), 1)
        
        storage.write_file(output_path, memfile.read())
        
    return output_path
