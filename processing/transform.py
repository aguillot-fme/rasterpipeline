import argparse

from storage import StorageBackend, get_storage_backend
from .helpers.raster_utils import read_raster_data
from .helpers import find_raster_by_id
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


def _derive_output_path(input_path: str, file_id: str | None = None) -> str:
    candidate = input_path
    if "/raw/" in input_path:
        candidate = input_path.replace("/raw/", "/processed/", 1)
    elif "raw" in input_path:
        candidate = input_path.replace("raw", "processed", 1)

    if candidate.endswith(".tif"):
        candidate = candidate[:-4] + "_ndvi.tif"
    elif file_id:
        candidate = f"processed/{file_id}/ndvi.tif"
    else:
        candidate = f"{candidate}_ndvi"

    return candidate


def main():
    parser = argparse.ArgumentParser(description="Compute NDVI for an ingested raster.")
    parser.add_argument("--file-id", help="Ingestion file id to locate rasters in storage.")
    parser.add_argument("--red-band-path", help="Explicit path to the red band raster.")
    parser.add_argument("--nir-band-path", help="Explicit path to the NIR band raster (defaults to red).")
    parser.add_argument("--output-path", help="Destination path; defaults to processed/<file_id>/_ndvi.tif")
    parser.add_argument("--red-band-index", type=int, default=1)
    parser.add_argument("--nir-band-index", type=int, default=1)
    args = parser.parse_args()

    storage = get_storage_backend()
    red_path = args.red_band_path or find_raster_by_id(storage, args.file_id, base_dir="raw", pattern="*.tif")
    nir_path = args.nir_band_path or red_path
    output = args.output_path or _derive_output_path(red_path, args.file_id)

    result_path = calculate_ndvi(
        storage,
        red_path,
        nir_path,
        output,
        red_band_index=args.red_band_index,
        nir_band_index=args.nir_band_index,
    )
    print(result_path)


if __name__ == "__main__":
    main()
