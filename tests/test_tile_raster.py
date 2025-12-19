import os
import shutil
import tempfile
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
import pandas as pd
from unittest.mock import MagicMock, patch
import sys

# Add scripts to path to import tile_raster
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.tile_raster import tile_raster

@pytest.fixture
def mock_raster_file():
    """Create a temporary dummy raster file."""
    tmp_dir = tempfile.mkdtemp()
    filename = os.path.join(tmp_dir, "test_raster.tif")
    
    # Create a 512x512 raster
    data = np.zeros((3, 512, 512), dtype=np.uint8)
    transform = from_origin(0, 512, 1, 1)
    
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=512, width=512,
        count=3, dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data)
        
    yield filename
    shutil.rmtree(tmp_dir)

@patch("fsspec.filesystem")
def test_tile_raster_local(mock_fs, mock_raster_file):
    """Test tiling logic on a local file."""
    output_dir = tempfile.mkdtemp()
    
    try:
        # Tile into 256x256 -> Should produce 4 tiles (2x2)
        tile_raster(mock_raster_file, output_dir, tile_size=256)
        
        # Check files
        files = os.listdir(output_dir)
        tifs = [f for f in files if f.endswith('.tif')]
        assert len(tifs) == 4
        
        # Check Index
        assert 'index.parquet' in files
        df = pd.read_parquet(os.path.join(output_dir, 'index.parquet'))
        assert len(df) == 4
        assert 'dataset_id' in df.columns
        assert 'tile_id' in df.columns
        
    finally:
        shutil.rmtree(output_dir)

@patch("fsspec.filesystem")
def test_tile_raster_s3_upload(mock_fs_cls, mock_raster_file):
    """Test that S3 upload logic is triggered."""
    output_dir = "s3://bucket/test_out"
    mock_fs_instance = MagicMock()
    mock_fs_cls.return_value = mock_fs_instance
    
    # Run
    tile_raster(mock_raster_file, output_dir, tile_size=256)
    
    # Assert put was called
    # tile_raster downloads local input (already local here)
    # then tiles to temp
    # then puts temp dir to s3
    assert mock_fs_instance.put.called
    args, _ = mock_fs_instance.put.call_args
    assert args[1] == output_dir # Destination
