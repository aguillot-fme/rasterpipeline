import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys

torch = pytest.importorskip("torch")

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.compute_embeddings import compute_embeddings

def _has_parquet_writer() -> bool:
    import importlib.util

    return any(
        importlib.util.find_spec(mod) is not None
        for mod in ("pyarrow", "fastparquet", "duckdb")
    )


if not _has_parquet_writer():
    pytest.skip(
        "No parquet engine available (need one of: pyarrow, fastparquet, duckdb)",
        allow_module_level=True,
    )


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        import duckdb

        con = duckdb.connect()
        con.register("df", df)
        safe_path = path.replace("'", "''")
        con.execute(f"COPY df TO '{safe_path}' (FORMAT PARQUET)")
        con.close()


@pytest.fixture
def mock_tiles_dir():
    tmp_dir = tempfile.mkdtemp()
    
    # Create fake tiles
    df_data = []
    for i in range(2):
        tile_name = f"tile_{i}.tif"
        # Touch file
        with open(os.path.join(tmp_dir, tile_name), 'wb') as f:
            f.write(b"fake data")
        
        df_data.append({
            'dataset_id': 'test',
            'tile_id': f"tile_{i}",
            'tile_path': tile_name
        })
    
    # Create index
    df = pd.DataFrame(df_data)
    _write_parquet(df, os.path.join(tmp_dir, "index.parquet"))
    
    yield tmp_dir
    # cleanup handled by test runner or OS eventually, or explicitly if needed

@patch("scripts.compute_embeddings.AutoModel")
@patch("scripts.compute_embeddings.AutoImageProcessor")
@patch("scripts.compute_embeddings.rasterio")
def test_compute_embeddings_local(mock_rasterio, mock_processor, mock_model, mock_tiles_dir):
    """Test embedding generation loop locally."""
    output_dir = tempfile.mkdtemp()

    # Mock Rasterio
    mock_src = MagicMock()
    # Return random image array (3, 224, 224)
    mock_src.read.return_value = np.zeros((3, 224, 224), dtype=np.uint8)
    mock_src.width = 224
    mock_src.height = 224
    mock_src.transform = MagicMock()
    mock_rasterio.open.return_value.__enter__.return_value = mock_src
    mock_rasterio.windows.Window.side_effect = lambda col, row, w, h: (col, row, w, h)
    mock_rasterio.windows.bounds.side_effect = lambda window, _transform: (
        float(window[0]),
        float(window[1]),
        float(window[0] + window[2]),
        float(window[1] + window[3]),
    )

    # Mock Model Output
    mock_model_instance = MagicMock()
    # last_hidden_state shape: (batch, seq, hidden) -> (1, 197, 384) e.g.
    mock_out = MagicMock()
    mock_out.last_hidden_state = torch.zeros((1, 10, 384))
    mock_model_instance.return_value = mock_out
    mock_model.from_pretrained.return_value.to.return_value = mock_model_instance
    
    # Mock Processor
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    # Behave like a dict for **inputs unpacking if the code uses it, 
    # but the code does inputs = ... .to(device), then model(**inputs).
    # If model(**inputs) is called, inputs must be a dict-like or unpacking works.
    # Actually, model(**inputs) means inputs is a dict.
    # So .to() should return a dict.
    mock_inputs.to.return_value = {'pixel_values': torch.zeros((1, 3, 224, 224))} 
    mock_processor.from_pretrained.return_value.return_value = mock_inputs

    # Run
    compute_embeddings(mock_tiles_dir, output_dir)
    
    # Check output
    out_file = os.path.join(output_dir, "embeddings.parquet")
    assert os.path.exists(out_file)
    
    df_out = pd.read_parquet(out_file)
    assert len(df_out) == 18
    assert "embedding" in df_out.columns
    assert "patch_row" in df_out.columns
    assert "patch_col" in df_out.columns
    assert "patch_min_x" in df_out.columns
    # Check embedding is list/array
    assert isinstance(df_out.iloc[0]['embedding'], np.ndarray) or isinstance(df_out.iloc[0]['embedding'], list)

@patch("fsspec.filesystem")
@patch("pandas.read_parquet")
def test_compute_embeddings_s3_index(mock_pd_read, mock_fs, mock_tiles_dir):
    """Test that S3 index reading uses storage_options."""
    # We abort early to just check read_parquet call
    mock_pd_read.side_effect = Exception("Stop here") 
    
    fs_args = '{"AWS_ENDPOINT_URL": "http://minio", "STORAGE_TYPE": "s3"}'
    try:
        compute_embeddings("s3://bucket/tiles", "/tmp/out", fs_args)
    except Exception:
        pass
    
    mock_pd_read.assert_called()
    call_kwargs = mock_pd_read.call_args[1]
    assert call_kwargs["storage_options"]["client_kwargs"]["endpoint_url"] == "http://minio"
    assert "STORAGE_TYPE" not in call_kwargs["storage_options"]
