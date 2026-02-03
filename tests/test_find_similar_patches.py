import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Add project root to path to import scripts.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.find_similar_patches import find_similar_patches


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        import duckdb

        con = duckdb.connect()
        con.register("df", df)
        safe_path = path.replace("'", "''")
        con.execute(f"COPY df TO '{safe_path}' (FORMAT PARQUET)")
        con.close()


def test_find_similar_patches_topk(tmp_path, monkeypatch):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(parents=True)

    emb_df = pd.DataFrame(
        {
            "tile_id": ["tile_a", "tile_b"],
            "tile_path": ["tile_a.tif", "tile_b.tif"],
            "patch_id": ["tile_a_r0_c0", "tile_b_r0_c0"],
            "patch_min_x": [0.0, 100.0],
            "patch_min_y": [0.0, 0.0],
            "patch_max_x": [50.0, 150.0],
            "patch_max_y": [50.0, 50.0],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )
    _write_parquet(emb_df, embeddings_dir / "embeddings.parquet")

    def fake_load_patch_image(path: str, _storage_options: dict):
        return SimpleNamespace(path=path)

    def fake_embedder(image):
        return np.array([1.0, 0.0]) if "q1" in image.path else np.array([0.0, 1.0])

    monkeypatch.setattr("scripts.find_similar_patches._load_patch_image", fake_load_patch_image)

    out_path = tmp_path / "results.parquet"
    csv_path = tmp_path / "results.csv"
    results = find_similar_patches(
        embeddings_dir=str(embeddings_dir),
        query_paths=["q1.tif", "q2.tif"],
        output_path=str(out_path),
        fs_args_str="{}",
        top_k=1,
        embedder=fake_embedder,
    )

    assert os.path.exists(out_path)
    assert len(results) == 2
    assert results.iloc[0]["match_patch_id"] == "tile_a_r0_c0"
    assert results.iloc[1]["match_patch_id"] == "tile_b_r0_c0"
    assert os.path.exists(csv_path)


def test_find_similar_patches_query_coords(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(parents=True)

    emb_df = pd.DataFrame(
        {
            "tile_id": ["tile_a", "tile_b"],
            "tile_path": ["tile_a.tif", "tile_b.tif"],
            "patch_id": ["tile_a_r0_c0", "tile_b_r0_c0"],
            "patch_min_x": [0.0, 100.0],
            "patch_min_y": [0.0, 0.0],
            "patch_max_x": [100.0, 200.0],
            "patch_max_y": [100.0, 100.0],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )
    _write_parquet(emb_df, embeddings_dir / "embeddings.parquet")

    out_path = tmp_path / "coords_results.parquet"
    results = find_similar_patches(
        embeddings_dir=str(embeddings_dir),
        query_paths=[],
        output_path=str(out_path),
        fs_args_str="{}",
        top_k=1,
        query_coords=[{"x": 50.0, "y": 50.0}],
    )

    assert len(results) == 1
    assert results.iloc[0]["match_patch_id"] == "tile_a_r0_c0"
