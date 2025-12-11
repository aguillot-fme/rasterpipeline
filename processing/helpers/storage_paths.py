from typing import Optional

from storage import StorageBackend


def find_raster_by_id(
    storage: StorageBackend,
    file_id: str,
    base_dir: str = "raw",
    pattern: str = "*.tif",
) -> str:
    """
    Locate the first raster file for a given file_id in storage.
    """
    clean_id = (file_id or "").strip()
    if not clean_id:
        raise FileNotFoundError(f"No raster found: empty file_id for base_dir={base_dir}")

    search_pattern = f"*{file_id}*/*"
    if pattern:
        search_pattern = f"*{clean_id}*/{pattern}"
    files = storage.list_files(base_dir, pattern=search_pattern)
    candidates = sorted(files)
    if not candidates:
        raise FileNotFoundError(f"No raster found for id={clean_id} under {base_dir}")
    return candidates[0]
