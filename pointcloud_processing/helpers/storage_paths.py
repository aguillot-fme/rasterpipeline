from storage import StorageBackend


def find_pointcloud_by_id(
    storage: StorageBackend,
    file_id: str,
    base_dir: str,
    pattern: str,
) -> str:
    clean_id = (file_id or "").strip()
    if not clean_id:
        raise FileNotFoundError("Empty file_id")

    search_pattern = f"*{clean_id}*/{pattern}"
    files = storage.list_files(base_dir, pattern=search_pattern)
    candidates = sorted(files)
    if not candidates:
        raise FileNotFoundError(f"No pointcloud found for id={clean_id} under {base_dir}")
    return candidates[0]

