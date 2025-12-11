import os
import glob
import shutil
from typing import List, Optional
from .base import StorageBackend

class LocalStorage(StorageBackend):
    """
    Storage backend implementation for the local filesystem.
    """

    def __init__(self, base_path: str = ""):
        self.base_path = base_path

    def _full_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        full_path = self._full_path(path)
        if not os.path.exists(full_path):
            return []
        
        if pattern:
            search_pattern = os.path.join(full_path, pattern)
            files = glob.glob(search_pattern)
            return [f for f in files if os.path.isfile(f)]
        else:
            return [os.path.join(full_path, f) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]

    def read_file(self, path: str) -> bytes:
        full_path = self._full_path(path)
        with open(full_path, 'rb') as f:
            return f.read()

    def write_file(self, path: str, data: bytes) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        full_path = self._full_path(path)
        return os.path.exists(full_path)

    def delete_file(self, path: str) -> None:
        full_path = self._full_path(path)
        if os.path.exists(full_path):
            os.remove(full_path)

    def get_local_path(self, path: str) -> str:
        return self._full_path(path)
