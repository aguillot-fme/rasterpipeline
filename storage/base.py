from abc import ABC, abstractmethod
from typing import List, BinaryIO, Optional
import os

class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    """

    @abstractmethod
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory."""
        pass

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        """Read file content as bytes."""
        pass

    @abstractmethod
    def write_file(self, path: str, data: bytes) -> None:
        """Write bytes to a file."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """Delete a file."""
        pass
    
    @abstractmethod
    def get_local_path(self, path: str) -> str:
        """
        Get a local filesystem path for the file.
        If the storage is remote, this might involve downloading the file to a temporary location.
        For local storage, it returns the absolute path.
        """
        pass
