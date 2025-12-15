# pointcloudlib/pointcloud.py

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PointCloud:
    """
    A class to represent a 3D point cloud.
    """

    def __init__(self, points: np.ndarray, header: Optional[dict] = None):
        """
        Initialize the PointCloud object.

            points (np.ndarray): An (N, 3) array of XYZ coordinates.
            header (Optional[dict]): Optional header information.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points array must be of shape (N, 3).")

        self.points = points  # Store XYZ as NumPy array
        self._attributes: Dict[str, np.ndarray] = {}  # Store additional dimensions dynamically
        self.header: dict = header or {"extra_dimensions": []}
        self._kdtree = None  # Lazy initialization

        logger.info(f"PointCloud initialized with points shape: {self.points.shape}")

    def sync_from_las(self, las):
        """
        Sync attributes and header from a Laspy LAS object.

        Parameters:
            las: A Laspy LAS object.
        """
        self.points = np.vstack((las.x, las.y, las.z)).T
        logger.info(f"PointCloud points synchronized from LAS file with shape: {self.points.shape}")

        for dim in las.point_format.dimensions:
            if dim.name not in {"X", "Y", "Z"}:
                data = getattr(las, dim.name).astype(dim.dtype.name)
                self.set_attribute(dim.name, data)

        self.header = {
            "scale": las.header.scales,
            "offset": las.header.offsets,
            "point_format_id": las.header.point_format.id,
            "version": las.header.version,
            "extra_dimensions": [
                {"name": dim.name, "dtype": dim.dtype.name}
                for dim in las.point_format.dimensions
                if dim.name not in {"X", "Y", "Z"}
            ],
        }
        logger.info("Header synchronized from LAS file.")

    def set_attribute(self, name: str, value: np.ndarray):
        """
        Set an attribute and update the header accordingly.

        Parameters:
            name (str): Name of the attribute.
            value (np.ndarray): Data array for the attribute.
        """
        if not isinstance(name, str):
            raise TypeError("Attribute name must be a string.")
        if not isinstance(value, np.ndarray):
            raise TypeError("Attribute value must be a numpy.ndarray.")

        self._attributes[name] = value
        self.update_header_dimension(name, str(value.dtype))
        logger.info(f"Attribute '{name}' set with dtype '{value.dtype}'.")

    def get_attribute(self, name: str) -> Optional[np.ndarray]:
        """
        Get an attribute by name.

        Parameters:
            name (str): Name of the attribute.

        Returns:
            Optional[np.ndarray]: The attribute data if it exists, else None.
        """
        return self._attributes.get(name)

    def remove_attribute(self, name: str):
        """
        Remove an attribute and its corresponding header entry.

        Parameters:
            name (str): Name of the attribute to remove.
        """
        if name in self._attributes:
            del self._attributes[name]
            self.header["extra_dimensions"] = [
                dim for dim in self.header["extra_dimensions"] if dim["name"] != name
            ]
            logger.info(f"Attribute '{name}' and its header entry removed.")
        else:
            logger.warning(f"Attempted to remove non-existent attribute '{name}'.")

    def list_attributes(self) -> List[str]:
        """
        List all attribute names.

        Returns:
            List[str]: List of attribute names.
        """
        return list(self._attributes.keys())

    @property
    def kdtree(self):
        """
        Lazy initialization of the KDTree.

        Returns:
            cKDTree: The KDTree object for the point cloud.
        """
        if self._kdtree is None:
            self._kdtree = cKDTree(self.points)
            logger.info("KDTree initialized.")
        return self._kdtree

    def update_header_dimension(self, name: str, dtype: str):
        """
        Add or update an extra dimension in the header.

        Parameters:
            name (str): Name of the dimension (e.g., 'nx', 'eig1').
            dtype (str): Data type of the dimension (e.g., 'float32').
        """
        for dim in self.header["extra_dimensions"]:
            if dim["name"] == name:
                dim["dtype"] = dtype
                logger.debug(f"Header dimension '{name}' updated to dtype '{dtype}'.")
                return
        self.header["extra_dimensions"].append({"name": name, "dtype": dtype})
        logger.debug(f"Header dimension '{name}' added with dtype '{dtype}'.")

    def compute_neighbors(self, radius: Optional[float] = None, k: Optional[int] = None) -> np.ndarray:
        """
        Compute neighbors for each point in the point cloud.

        Parameters:
            radius (Optional[float]): Radius to use for neighbor search.
            k (Optional[int]): Number of nearest neighbors to find.

        Returns:
            np.ndarray: Indices of neighboring points.

        Raises:
            ValueError: If neither 'radius' nor 'k' is provided.
        """
        if radius is not None:
            neighbors = self.kdtree.query_ball_point(self.points, r=radius, workers=-1)
            return neighbors
        elif k is not None:
            distances, indices = self.kdtree.query(self.points, k=k, workers=-1)
            return indices
        else:
            raise ValueError("Either 'radius' or 'k' must be provided to compute neighbors.")
