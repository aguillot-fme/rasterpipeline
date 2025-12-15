# pointcloudlib/preprocessing.py

import numpy as np
from typing import Optional, Tuple, Dict
import logging

from .pointcloud import PointCloud
from .features import GeometricFeatures

logger = logging.getLogger(__name__)

class PointCloudPreprocessing:
    @staticmethod
    def compute_neighbors(pc: PointCloud, radius: Optional[float] = None, k: Optional[int] = None) -> np.ndarray:
        return pc.compute_neighbors(radius=radius, k=k)

    @staticmethod
    def voxel_down_sample(pc: PointCloud, voxel_size: float) -> np.ndarray:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        points = pc.points
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        dtype = [('vx', np.int32), ('vy', np.int32), ('vz', np.int32), ('index', np.int32)]
        data = np.empty(len(points), dtype=dtype)
        data['vx'] = voxel_coords[:, 0]
        data['vy'] = voxel_coords[:, 1]
        data['vz'] = voxel_coords[:, 2]
        data['index'] = np.arange(len(points), dtype=np.int32)
        data.sort(order=["vx", "vy", "vz", "index"])
        _, unique_indices = np.unique(data[['vx', 'vy', 'vz']], return_index=True)
        selected_indices = data['index'][unique_indices]
        selected_indices.sort()
        logger.info(f"Voxel downsampled to {len(selected_indices)} points from {len(points)}.")
        return selected_indices

    @staticmethod
    def filter_by_height(pc: PointCloud, ground_indices: np.ndarray, ground_offset: float) -> np.ndarray:
        height = pc.get_attribute("height")
        if height is None:
            height = GeometricFeatures.compute_height(pc, ground_indices=ground_indices, method="weighted")
        mask = height > ground_offset
        logger.info(f"Filtered {mask.sum()} points above ground offset {ground_offset}.")
        return mask

    @staticmethod
    def find_lowest_points_by_cell(pc: PointCloud, cell_size: float = 3.5) -> Tuple[np.ndarray, Dict]:
        points = pc.points
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        grid_x = np.floor((points[:, 0] - x_min) / cell_size).astype(int)
        grid_y = np.floor((points[:, 1] - y_min) / cell_size).astype(int)
        dtype = [("gx", int), ("gy", int), ("z", float), ("index", int)]
        data = np.empty(len(points), dtype=dtype)
        data["gx"] = grid_x
        data["gy"] = grid_y
        data["z"] = points[:, 2]
        data["index"] = np.arange(len(points))
        data.sort(order=["gx", "gy", "z"])
        _, unique_indices = np.unique(data[["gx", "gy"]], return_index=True)
        lowest_indices = data["index"][unique_indices]
        voxels, counts = np.unique(data[["gx", "gy"]], return_counts=True, axis=0)
        voxel_counts = {tuple(v): c for v, c in zip(voxels, counts)}
        logger.info(f"Found {len(lowest_indices)} lowest points across {len(voxel_counts)} cells.")
        return lowest_indices, voxel_counts

    @staticmethod
    def filter_lowest_points_by_voxel(pc: PointCloud, ground_indices: np.ndarray, voxel_size: float = 1.0) -> np.ndarray:
        ground_points = pc.points[ground_indices]
        voxel_coords = np.floor(ground_points[:, :2] / voxel_size).astype(int)
        dtype = [("vx", int), ("vy", int), ("z", float), ("index", int)]
        data = np.empty(len(ground_points), dtype=dtype)
        data["vx"] = voxel_coords[:, 0]
        data["vy"] = voxel_coords[:, 1]
        data["z"] = ground_points[:, 2]
        data["index"] = ground_indices
        data.sort(order=["vx", "vy", "z"])
        voxels = np.stack((data["vx"], data["vy"]), axis=-1)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        lowest_ground_indices = data["index"][unique_indices]
        logger.info(f"Filtered to {len(lowest_ground_indices)} lowest ground points by voxel.")
        return lowest_ground_indices
