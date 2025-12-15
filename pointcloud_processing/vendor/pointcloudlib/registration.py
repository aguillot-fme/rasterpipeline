# pointcloudlib/registration.py

import numpy as np
import logging
from typing import Tuple, Optional

from .pointcloud import PointCloud
from .features import FeatureAggregation, GeometricFeatures
from .geometry import GeometryUtils, MathUtils

logger = logging.getLogger(__name__)

from scipy.spatial import cKDTree

class PointCloudRegistration:
    """
    A class for registering a source point cloud to a target point cloud,
    using feature matching and/or ICP methods.
    """

    def __init__(self, source_cloud: PointCloud, target_cloud: PointCloud):
        """
        Initialize with a source cloud and a target cloud.
        """
        self.source_cloud = source_cloud
        self.target_cloud = target_cloud
        logger.info("PointCloudRegistration initialized with source and target clouds.")

    def register_with_feature_matching(
    self,
    k_neighbors: int = 20,
    curvature_threshold: float = 0.1,
    feature_max_distance: float = 0.1,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, "PointCloud"]:
        """
        Registers the source point cloud to the target point cloud using feature matching and ICP.
        """
        required_keys = ["eig1", "eig2", "eig3", "nx", "ny", "nz","linearity", "planarity", "curvature"]

        # Ensure eigenvalues and normals are computed for source
        if not all(self.source_cloud.get_attribute(key) is not None for key in required_keys):
            GeometricFeatures.compute_eigenvalues_and_normals(self.source_cloud, k_neighbors=k_neighbors)
            GeometricFeatures.compute_additional_features(self.source_cloud)
            # Also compute derived features (linearity, planarity, curvature)
            GeometricFeatures.compute_eigenvalue_derived_features(self.source_cloud)
            logger.info("Computed normals, additional, and eigenvalue-derived features for source cloud.")
        else:
            logger.debug("Source cloud already has required basic attributes.")

        # Ensure eigenvalues and normals are computed for target
        if not all(self.target_cloud.get_attribute(key) is not None for key in required_keys):
            GeometricFeatures.compute_eigenvalues_and_normals(self.target_cloud, k_neighbors=k_neighbors)
            GeometricFeatures.compute_additional_features(self.target_cloud)
            GeometricFeatures.compute_eigenvalue_derived_features(self.target_cloud)
            logger.info("Computed normals, additional, and eigenvalue-derived features for target cloud.")
        else:
            logger.debug("Target cloud already has required basic attributes.")

        # Now that we have derived features, the required ones for matching are present:
        feature_names = ["linearity", "planarity", "curvature"]

        # Aggregate features from source and target
        source_feature_matrix = FeatureAggregation.aggregate_features(self.source_cloud, feature_names)
        target_feature_matrix = FeatureAggregation.aggregate_features(self.target_cloud, feature_names)

        # Detect keypoints based on curvature
        source_keypoints = GeometricFeatures.detect_keypoints_from_features(self.source_cloud, threshold=curvature_threshold, feature_name="curvature")
        target_keypoints = GeometricFeatures.detect_keypoints_from_features(self.target_cloud, threshold=curvature_threshold, feature_name="curvature")

        logger.info(f"Detected {len(source_keypoints)} source keypoints and {len(target_keypoints)} target keypoints.")

        # Subset features for keypoints
        source_feature_subset = source_feature_matrix[source_keypoints]
        target_feature_subset = target_feature_matrix[target_keypoints]

        source_points_subset = self.source_cloud.points[source_keypoints]
        target_points_subset = self.target_cloud.points[target_keypoints]

        # Match features
        src_indices, tgt_indices = self.match_features(source_feature_subset, target_feature_subset, max_distance=feature_max_distance)
        logger.info(f"Found {len(src_indices)} feature matches.")

        if len(src_indices) < 3:
            raise ValueError("Not enough matching features to compute initial transformation.")

        # Compute initial transformation using matched points
        matched_source_points = source_points_subset[src_indices]
        matched_target_points = target_points_subset[tgt_indices]
        init_transform = self.compute_best_fit_transform(matched_source_points, matched_target_points)
        logger.debug(f"Initial transformation matrix:\n{init_transform}")

        # Proceed with ICP using the initial transformation
        transform, transformed_source_cloud = self.register_icp(
            max_iterations=max_iterations,
            tolerance=tolerance,
            init_transform=init_transform,
        )

        return transform, transformed_source_cloud


    def register_icp(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        init_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, PointCloud]:
        """
        Registers the source point cloud to the target using ICP.
        
        Returns:
          transform (4x4) matrix, transformed source cloud (PointCloud).
        """
        source_points = self.source_cloud.points.copy()
        target_points = self.target_cloud.points

        # Apply initial transform if provided
        if init_transform is not None:
            source_points = self.apply_transformation(source_points, init_transform)
            logger.debug("Applied initial transform to source points.")

        # Build KDTree for target
        target_kdtree = self.target_cloud.kdtree
        transform = init_transform if init_transform is not None else np.eye(4)
        prev_error = float("inf")

        logger.info(f"Starting ICP with max_iterations={max_iterations}, tolerance={tolerance}")

        for i in range(max_iterations):
            distances, indices = target_kdtree.query(source_points)
            matched_tgt_points = target_points[indices]

            # Compute transformation
            T = self.compute_best_fit_transform(source_points, matched_tgt_points)
            source_points = self.apply_transformation(source_points, T)
            transform = T @ transform

            mean_error = np.mean(distances)
            logger.debug(f"ICP Iteration {i+1}: mean_error={mean_error}")

            if abs(prev_error - mean_error) < tolerance:
                logger.info(f"Convergence reached at iteration {i+1}.")
                break
            prev_error = mean_error

        # Create a new PointCloud with the final transformed points
        transformed_source_cloud = PointCloud(source_points, header=self.source_cloud.header.copy())

        # Optionally copy existing attributes from the original source
        for attr_name in ["eig1", "eig2", "eig3", "nx", "ny", "nz", "curvature"]:
            attr_val = self.source_cloud.get_attribute(attr_name)
            if attr_val is not None:
                transformed_source_cloud.set_attribute(attr_name, attr_val.copy())
                logger.debug(f"Copied '{attr_name}' to the transformed source cloud.")

        logger.info("ICP registration completed.")
        return transform, transformed_source_cloud

    @staticmethod
    def compute_best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute a 4x4 transform that maps A onto B using SVD.
        """
        assert A.shape == B.shape, "Source and target must have the same shape."
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def apply_transformation(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Apply a 4x4 transformation matrix to an (N,3) array of points.
        """
        ones = np.ones((points.shape[0], 1))
        hom_points = np.hstack((points, ones))
        transformed = hom_points @ T.T
        return transformed[:, :3]

    @staticmethod
    def match_features(
        source_features: np.ndarray, 
        target_features: np.ndarray, 
        max_distance: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match features between source_features and target_features (NxD each).
        """
        kdtree = cKDTree(target_features)
        distances, indices = kdtree.query(source_features, distance_upper_bound=max_distance)
        valid_mask = distances < max_distance
        source_indices = np.where(valid_mask)[0]
        target_indices = indices[valid_mask]
        return source_indices, target_indices
