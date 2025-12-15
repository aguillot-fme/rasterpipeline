# pointcloudlib/features.py

import numpy as np
import logging
from typing import Optional, List, Dict

from .pointcloud import PointCloud
from .geometry import MathUtils, GeometryUtils

logger = logging.getLogger(__name__)

class GeometricFeatures:
    """
    Compute eigenvalues, normals, curvature, and other geometric features.
    """
    EPS = 1e-9

    @staticmethod
    def compute_eigenvalues_and_normals(pc: PointCloud, k_neighbors: int) -> None:
        required_keys = [
            "eig1", "eig2", "eig3", 
            "nx", "ny", "nz", 
            "eigvec1_x", "eigvec1_y", "eigvec1_z",
            "eigvec2_x", "eigvec2_y", "eigvec2_z",
            "eigvec3_x", "eigvec3_y", "eigvec3_z"
        ]
        if all(pc.get_attribute(k) is not None for k in required_keys):
            logger.info("Eigenvalues, normals, and eigenvectors already present. Skipping computation.")
            return

        indices = pc.compute_neighbors(k=k_neighbors)
        points = pc.points
        neighbor_points = points[indices]
        mean_neighbors = neighbor_points.mean(axis=1, keepdims=True)
        centered_neighbors = neighbor_points - mean_neighbors
        cov_matrices = MathUtils.compute_covariance(centered_neighbors)
        eig_vals, eig_vecs = MathUtils.eigen_decomposition(cov_matrices)

        eig1 = eig_vals[:, 0]
        eig2 = eig_vals[:, 1]
        eig3 = eig_vals[:, 2]

        pc.set_attribute("eig1", eig1.astype(np.float32))
        pc.set_attribute("eig2", eig2.astype(np.float32))
        pc.set_attribute("eig3", eig3.astype(np.float32))

        eigvec1 = eig_vecs[:, :, 0]
        eigvec2 = eig_vecs[:, :, 1]
        eigvec3 = eig_vecs[:, :, 2]

        pc.set_attribute("eigvec1_x", eigvec1[:, 0].astype(np.float32))
        pc.set_attribute("eigvec1_y", eigvec1[:, 1].astype(np.float32))
        pc.set_attribute("eigvec1_z", eigvec1[:, 2].astype(np.float32))
        pc.set_attribute("eigvec2_x", eigvec2[:, 0].astype(np.float32))
        pc.set_attribute("eigvec2_y", eigvec2[:, 1].astype(np.float32))
        pc.set_attribute("eigvec2_z", eigvec2[:, 2].astype(np.float32))
        pc.set_attribute("eigvec3_x", eigvec3[:, 0].astype(np.float32))
        pc.set_attribute("eigvec3_y", eigvec3[:, 1].astype(np.float32))
        pc.set_attribute("eigvec3_z", eigvec3[:, 2].astype(np.float32))

        normals = eigvec1
        normals_normalized = MathUtils.normalize_vectors(normals)
        pc.set_attribute("nx", normals_normalized[:, 0].astype(np.float32))
        pc.set_attribute("ny", normals_normalized[:, 1].astype(np.float32))
        pc.set_attribute("nz", normals_normalized[:, 2].astype(np.float32))

        logger.info("Eigenvalues, eigenvectors, and normals computed and stored.")

    @staticmethod
    def compute_eigenvalue_derived_features(pc: PointCloud) -> None:
        eig1 = pc.get_attribute("eig1")
        eig2 = pc.get_attribute("eig2")
        eig3 = pc.get_attribute("eig3")
        if eig1 is None or eig2 is None or eig3 is None:
            raise ValueError("Eigenvalues (eig1, eig2, eig3) must be computed before computing derived features.")
        EPS = GeometricFeatures.EPS
        curvature = eig1 / (eig1 + eig2 + eig3 + EPS)
        linearity = (eig3 - eig2) / (eig3 + EPS)
        planarity = (eig2 - eig1) / (eig3 + EPS)
        scattering = eig1 / (eig3 + EPS)
        omnivariance = (eig1 * eig2 * eig3) ** (1/3)
        anisotropy = (eig3 - eig1) / (eig3 + EPS)
        surface_variation = eig1 / (eig1 + eig2 + eig3 + EPS)
        eig_safe = np.stack((eig1, eig2, eig3), axis=1)
        eig_safe[eig_safe <= 0] = EPS
        eigen_entropy = -(eig_safe[:, 0]*np.log(eig_safe[:, 0] + EPS) +
                          eig_safe[:, 1]*np.log(eig_safe[:, 1] + EPS) +
                          eig_safe[:, 2]*np.log(eig_safe[:, 2] + EPS))
        curv_change = eig1 / (eig3 + EPS)
        pc.set_attribute("curvature", curvature.astype(np.float32))
        pc.set_attribute("linearity", linearity.astype(np.float32))
        pc.set_attribute("planarity", planarity.astype(np.float32))
        pc.set_attribute("scattering", scattering.astype(np.float32))
        pc.set_attribute("omnivariance", omnivariance.astype(np.float32))
        pc.set_attribute("anisotropy", anisotropy.astype(np.float32))
        pc.set_attribute("surface_variation", surface_variation.astype(np.float32))
        pc.set_attribute("eigen_entropy", eigen_entropy.astype(np.float32))
        pc.set_attribute("curv_change", curv_change.astype(np.float32))
        logger.info("Eigenvalue-based derived features computed and stored.")

    @staticmethod
    def compute_additional_features(pc: PointCloud) -> None:
        additional_feature_keys = ["plane_residuals", "dist_to_plane", "knn_max_h_diff", "knn_h_std", "normal_deviation", "deviation_variance"]
        if not all(pc.get_attribute(key) is not None for key in additional_feature_keys):
            logger.info("Additional per-point features not found. Computing them.")
            k_neighbors = 10
            indices = pc.compute_neighbors(k=k_neighbors)
            points = pc.points
            neighbor_points = points[indices]
            plane_coeffs, plane_residuals = GeometryUtils.fit_plane_least_squares(neighbor_points)
            a = plane_coeffs[:, 0]
            b = plane_coeffs[:, 1]
            c = plane_coeffs[:, 2]
            d = plane_coeffs[:, 3]
            denom = np.sqrt(a**2 + b**2 + c**2) + GeometricFeatures.EPS
            dist_to_plane = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / denom
            z_diff = neighbor_points[:, :, 2] - points[:, 2][:, np.newaxis]
            knn_max_h_diff = z_diff.max(axis=1)
            knn_h_std = z_diff.std(axis=1)
            nx = pc.get_attribute("nx")
            ny = pc.get_attribute("ny")
            nz = pc.get_attribute("nz")
            if nx is None or ny is None or nz is None:
                raise ValueError("Normals not computed before additional features.")
            normals = np.column_stack((nx, ny, nz))
            normal_deviation = GeometryUtils.angle_with_vertical(normals)
            deviation_variance = np.var(normal_deviation)
            pc.set_attribute("plane_residuals", plane_residuals.astype(np.float32))
            pc.set_attribute("dist_to_plane", dist_to_plane.astype(np.float32))
            pc.set_attribute("knn_max_h_diff", knn_max_h_diff.astype(np.float32))
            pc.set_attribute("knn_h_std", knn_h_std.astype(np.float32))
            pc.set_attribute("normal_deviation", normal_deviation.astype(np.float32))
            pc.set_attribute("deviation_variance", np.full(len(points), deviation_variance, dtype=np.float32))
            logger.info("Additional per-point features computed and stored.")
        else:
            logger.info("Additional per-point features already present.")

    @staticmethod
    def compute_height(pc: PointCloud, ground_indices: np.ndarray, method: str = "weighted", **kwargs) -> np.ndarray:
        logger.info(f"Computing height using the '{method}' method.")
        if method == "weighted":
            return GeometricFeatures.compute_height_from_ground_weighted(pc, ground_indices, **kwargs)
        elif method == "delaunay":
            return GeometricFeatures.compute_height_from_ground_delaunay(pc, ground_indices, **kwargs)
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'weighted' or 'delaunay'.")

    @staticmethod
    def compute_height_from_ground_weighted(pc: PointCloud, ground_indices: np.ndarray, k: int = 3) -> np.ndarray:
        logger.info("Starting height computation using weighted average elevation.")
        from scipy.spatial import cKDTree
        ground_points = pc.points[ground_indices]
        ground_tree = cKDTree(ground_points[:, :2])
        distances, idxs = ground_tree.query(pc.points[:, :2], k=k)
        if k == 1:
            distances = distances[:, np.newaxis]
            idxs = idxs[:, np.newaxis]
        neighbor_z_values = ground_points[idxs, 2]
        weights = 1.0 / (distances + GeometricFeatures.EPS)
        weights /= weights.sum(axis=1, keepdims=True)
        avg_elevation = (weights * neighbor_z_values).sum(axis=1)
        height = pc.points[:, 2] - avg_elevation
        pc.set_attribute("height", height.astype(np.float32))
        logger.info("Height above ground computed and stored.")
        return height

    @staticmethod
    def compute_height_from_ground_delaunay(pc: PointCloud, ground_indices: np.ndarray) -> np.ndarray:
        logger.info("Starting height computation using Delaunay triangulation.")
        from scipy.spatial import Delaunay, cKDTree
        ground_points = pc.points[ground_indices]
        if len(ground_points) < 3:
            raise ValueError("Not enough ground points to perform triangulation.")
        delaunay = Delaunay(ground_points[:, :2])
        simplices = delaunay.find_simplex(pc.points[:, :2])
        outside_hull = simplices == -1
        if np.any(outside_hull):
            logger.warning(f"{np.sum(outside_hull)} points are outside the convex hull.")
        triangle_vertices = delaunay.simplices[simplices[~outside_hull]]
        p1 = ground_points[triangle_vertices[:, 0]]
        p2 = ground_points[triangle_vertices[:, 1]]
        p3 = ground_points[triangle_vertices[:, 2]]
        v0 = p2[:, :2] - p1[:, :2]
        v1 = p3[:, :2] - p1[:, :2]
        v2 = pc.points[~outside_hull, :2] - p1[:, :2]
        d00 = np.einsum('ij,ij->i', v0, v0)
        d01 = np.einsum('ij,ij->i', v0, v1)
        d11 = np.einsum('ij,ij->i', v1, v1)
        d20 = np.einsum('ij,ij->i', v2, v0)
        d21 = np.einsum('ij,ij->i', v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        interpolated_z = np.full(len(pc.points), np.nan, dtype=np.float32)
        interpolated_z[~outside_hull] = (u * p1[:, 2] + v * p2[:, 2] + w * p3[:, 2]).astype(np.float32)
        if np.any(outside_hull):
            from scipy.spatial import cKDTree
            ground_tree = cKDTree(ground_points[:, :2])
            distances, indices = ground_tree.query(pc.points[outside_hull, :2])
            interpolated_z[outside_hull] = ground_points[indices, 2]
        height = pc.points[:, 2] - interpolated_z
        pc.set_attribute("height", height.astype(np.float32))
        logger.info("Height above ground computed using Delaunay triangulation and stored.")
        return height
    
    @staticmethod
    def compute_local_roughness(pc: PointCloud, k: int = 10) -> np.ndarray:
        """
        Compute the local roughness for each point in the point cloud.
        Roughness is defined as the standard deviation of the z-values among the k-nearest neighbors.
        
        Parameters:
            pc (PointCloud): The input point cloud.
            k (int): The number of nearest neighbors to consider.

        Returns:
            np.ndarray: A 1D array with the local roughness value for each point.
        """
        # Get the indices of k nearest neighbors for each point using the point cloud's KDTree.
        try:
            # If the point cloud was built with a KDTree, this uses it
            indices = pc.compute_neighbors(k=k)
        except Exception as e:
            logger.error(f"Error computing neighbors: {e}")
            raise
        
        roughness = np.zeros(len(pc.points), dtype=np.float32)
        for i in range(len(pc.points)):
            # For each point, extract the z-values of its neighbors.
            # If a point has less than 2 neighbors, we define roughness as 0.
            neighbor_indices = indices[i]
            if neighbor_indices.size < 2:
                roughness[i] = 0.0
            else:
                z_vals = pc.points[neighbor_indices, 2]
                roughness[i] = np.std(z_vals)
        
        # Optionally, set the computed roughness as an attribute for further use.
        pc.set_attribute("local_roughness", roughness)
        logger.info("Local roughness computed and stored as attribute 'local_roughness'.")
        return roughness

    @staticmethod
    def detect_keypoints_from_features(pc: PointCloud, curvature_threshold: float = 0.1) -> np.ndarray:
        curvature = pc.get_attribute("curvature")
        if curvature is None:
            eig1 = pc.get_attribute("eig1")
            eig2 = pc.get_attribute("eig2")
            eig3 = pc.get_attribute("eig3")
            if eig1 is None or eig2 is None or eig3 is None:
                raise ValueError("Eigenvalues not found. Cannot compute curvature.")
            curvature = eig1 / (eig1 + eig2 + eig3 + 1e-9)
            pc.set_attribute("curvature", curvature)
        keypoints = np.where(curvature > curvature_threshold)[0]
        logger.info(f"Detected {len(keypoints)} keypoints based on curvature threshold {curvature_threshold}.")
        return keypoints

    @staticmethod
    def fit_polynomial_surface_filter(pc: PointCloud, subset_indices: np.ndarray, max_deviation: float = 0.2, degree: int = 2, direction: str = "both") -> np.ndarray:
        if subset_indices.size == 0:
            logger.warning("No points provided for polynomial surface fitting.")
            return subset_indices
        points = pc.points[subset_indices]
        X = points[:, :2]
        y = points[:, 2]
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        polynomial_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        min_samples = {2: 10, 3: 20, 4: 30}.get(degree, 10)
        model = RANSACRegressor(estimator=polynomial_model, residual_threshold=max_deviation, min_samples=min_samples, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        if direction == "both":
            inlier_mask = np.abs(residuals) <= max_deviation
        elif direction == "above":
            inlier_mask = residuals <= max_deviation
        elif direction == "below":
            inlier_mask = residuals >= -max_deviation
        else:
            raise ValueError("Invalid direction. Use 'both', 'above', or 'below'.")
        valid_indices = subset_indices[inlier_mask]
        logger.info(f"Filtered subset down to {len(valid_indices)} points after polynomial surface fitting.")
        return valid_indices

class StatisticalFeatures:
    @staticmethod
    def compute_point_density_voxel(pc: PointCloud, voxel_size: float = 1.0) -> Dict[str, float]:
        points = pc.points
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        min_coords = points.min(axis=0)
        voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
        voxel_keys = voxel_indices[:, 0] + voxel_indices[:, 1]*100000 + voxel_indices[:, 2]*100000000
        unique_voxels, counts = np.unique(voxel_keys, return_counts=True)
        avg_density = counts.mean()
        min_density = counts.min()
        max_density = counts.max()
        deciles = np.percentile(counts, [10,20,30,40,50,60,70,80,90])
        density_stats = {
            "average_density": avg_density,
            "min_density": min_density,
            "max_density": max_density,
            "decile_10": deciles[0],
            "decile_20": deciles[1],
            "decile_30": deciles[2],
            "decile_40": deciles[3],
            "decile_50": deciles[4],
            "decile_60": deciles[5],
            "decile_70": deciles[6],
            "decile_80": deciles[7],
            "decile_90": deciles[8]
        }
        logger.info("Point density statistics computed.")
        return density_stats

    @staticmethod
    def select_keypoints_by_attribute(pc: PointCloud, attribute_name: str, threshold: float, mode: str="greater") -> np.ndarray:
        attr = pc.get_attribute(attribute_name)
        if attr is None:
            raise ValueError(f"Attribute '{attribute_name}' not found.")
        if mode == "greater":
            mask = attr > threshold
        elif mode == "less":
            mask = attr < threshold
        elif mode == "greater_equal":
            mask = attr >= threshold
        elif mode == "less_equal":
            mask = attr <= threshold
        else:
            raise ValueError("Invalid mode.")
        keypoints = np.where(mask)[0]
        logger.info(f"Selected {len(keypoints)} keypoints based on attribute '{attribute_name}' with threshold {threshold} and mode '{mode}'.")
        return keypoints

class FeatureAggregation:
    @staticmethod
    def propagate_features_from_keypoints(pc: PointCloud, keypoint_indices: np.ndarray, feature_attributes: List[str], radius: float = 1.0, default_val: float = -999.0) -> None:
        if len(keypoint_indices) == 0:
            logger.warning("No keypoints given; propagation skipped.")
            return
        points = pc.points
        from scipy.spatial import cKDTree
        key_tree = cKDTree(points[keypoint_indices])
        distances, indices = key_tree.query(points, distance_upper_bound=radius)
        no_keypoint_mask = (distances == np.inf)
        valid_mask = ~no_keypoint_mask
        for attr in feature_attributes:
            key_attr_values = pc.get_attribute(attr)
            if key_attr_values is None:
                raise ValueError(f"Attribute '{attr}' not found in the point cloud.")
            out_arr = pc.get_attribute(attr)
            if out_arr is None:
                out_arr = np.full(len(points), default_val, dtype=key_attr_values.dtype)
            else:
                out_arr = out_arr.copy()
            out_arr[valid_mask] = key_attr_values[keypoint_indices[indices[valid_mask]]]
            out_arr[no_keypoint_mask] = default_val
            pc.set_attribute(attr, out_arr)
        logger.info("Features propagated from keypoints.")

    @staticmethod
    def aggregate_features(pc: PointCloud, feature_names: List[str]) -> np.ndarray:
        feature_arrays = []
        for fname in feature_names:
            attr_data = pc.get_attribute(fname)
            if attr_data is None:
                raise ValueError(f"Required feature '{fname}' not found in the point cloud.")
            feature_arrays.append(attr_data.reshape(-1, 1))
        feature_matrix = np.hstack(feature_arrays)
        logger.debug(f"Aggregated features into a matrix with shape {feature_matrix.shape}.")
        return feature_matrix
