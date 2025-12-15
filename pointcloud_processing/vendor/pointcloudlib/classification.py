# pointcloudlib/classification.py

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from collections import Counter
from scipy.spatial import cKDTree

from .pointcloud import PointCloud
from .features import FeatureAggregation, GeometricFeatures
from .enums import ClassificationRule
from .preprocessing import PointCloudPreprocessing
from .geometry import GeometryUtils

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class Classification:
    @staticmethod
    def classify_noise(pc: PointCloud,
                       radius: float = 0.1,
                       min_neighbors: int = 5,
                       noise_class: int = 7) -> np.ndarray:
        """
        Classify points as noise based on the number of neighbors.
        Points having fewer than min_neighbors within the specified radius
        are marked with noise_class.
        """
        neighbor_indices = pc.compute_neighbors(radius=radius)
        neighbor_counts = np.array([len(nbrs) for nbrs in neighbor_indices])
        noise_mask = neighbor_counts < min_neighbors
        classification = pc.get_attribute("classification")
        if classification is None:
            classification = np.zeros(len(pc.points), dtype=np.uint8)
        classification[noise_mask] = noise_class
        pc.set_attribute("classification", classification)
        logger.info(f"Classified {np.sum(noise_mask)} points as noise.")
        return noise_mask

    @staticmethod
    def classify_outliers(pc: PointCloud,
                          method: str = "lof",
                          n_neighbors: int = 20,
                          contamination: float = 0.1,
                          outlier_class: int = 9,
                          store_scores: bool = False) -> np.ndarray:
        """
        Detect outliers in the point cloud using scikit‐learn outlier detection methods.
        If method "lof" (Local Outlier Factor) is chosen and store_scores is True,
        the LOF scores are stored in attribute "lof_score". Outliers are then marked
        with outlier_class.
        """
        if method.lower() == "lof":
            detector = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            labels = detector.fit_predict(pc.points)
            outlier_mask = (labels == -1)
            if store_scores:
                # More negative scores indicate stronger outlier–ness;
                # we store the positive values (i.e. -negative_outlier_factor)
                lof_scores = -detector.negative_outlier_factor_.astype(np.float32)
                pc.set_attribute("lof_score", lof_scores)
                logger.info("Stored LOF scores in attribute 'lof_score'.")
        elif method.lower() == "iforest":
            detector = IsolationForest(contamination=contamination, random_state=42)
            labels = detector.fit_predict(pc.points)
            outlier_mask = (labels == -1)
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        classification = pc.get_attribute("classification")
        if classification is None:
            classification = np.zeros(len(pc.points), dtype=np.uint8)
        classification[outlier_mask] = outlier_class
        pc.set_attribute("classification", classification)
        logger.info(f"Detected {np.sum(outlier_mask)} outliers using method '{method}'.")
        return outlier_mask

    @staticmethod
    def compute_based_on_neighbors(
        pc: PointCloud,
        reference_mask: Optional[np.ndarray] = None,
        target_mask: Optional[np.ndarray] = None,
        target_attribute_name: str = "classification",
        output_attribute_name: Optional[str] = None,
        radius: Optional[float] = None,
        k: Optional[int] = None,
        rule: ClassificationRule = ClassificationRule.MAJORITY,
    ) -> np.ndarray:
        """
        Computes a new attribute value for each point based on the values of its neighbors.
        Uses the existing KDTree in the point cloud if no reference mask is provided.

        Parameters:
            pc (PointCloud): The input point cloud.
            reference_mask (Optional[np.ndarray]): Boolean mask to select reference points.
            target_mask (Optional[np.ndarray]): Boolean mask to select target points (if None, all points are updated).
            target_attribute_name (str): The attribute to use for neighbor values.
            output_attribute_name (Optional[str]): If provided, results are stored here; otherwise, the target attribute is overwritten.
            radius (Optional[float]): Radius for neighbor search (mutually exclusive with k).
            k (Optional[int]): Number of nearest neighbors (mutually exclusive with radius).
            rule (ClassificationRule): Rule to use (e.g. majority, min, max, closest).

        Returns:
            np.ndarray: The computed attribute values.
        """
        if radius is None and k is None:
            raise ValueError("Either 'radius' or 'k' must be provided.")
        if radius is not None and k is not None:
            raise ValueError("'radius' and 'k' cannot both be provided.")

        target_values = pc.get_attribute(target_attribute_name)
        if target_values is None:
            raise ValueError(f"Target attribute '{target_attribute_name}' not found in the point cloud.")

        # Use existing KDTree if no reference mask is provided.
        if reference_mask is None:
            reference_points = pc.points
            reference_indices = np.arange(len(pc.points))
            tree = pc.kdtree
        else:
            if len(reference_mask) != len(pc.points):
                raise ValueError("reference_mask must have the same length as the point cloud.")
            reference_points = pc.points[reference_mask]
            reference_indices = np.arange(len(pc.points))[reference_mask]
            tree = cKDTree(reference_points)

        if target_mask is None:
            target_indices = np.arange(len(pc.points))
        else:
            if len(target_mask) != len(pc.points):
                raise ValueError("target_mask must have the same length as the point cloud.")
            target_indices = np.where(target_mask)[0]
        target_points = pc.points[target_indices]

        computed_values = target_values.copy()

        if radius is not None:
            neighbor_indices_list = tree.query_ball_point(target_points, r=radius, workers=-1)
        elif k is not None:
            distances, neighbor_indices = tree.query(target_points, k=k, workers=-1)
            neighbor_indices_list = neighbor_indices.tolist()

        for idx, neighbors in zip(target_indices, neighbor_indices_list):
            if not neighbors:
                continue
            mapped_neighbors = reference_indices[neighbors]
            neighbor_vals = target_values[mapped_neighbors]
            if rule == ClassificationRule.MAJORITY:
                most_common = Counter(neighbor_vals).most_common(1)
                computed_values[idx] = most_common[0][0] if most_common else target_values[idx]
            elif rule == ClassificationRule.MIN:
                computed_values[idx] = neighbor_vals.min()
            elif rule == ClassificationRule.MAX:
                computed_values[idx] = neighbor_vals.max()
            elif rule == ClassificationRule.CLOSEST:
                computed_values[idx] = neighbor_vals[0]
            else:
                raise ValueError(f"Unknown rule: {rule}")

        output_name = output_attribute_name or target_attribute_name
        pc.set_attribute(output_name, computed_values)
        logger.info(f"Updated attribute '{output_name}' using rule '{rule.value}'.")
        return computed_values

    @staticmethod
    def detect_keypoints_from_features(pc: PointCloud,
                                       feature_name: str = "curvature",
                                       threshold: float = 0.1) -> np.ndarray:
        """
        Detect keypoints based on a chosen feature (default is curvature).
        The function computes the geometric features if not already available.
        
        Parameters:
            pc (PointCloud): The input point cloud.
            feature_name (str): The name of the feature to use (e.g., "curvature", "planarity").
            threshold (float): The threshold value above which a point is considered a keypoint.
        
        Returns:
            np.ndarray: Indices of detected keypoints.
        """
        # Ensure the feature is computed; if not, compute geometric features.
        feat = pc.get_attribute(feature_name)
        if feat is None:
            # This call computes and stores a set of features, including curvature.
            GeometricFeatures.compute_geometric_features(pc)
            feat = pc.get_attribute(feature_name)
            if feat is None:
                raise ValueError(f"Feature '{feature_name}' could not be computed.")
        keypoint_indices = np.where(feat > threshold)[0]
        logger.info(f"Detected {len(keypoint_indices)} keypoints based on {feature_name} threshold {threshold}.")
        return keypoint_indices

    @staticmethod
    def classify_ruggedness(pc: PointCloud,
                            planar_threshold: float = 0.1,
                            rugged_threshold: float = 0.4,
                            normal_threshold: float = 15.0,
                            ground_offset: float = 2.0,
                            ground_class: int = 2,
                            building_class: int = 6,
                            vegetation_class: int = 3,
                            height_attribute: Optional[str] = None,
                            radius: float = 1.0) -> np.ndarray:
        """
        Classify points as buildings or vegetation based on geometric features.
        Requires precomputed normals and eigenvalues.
        
        Parameters:
            pc (PointCloud): The input point cloud.
            planar_threshold (float): Threshold on planarity for buildings.
            rugged_threshold (float): Threshold on ruggedness for vegetation.
            normal_threshold (float): Angular threshold (in degrees) for normal deviation.
            ground_offset (float): Minimum height above ground to be considered non-ground.
            ground_class (int): The class value for ground.
            building_class (int): The class value to assign for building-like points.
            vegetation_class (int): The class value to assign for vegetation-like (rugged) points.
            height_attribute (Optional[str]): If provided, use this attribute for height; otherwise, use "height".
            radius (float): Radius for neighborhood queries if needed.
        
        Returns:
            np.ndarray: Updated classification array.
        """
        classification = pc.get_attribute("classification")
        if classification is None:
            classification = np.zeros(len(pc.points), dtype=np.uint8)
        
        # Retrieve required geometric attributes
        eig1 = pc.get_attribute("eig1")
        eig2 = pc.get_attribute("eig2")
        eig3 = pc.get_attribute("eig3")
        nx = pc.get_attribute("nx")
        ny = pc.get_attribute("ny")
        nz = pc.get_attribute("nz")
        if any(attr is None for attr in [eig1, eig2, eig3, nx, ny, nz]):
            raise ValueError("Required geometric attributes are missing for ruggedness classification.")
        EPSILON = 1e-9
        # Compute planarity: here defined as (eig2 - eig1)/(eig3 + EPSILON)
        planarity = (eig2 - eig1) / (eig3 + EPSILON)

        # Retrieve or compute height
        if height_attribute:
            height = pc.get_attribute(height_attribute)
            if height is None:
                ground_indices = np.where(classification == ground_class)[0]
                height = GeometricFeatures.compute_height(pc, ground_indices=ground_indices, method="weighted")
                pc.set_attribute(height_attribute, height.astype(np.float32))
        else:
            height = pc.get_attribute("height")
            if height is None:
                # Use preprocessing to determine ground seeds if height is missing
                ground_indices, _ = PointCloudPreprocessing.find_lowest_points_by_cell(pc, cell_size=3.5)
                height = GeometricFeatures.compute_height(pc, ground_indices=ground_indices, method="weighted")
                pc.set_attribute("height", height.astype(np.float32))

        candidate_mask = (classification != ground_class) & (height > ground_offset)

        # Compute normal deviation: angle with vertical
        normal_deviation = pc.get_attribute("normal_deviation")
        if normal_deviation is None:
            normals = np.column_stack((nx, ny, nz))
            normal_deviation = GeometryUtils.angle_with_vertical(normals)
            pc.set_attribute("normal_deviation", normal_deviation.astype(np.float32))

        # Determine building vs. vegetation using planarity and normal deviation
        building_mask = candidate_mask & (planarity > planar_threshold) & (normal_deviation < np.deg2rad(normal_threshold))
        vegetation_mask = candidate_mask & ((planarity > rugged_threshold) | (normal_deviation > np.deg2rad(normal_threshold)))
        classification[building_mask] = building_class
        classification[vegetation_mask] = vegetation_class

        pc.set_attribute("classification", classification)
        logger.info(f"Ruggedness classification done. Buildings: {np.sum(building_mask)}, Vegetation: {np.sum(vegetation_mask)}.")
        return classification

    @staticmethod
    def classify_ground_axelsson(pc: PointCloud,
                                 max_slope: float = 0.1,
                                 grid_size: float = 1.0,
                                 elevation_threshold: float = 0.1,
                                 ground_class: int = 2,
                                 method: str = "weighted",
                                 **kwargs) -> np.ndarray:
        """
        Classify ground points using an Axelsson-inspired approach.
        Initial ground seeds are obtained from lowest points in grid cells, and then refined by height filtering.
        
        Parameters:
            pc (PointCloud): The input point cloud.
            max_slope (float): Maximum slope for initial seed selection (unused in this simplified version).
            grid_size (float): Grid cell size for initial seed selection.
            elevation_threshold (float): Height threshold for ground classification.
            ground_class (int): Class value for ground points.
            method (str): Method to compute height ("weighted" supported).
            **kwargs: Additional keyword arguments for height computation.
        
        Returns:
            np.ndarray: Updated classification array with ground points set.
        """
        seed_indices, _ = PointCloudPreprocessing.find_lowest_points_by_cell(pc, cell_size=grid_size)
        classification = pc.get_attribute("classification")
        if classification is None:
            classification = np.zeros(len(pc.points), dtype=np.uint8)
        # Temporarily mark seed indices with a special code (e.g. 23)
        classification[seed_indices] = 23
        height = GeometricFeatures.compute_height(pc, ground_indices=seed_indices, method=method, **kwargs)
        below_threshold = (height <= elevation_threshold)
        all_ground_indices = np.unique(np.concatenate([seed_indices, np.where(below_threshold)[0]]))
        classification[all_ground_indices] = ground_class
        pc.set_attribute("classification", classification)
        logger.info(f"Axelsson ground classification completed. {len(all_ground_indices)} ground points assigned class {ground_class}.")
        return classification

    @staticmethod
    def train_sklearn_model(pc: PointCloud,
                            feature_names: List[str],
                            labels: np.ndarray,
                            model: BaseEstimator) -> BaseEstimator:
        """
        Train a scikit-learn model using features aggregated from the point cloud.
        """
        features = FeatureAggregation.aggregate_features(pc, feature_names)
        model.fit(features, labels)
        logger.info(f"Trained sklearn model '{model.__class__.__name__}' with {len(labels)} samples.")
        return model

    @staticmethod
    def predict_with_model(pc: PointCloud,
                           feature_names: List[str],
                           model: BaseEstimator,
                           output_attribute_name: str = "predicted_class") -> np.ndarray:
        """
        Use a trained scikit-learn model to predict classes for the point cloud.
        """
        features = FeatureAggregation.aggregate_features(pc, feature_names)
        predicted_labels = model.predict(features)
        pc.set_attribute(output_attribute_name, predicted_labels)
        logger.info(f"Predicted classes using model '{model.__class__.__name__}' and stored in '{output_attribute_name}'.")
        return predicted_labels
