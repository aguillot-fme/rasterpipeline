# pointcloudlib/clustering.py

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from collections import Counter
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    KMeans,
    AgglomerativeClustering,
    MeanShift,
    SpectralClustering,
    OPTICS,
)
from sklearn.neighbors import NearestNeighbors

from .pointcloud import PointCloud
from .geometry import GeometryUtils
logger = logging.getLogger(__name__)

class Clustering:
    @staticmethod
    def hierarchical_clustering(pc: PointCloud, distance_threshold: float = 1.5) -> Tuple[np.ndarray, Dict[int, int]]:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        labels = clustering.fit_predict(pc.points)
        cluster_info = Clustering._cluster_info(labels)
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        logger.info(f"Hierarchical clustering completed with {len(cluster_info)} clusters.")
        return labels, cluster_info

    @staticmethod
    def kmeans_clustering(pc: PointCloud, k: int = 5) -> Tuple[np.ndarray, Dict[int, int]]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pc.points)
        cluster_info = Clustering._cluster_info(labels)
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        logger.info(f"KMeans clustering completed with {k} clusters.")
        return labels, cluster_info

    @staticmethod
    def estimate_eps(pc: PointCloud, k: int = 4) -> float:
        nbrs = NearestNeighbors(n_neighbors=k).fit(pc.points)
        distances, indices = nbrs.kneighbors(pc.points)
        kth_distances = distances[:, k-1]
        kth_distances.sort()
        eps = np.percentile(kth_distances, 90)
        logger.info(f"Estimated eps for DBSCAN: {eps}")
        return eps

    @staticmethod
    def dbscan_clustering(pc: PointCloud, eps: Optional[float] = None, min_samples: int = 10, metric: str = "euclidean", algorithm: str = "auto", leaf_size: int = 30, p: Optional[int] = None, n_jobs: Optional[int] = -1, auto_eps: bool = False, min_cluster_size: Optional[int] = 1000) -> Tuple[np.ndarray, Dict[int, int]]:
        if auto_eps==False and eps is None:
            eps = Clustering.estimate_eps(pc, k=min_samples)
            logger.info(f"Auto-estimated eps: {eps}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm,
                        leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        labels = dbscan.fit_predict(pc.points)
        if min_cluster_size is not None and min_cluster_size > min_samples:
            unique_labels, counts = np.unique(labels, return_counts=True)
            for lbl, cnt in zip(unique_labels, counts):
                if lbl == -1:
                    continue
                if cnt < min_cluster_size:
                    labels[labels == lbl] = -1
            logger.info(f"Filtered out clusters smaller than {min_cluster_size} points.")
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        cluster_info = Clustering._cluster_info(labels)
        logger.info(f"DBSCAN clustering completed with {len(cluster_info)} clusters.")
        return labels, cluster_info

    @staticmethod
    def hdbscan_clustering(
        pc: "PointCloud",
        min_cluster_size: int = 100,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        p: Optional[float] = None,
        alpha: float = 1.0,
        cluster_selection_epsilon: float = 0.0,
        algorithm: str = "best",
        leaf_size: int = 40,
        core_dist_n_jobs: Optional[int] = 1
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        # Create and fit HDBSCAN with the specified parameters.
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            p=p,
            alpha=alpha,
            cluster_selection_epsilon=cluster_selection_epsilon,
            algorithm=algorithm,
            leaf_size=leaf_size,
            core_dist_n_jobs=core_dist_n_jobs
        )

        labels = hdb.fit_predict(pc.points)

        # Assign cluster IDs to the point cloud attributes
        pc.set_attribute("cluster_id", labels.astype(np.int32))

        # Summarize the cluster info
        cluster_info = Clustering._cluster_info(labels)
        logger.info(f"HDBSCAN clustering completed. Found {len(cluster_info)} clusters (excluding noise).")

        return labels, cluster_info
    
    @staticmethod
    def mean_shift_clustering(pc: PointCloud, bandwidth: Optional[float] = None) -> Tuple[np.ndarray, Dict[int, int]]:
        mean_shift = MeanShift(bandwidth=bandwidth)
        labels = mean_shift.fit_predict(pc.points)
        cluster_info = Clustering._cluster_info(labels)
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        logger.info(f"Mean Shift clustering completed with {len(cluster_info)} clusters.")
        return labels, cluster_info


    @staticmethod
    def spectral_clustering(pc: PointCloud, n_clusters: int = 5, affinity="nearest_neighbors") -> Tuple[np.ndarray, Dict[int, int]]:
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
        labels = spectral.fit_predict(pc.points)
        cluster_info = Clustering._cluster_info(labels)
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        logger.info(f"Spectral clustering completed with {n_clusters} clusters.")
        return labels, cluster_info

    @staticmethod
    def optics_clustering(pc: PointCloud, min_samples: int = 10, max_eps: float = np.inf, metric: str = "minkowski", p: float = 2, cluster_method: str = "xi", eps: float = None, xi: float = 0.05, predecessor_correction: bool = True, min_cluster_size: Optional[int] = None, algorithm: str = "auto", leaf_size: int = 30, workers: Optional[int] = None) -> Tuple[np.ndarray, Dict[int, int]]:
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, p=p, cluster_method=cluster_method,
                        eps=eps, xi=xi, predecessor_correction=predecessor_correction,
                        min_cluster_size=min_cluster_size, algorithm=algorithm, leaf_size=leaf_size)
        labels = optics.fit_predict(pc.points)
        cluster_info = Clustering._cluster_info(labels)
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        logger.info(f"OPTICS clustering completed with {len(cluster_info)} clusters.")
        return labels, cluster_info

    @staticmethod
    def ransac_clustering(pc: PointCloud, iterations: int = 1000, min_inliers: int = 50, normal_threshold: float = 30.0) -> Tuple[np.ndarray, Dict[int, int]]:
        points = pc.points
        labels = -np.ones(len(points), dtype=int)
        if len(points) < 3:
            cluster_info = Clustering._cluster_info(labels)
            logger.warning("Not enough points for RANSAC clustering.")
            return labels, cluster_info
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]
        plane, _ = GeometryUtils.fit_plane(sample_points)
        a, b, c, d = plane
        denom = np.sqrt(a*a + b*b + c*c)
        distances = np.abs(a*points[:, 0] + b*points[:, 1] + c*points[:, 2] + d) / denom
        inliers = distances < 0.1
        if inliers.sum() >= min_inliers:
            labels[inliers] = 0
            logger.info(f"RANSAC found a plane with {inliers.sum()} inliers.")
        else:
            logger.warning("RANSAC did not find a valid plane with sufficient inliers.")
        cluster_info = Clustering._cluster_info(labels)
        pc.set_attribute("cluster_id", labels.astype(np.int32))
        return labels, cluster_info

    @staticmethod
    def _cluster_info(labels: np.ndarray) -> Dict[int, int]:
        valid_labels = labels[labels >= 0]
        counts = Counter(valid_labels)
        return dict(counts)
