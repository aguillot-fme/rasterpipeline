# pointcloudlib/geometry.py

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from typing import Tuple, Optional, Dict
import logging

from .pointcloud import PointCloud

logger = logging.getLogger(__name__)

class MathUtils:
    @staticmethod
    def compute_covariance(centered_neighbors: np.ndarray) -> np.ndarray:
        cov_matrices = np.einsum('nqc,nqd->ncd', centered_neighbors, centered_neighbors) / centered_neighbors.shape[1]
        return cov_matrices

    @staticmethod
    def eigen_decomposition(cov_matrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = cov_matrices.shape[0]
        eig_vals = np.empty((N, 3), dtype=cov_matrices.dtype)
        eig_vecs = np.empty((N, 3, 3), dtype=cov_matrices.dtype)
        for i in range(N):
            w, v = np.linalg.eigh(cov_matrices[i])
            eig_vals[i] = w
            eig_vecs[i] = v
        logger.debug("Eigen decomposition completed for all covariance matrices.")
        return eig_vals, eig_vecs

    @staticmethod
    def unit_vector(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-12)

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = vectors / norms
            normalized[~np.isfinite(normalized)] = 0
        return normalized

    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return np.zeros_like(v)
        return v / norm

class GeometryUtils:
    @staticmethod
    def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points.shape[0] < 3:
            raise ValueError("Need at least 3 points to fit a plane.")
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        C, residuals, rank, s = np.linalg.lstsq(A, points[:, 2], rcond=None)
        a, b, d = C
        c = -1.0
        plane = np.array([a, b, c, d])
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a*a + b*b + c*c)
        return plane, distances

    @staticmethod
    def project_points_to_2d(points: np.ndarray, plane_coeffs: Tuple[float, float, float, float]) -> np.ndarray:
        a, b, c, d = plane_coeffs
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        ref = np.array([1, 0, 0])
        if np.allclose(normal, ref, atol=1e-6):
            ref = np.array([0, 1, 0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        P0 = -d * normal
        vectors = points - P0
        x_coords = np.dot(vectors, u)
        y_coords = np.dot(vectors, v)
        return np.column_stack((x_coords, y_coords))

    @staticmethod
    def project_polygon_to_3d(polygon_2d: np.ndarray, plane_coeffs: Tuple[float, float, float, float]) -> np.ndarray:
        a, b, c, d = plane_coeffs
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        ref = np.array([1, 0, 0])
        if np.allclose(normal, ref, atol=1e-6):
            ref = np.array([0, 1, 0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        P0 = -d * normal
        X = polygon_2d[:, 0]
        Y = polygon_2d[:, 1]
        points_3d = P0 + X[:, np.newaxis]*u + Y[:, np.newaxis]*v
        return points_3d

    @staticmethod
    def get_boundary_points(points_2d: np.ndarray, num_points: Optional[int] = None, method: str = "convex_hull") -> np.ndarray:
        if points_2d.shape[0] < 3:
            raise ValueError("Not enough points for a hull.")
        hull = ConvexHull(points_2d)
        boundary_points = points_2d[hull.vertices]
        if num_points is not None and num_points < len(boundary_points):
            indices = np.linspace(0, len(boundary_points)-1, num_points, dtype=int)
            boundary_points = boundary_points[indices]
        logger.info(f"Extracted {len(boundary_points)} boundary points using {method}.")
        return boundary_points

    @staticmethod
    def angle_with_vertical(vectors: np.ndarray) -> np.ndarray:
        ez = np.array([0, 0, 1], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        dot_products = (vectors * ez).sum(axis=1) / norms.ravel()
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)
        return angles

    @staticmethod
    def fit_plane_least_squares(neighbor_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_points, k, _ = neighbor_points.shape
        A = np.stack((neighbor_points[:, :, 0], neighbor_points[:, :, 1], np.ones((num_points, k))), axis=-1)
        B = neighbor_points[:, :, 2]
        plane_coeffs = np.zeros((num_points, 4), dtype=np.float32)
        plane_residual = np.zeros(num_points, dtype=np.float32)
        for i in range(num_points):
            Ai = A[i]
            Bi = B[i]
            if Ai.shape[0] < 3:
                plane_coeffs[i] = [0, 0, -1, 0]
                plane_residual[i] = 0.0
                continue
            R = -Bi
            C, residuals, rank, s = np.linalg.lstsq(Ai, R, rcond=None)
            a, b, d = C
            plane_coeffs[i] = [a, b, 1.0, d]
            if residuals.size > 0:
                plane_residual[i] = residuals[0] / Ai.shape[0]
            else:
                plane_residual[i] = 0.0
        logger.debug("Fitted planes to all points using least squares.")
        return plane_coeffs, plane_residual
