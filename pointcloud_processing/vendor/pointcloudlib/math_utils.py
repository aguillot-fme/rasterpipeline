# pointcloudlib/math_utils.py

import numpy as np
from typing import Tuple
import logging

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
