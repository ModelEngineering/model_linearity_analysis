"""
Module defining JacobianCluster, a container that pairs Jacobian matrices
with their simulation timepoints.
"""

from typing import List, Set, Tuple

import numpy as np # type: ignore


class JacobianCluster:
    """A container that stores Jacobian matrices and their associated timepoints."""

    def __init__(self, jacobians: List[np.ndarray], indices: List[int]) -> None:
        """
        Args:
            jacobians (List[np.ndarray]): _description_
            indices (List[int]): _description_
        """
        self._jacobians: List[np.ndarray] = jacobians
        self._indices: List[int] = indices
        self._shape: Tuple[int, int] = self._jacobians[0].shape if jacobians else (0, 0)
        self._max_cv: float = np.nan

    @property
    def min_index(self) -> int:
        """Return the minimum index associated with the Jacobian matrices in this cluster."""
        return min(self._indices) if self._indices else -1

    @property
    def max_index(self) -> int:
        """Return the maximum index associated with the Jacobian matrices in this cluster."""
        return max(self._indices) if self._indices else -1
    
    @property
    def isSequential(self) -> bool:
        """Return True if the indices in this cluster are sequential (contiguous), False otherwise."""
        sorted_indices = sorted(self._indices)
        return all(
            sorted_indices[i] == sorted_indices[i - 1] + 1
            for i in range(1, len(sorted_indices))
        )

    @property
    def max_cv(self) -> float:
        """
        Compute the maximum coefficient of variation (CV) across all Jacobian matrices in this cluster.

        Returns
        -------
        float
            The maximum CV value across all Jacobian matrices in this cluster.
        """
        if self._max_cv is not np.nan:
            return self._max_cv
        stack = np.stack(self._jacobians, axis=0)   # shape (n, rows, cols)
        mean_arr = np.mean(stack, axis=0)  # shape (rows, cols)
        std_arr  = np.std(stack, axis=0)
        cv_arr = std_arr / mean_arr if mean_arr != 0 else 0.0# shape (rows, cols)
        self._max_cv = np.max(cv_arr)
        return self._max_cv
    
    @classmethod
    def multiClusterMax(cls, clusters: List["JacobianCluster"]) -> float:
        """
        Compute the maximum CV across multiple JacobianCluster instances.

        Args:
            clusters (List[JacobianCluster]): A list of JacobianCluster instances.

        Returns
        -------
        float
            The maximum CV value across all provided JacobianCluster instances.
        """
        return max(cluster.max_cv for cluster in clusters) if clusters else np.nan