'''Container of a collection of Jacobian matrices and their timepoints and utilities.'''
import src.constants as cn

import numpy as np  # type: ignore
from typing import List, Set

class JacobianCollection(object):
    """A collection of Jacobian matrices over one or more simulation timepoints."""

    def __init__(self, jacobian_arr: np.ndarray, timepoints: np.ndarray) -> None:
        """
        Parameters
        ----------
        jacobian_arr : np.ndarray
            Array of shape (n_points, n_species, n_species) containing Jacobian matrices.
        timepoints : np.ndarray
            Array of timepoints corresponding to each Jacobian matrix in jacobian_arr.
        """
        self.jacobian_arr = jacobian_arr
        self.timepoints = timepoints

    def getTimes(self) -> Set[float]:
        """Return the unique set of timepoints in this collection."""
        return set(self.timepoints)

    @property
    def max_cv(self) -> float:
        """Compute the maximum coefficient of variation (CV = |std/mean|) across all Jacobian entries."""
        if self.jacobian_arr.size == 0:
            return 0.0
        mean_arr = np.mean(self.jacobian_arr, axis=0)
        std_arr = np.std(self.jacobian_arr, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_arr = np.abs(std_arr / mean_arr)
            cv_arr[~np.isfinite(cv_arr)] = 0.0
        return np.max(cv_arr)