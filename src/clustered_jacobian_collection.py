'''Jacobian collections that are the result of clustering'''

import src.constants as cn
from src.jacobian_collection import JacobianCollection  # type: ignore

import numpy as np  # type: ignore
from typing import List, Optional

class ClusteredJacobianCollection(object):
    """A collection of Jacobian matrices that have been clustered into contiguous time spans."""

    def __init__(self, jacobian_collections: List[JacobianCollection],
            end_time: Optional[float]=np.nan) -> None:
        """
        Parameters
        ----------
        jacobian_collections : list[JacobianCollection]
            List of Jacobian collections for each cluster.
        end_time : float, optional
            The end time of the simulation, used for normalization. If None, no normalization is applied
        """
        self.jacobian_collections = jacobian_collections
        self.end_time = end_time

    @property
    def max_cv(self) -> float:
        """Compute the maximum CV across all clusters."""
        if len(self.jacobian_collections) == 0:
            return np.nan
        return max(c.max_cv for c in self.jacobian_collections)