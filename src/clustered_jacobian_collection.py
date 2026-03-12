'''Jacobian collections that are the result of clustering'''

import src.constants as cn
from src.jacobian_collection import JacobianCollection  # type: ignore
from typing import List

class ClusteredJacobianCollection(object):
    """A collection of Jacobian matrices that have been clustered into contiguous time spans."""

    def __init__(self, jacobian_collections: List[JacobianCollection]) -> None:
        """
        Parameters
        ----------
        jacobian_collections : list[JacobianCollection]
            List of Jacobian collections for each cluster.
        """
        self.jacobian_collections = jacobian_collections

    @property
    def max_cv(self) -> float:
        """Compute the maximum CV across all clusters."""
        return max(c.max_cv for c in self.jacobian_collections)