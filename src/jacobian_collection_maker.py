"""
Creates a JacobianCollection. Can be used to reinstatiate a RoadRunner instance if it was deleted after initialization. 

Usage:

maker = JacobianCollectionMaker(model_string)
jacobian_collection = maker.makeCollection()
"""
import src.constants as cn
from jacobian_collection import JacobianCollection  # type: ignore

from typing import Union, List

import numpy as np  # type: ignore
import tellurium as te  # type: ignore


class JacobianCollectionMaker:

    def __init__(self, roadrunner_specification,
            start_time: float=cn.START_TIME,
            end_time: float=cn.END_TIME,
            num_points: int=cn.NUM_POINTS,
            is_keep_roadrunner: bool=True,
            ) -> None:
        """
        Parameters
        ----------
        roadrunner_specification : str or roadrunner.ExtendedRoadRunner
            SBML/Antimony model string or an existing RoadRunner instance.
        start_time : float
            Simulation start time.
        end_time : float
            Simulation end time.
        num_points : int
            Number of output timepoints.
        is_keep_roadrunner : bool
            If False, the RoadRunner instance is deleted after initialization to
            free memory. Can be restored later with restoreRoadRunner() only if
            initialized from a string.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._num_points = num_points
        # Store the string specification for later restoration; None if given an rr instance
        if isinstance(roadrunner_specification, str):
            self._roadrunner_specification = roadrunner_specification
        else:
            self._roadrunner_specification = None
        #
        self._roadrunner = self._getRoadRunner(roadrunner_specification)
        if not is_keep_roadrunner:
            self.deleteRoadRunner()

    def deleteRoadRunner(self):
        """Delete the RoadRunner instance to free memory. Can be restored later with restoreRoadRunner."""
        if hasattr(self, "_roadrunner"):
            del self._roadrunner

    def _getRoadRunner(self, roadrunner_specification) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
        """
        Sets self._roadrunner to a new RoadRunner instance loaded with the original model specification.

        This can be used to restore a RoadRunner instance after it was deleted, or to set a different model.

        """
        if isinstance(roadrunner_specification, str):
            roadrunner = self._loadModel(roadrunner_specification)
        elif hasattr(roadrunner_specification, "getFloatingSpeciesIds"):
            roadrunner = roadrunner_specification
        else:
            raise ValueError("jacobian_specification must be a model string or a RoadRunner instance.")
        return roadrunner

    def restoreRoadRunner(self):
        """
        Restore a RoadRunner instance if it was deleted after initialization.

        Raises
        ------
        ValueError
            If the original model cannot be restored (e.g., if it was not a string).
        """
        if hasattr(self, "_roadrunner"):
            return
        if self._roadrunner_specification is None:
            raise ValueError(
                "Could not restore RoadRunner instance because it was created with a RoadRunner instance, not a string."
            )
        self._roadrunner = self._getRoadRunner(self._roadrunner_specification)

    def _loadModel(self, model: str) -> te.roadrunner.ExtendedRoadRunner:  # type: ignore
        """
        Load a model from an SBML or Antimony string.

        SBML is detected by the presence of an XML declaration or <sbml> root tag.
        All other strings are treated as Antimony.

        Parameters
        ----------
        model : str
            SBML XML string or Antimony model string.

        Returns
        -------
        te.roadrunner.ExtendedRoadRunner
            The loaded RoadRunner model instance.
        """
        stripped = model.strip()
        if "<?xml" in stripped or "<sbml" in stripped:
            return te.loadSBMLModel(model)
        return te.loada(model)

    def makeCollection(self)-> JacobianCollection:
        """
        Run simulations and collect the full Jacobian at each timepoint.

        The model is reset, simulated once to obtain the output timepoints, reset
        again, then stepped forward point-by-point so a Jacobian can be captured
        at each output time.

        Returns
        -------
        JacobianCollection
            Collection of Jacobian matrices and their corresponding timepoints.

        Raises
        ------
        ValueError
            If the model has no floating species after reset.
        """
        self._roadrunner.reset()
        result_arr = self._roadrunner.simulate(self._start_time, self._end_time, self._num_points)
        times_arr = np.array(result_arr["time"])  # copy before reset invalidates buffer

        self._roadrunner.reset()
        jacobians = []
        for i, t in enumerate(times_arr):
            if i == 0:
                self._roadrunner.simulate(self._start_time, t + 1e-10, 2)
            else:
                self._roadrunner.simulate(times_arr[i - 1], t, 2)
            jacobians.append(np.array(self._roadrunner.getFullJacobian()).copy())

        return JacobianCollection(np.array(jacobians), times_arr)