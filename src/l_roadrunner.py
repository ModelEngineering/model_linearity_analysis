'''An abstraction of the RoadRunner simulator for use in linearity analysis.'''

"""
Manages creation and lifecycle of a RoadRunner simulation instance.

Usage::

    roadrunner = Roadrunner(roadrunner_specification, start_time=0, end_time=20, num_points=100)
    data = roadrunner.simulate() # Simulates for the timepoints
    ss_data = roadrunner.steadyState() # Simulates steady state
    species_names = roadrunner.species_names
    jacobians = roadrunner.makeJacobians()
"""
import src.constants as cn  # type: ignore
import tellurium as te  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, Tuple


class LRoadrunner(object):
    """Creates and manages the lifecycle of a RoadRunner simulation instance."""

    def __init__(self, roadrunner_specification,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = None,
            num_points: int = cn.NUM_POINTS,
            ) -> None:
        """
        Parameters
        ----------
        roadrunner_specification: str
            SBML/Antimony model string
        start_time : float
            Simulation start time.
        end_time : float
            Simulation end time or calculate the first time at which the model reaches steady state if None.
        num_points : int
            Number of output timepoints.
        """
        self.start_time = start_time
        self.num_points = num_points
        self._end_time = end_time
        self._roadrunner = self._loadRoadRunner(roadrunner_specification)

    @property
    def roadrunner(self) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
        self._roadrunner.reset()
        return self._roadrunner

    def _loadRoadRunner(self, roadrunner_specification) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
        """
        Return a RoadRunner instance from a model string or an existing instance.

        Parameters
        ----------
        roadrunner_specification : str or roadrunner.ExtendedRoadRunner

        Raises
        ------
        ValueError
            If roadrunner_specification is neither a string nor a RoadRunner instance.
        """
        if isinstance(roadrunner_specification, str):
            return self._loadModel(roadrunner_specification)
        if hasattr(roadrunner_specification, "getFloatingSpeciesIds"):
            return roadrunner_specification
        raise ValueError(
            "No RoadRunner instance found. Specification is not a valid model."
        )

    def _loadModel(self, model: str) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
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
        """
        stripped = model.strip()
        if "<?xml" in stripped or "<sbml" in stripped:
            return te.loadSBMLModel(model)
        return te.loada(model)

    @property
    def end_time(self) -> float:
        """
        Find the first simulation end time at which the model reaches steady state.

        The algorithm proceeds in two phases:

        1. Starting from 1 s, double the end time until simulating to that time
        outputs the model within ``threshold`` of the steady-state concentrations
        (deviation normalised by the mean absolute steady-state concentration).
        2. Halve the end time until the model no longer reaches steady state,
        then return the previous (shortest) time that did reach steady state.
        Handles concentrations with different orders of magnitude by normalising 
        the deviation by the mean absolute steady-state concentration.
        This means that the same threshold can be applied to models with very
        different concentration scales.

        Parameters
        ----------
        threshold : float
            Maximum normalised deviation from steady-state concentrations
            accepted as "at steady state". Default is 0.01.

        Returns
        -------
        float
            The first simulation end time at which the model is at steady state.

        Raises
        ------
        ValueError
            If no steady state is found within a simulation time of 1e9.
        """
        if self._end_time is not None:
            return self._end_time
        #
        threshold = 0.01 # In units of the steady state concentrations

        rr = self.roadrunner
        try:
            rr.steadyState()
        except RuntimeError as e:
            raise ValueError(
                "Could not find a steady state for this model. "
                "This may be because the model is invalid or unbounded."
            ) from e
        ss_arr = np.array(rr.getFloatingSpeciesConcentrations())
        ss_arr = np.array([max(v, 1e-8) for v in ss_arr])  # Avoid division by zero in _isAtSteadyState.

        def _isAtSteadyState(end_time: float) -> bool:
            rr.reset()
            rr.simulate(self.start_time, end_time, 2)
            final_arr = np.array(rr.getFloatingSpeciesConcentrations())
            normalized_arr = final_arr / ss_arr
            divergence = np.max(np.abs(normalized_arr - 1))
            return bool(divergence < threshold)

        # Phase 1: find an end_time where the model is at steady state.
        end_time = 1.0
        while not _isAtSteadyState(end_time):
            end_time *= 2
            if end_time > 1e9:
                raise ValueError(
                    "Could not find a steady state within a reasonable simulation time."
                )

        # Phase 2: find the first time the model enters steady state.
        # This is done by binary reduction in the time intervals
        MIN_TIME = 1e-8
        MAX_TIME = 1e8
        floor = end_time / 2
        ceiling = end_time
        while ceiling > MIN_TIME and ceiling < MAX_TIME:
            delta = ceiling - floor
            test_time = floor + delta / 2
            if _isAtSteadyState(test_time):
                floor = test_time
                end_time = test_time
            else:
                ceiling = test_time
            if 1 - floor/ceiling < threshold:
                break
        #
        self._end_time = end_time
        return self._end_time
    
    def getSteadyState(self) -> np.ndarray:
        """
        Simulate the model to steady state and return the floating species concentrations.

        Returns
        -------
        np.ndarray
            1-D array of steady-state floating species concentrations.
        """
        rr = self.roadrunner
        rr.reset()
        rr.steadyState()
        return np.array(rr.getFloatingSpeciesConcentrations())
    
    def simulate(self) -> np.ndarray:
        """
        Simulate the model for the timepoints defined by start_time, end_time, and num_points.
        Resets before simulating.

        Returns
        -------
        np.ndarray
            2-D array of shape (num_points, num_floating_species) containing the floating species concentrations at each timepoint.
        """
        result_arr = self.roadrunner.simulate(self.start_time, self.end_time, self.num_points)
        return np.array(result_arr[:, 1:])  # Exclude time column
    
    def makeJacobians(self)->Tuple[np.ndarray, np.ndarray]:
        """
        Run simulations and collect the full Jacobian at each timepoint.

        The model is reset, simulated once to obtain the output timepoints, reset
        again, then stepped forward point-by-point so a Jacobian can be captured
        at each output time.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the Jacobian matrices and their corresponding timepoints.

        Raises
        ------
        ValueError
            If the model has no floating species after reset.
        """
        if len(self.roadrunner.getFloatingSpeciesIds()) == 0:
            raise ValueError("Model has no floating species; cannot compute Jacobian.")
        result_arr = self.roadrunner.simulate(self.start_time, self._end_time, self.num_points)
        times_arr = np.array(result_arr["time"])  # copy before reset invalidates buffer

        self._roadrunner.reset()
        jacobians = []
        for i, t in enumerate(times_arr):
            if i == 0:
                self._roadrunner.simulate(self.start_time, t + 1e-10, 2)
            else:
                self._roadrunner.simulate(times_arr[i - 1], t, 2)
            jacobians.append(np.array(self._roadrunner.getFullJacobian()).copy())

        return np.array(jacobians), times_arr