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
import regex as re # type: ignore
from typing import Optional, Tuple

DEFAULT_END_TIME = 10.0
DEFAULT_END_TIME_STR = 'uniformTimeCourse id="auto_ten_seconds"'


class LRoadrunner(object):
    """Creates and manages the lifecycle of a RoadRunner simulation instance."""

    def __init__(self, roadrunner_specification: str,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = None,
            num_points: int = cn.NUM_POINTS,
            sedml_str: Optional[str] = None,
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
        sedml_str : str, optional
            SED-ML string for simulation setup (default: None).
        """
        if not isinstance(roadrunner_specification, str):
            raise ValueError("roadrunner_specification must be a string containing an SBML or Antimony model.")
        self.specification = roadrunner_specification
        # Verify the specification can be loaded as a model, to fail fast if the model is invalid. The loaded model is not stored, so this does not affect the state of the object or its getRoadrunner() method.
        self._loadModel(roadrunner_specification)
        self.start_time = start_time
        self.num_points = num_points
        self._end_time = end_time
        self._sedml_str = sedml_str
        self._end_time_source: Optional[str] = None

    def getRoadrunner(self) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
        return self._loadRoadrunner(self.specification)

    def _loadRoadrunner(self, roadrunner_specification) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
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
        Get the simulation end time, calculating it if not already set.

        If an end time was provided at initialization, that value is returned.
        Otherwise, the end time is calculated using the following methods in order of precedence:
        1. Extracting from the SED-ML string (if provided and valid).
        2. Finding the first simulation end time at which the model reaches steady state.
        3. Estimating from the Jacobian at t=0 (1 / |λ_min|), but only when the rank of the left null space of the stoichiometry matrix is greater than or equal to the number of near-zero eigenvalues of the Jacobian.
        4. Defaulting to 10.0 seconds.

        Returns
        -------
        float
            The simulation end time.
        """
        # Check if end time was has been calculated or provided
        if self._end_time is not None:
            return self._end_time
        #
        # Try to calculate from SED-ML string, if provided.
        self._end_time = self._calculateEndtimeSBML()
        if self._end_time is not None:
            self._end_time_source = cn.ENDTIME_SOURCE_SEDML
            return self._end_time
        # Try to calculate from steady state simulation.
        self._end_time = self._calculateEndtimeSteadystate()
        if self._end_time is not None:
            self._end_time_source = cn.ENDTIME_SOURCE_STEADYSTATE
            return self._end_time
        # Try to calculate from Jacobian at t=0.
        self._end_time = self._calculateEndtimeJacobian()
        if self._end_time is not None:
            self._end_time_source = cn.ENDTIME_SOURCE_RECIROCAL_MIN_EIGENVALUE
            return self._end_time
        # Could not find end_time
        raise ValueError(
            "Could not determine an appropriate end time for this model. "
            "This may be because the model is invalid or unbounded."
        )
    
    def _calculateEndtimeSteadystate(self) -> Optional[float]:
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
        """
        THRESHOLD = 0.01 # In units of the steady state concentrations
        #
        rr = self.getRoadrunner()
        rr.simulate(self.start_time, 5000, 2) # Get the simulation warm
        try:
            option_dct = {"allow_approx": True,
                    "approx_tolerance": 1e-3,
                    "relative_tolerance": 1e-3,
                    "maximum_iterations": 1000}
            solver = rr.getSteadyStateSolver()
            for k, v in option_dct.items():
                solver.setValue(k, v)
            rr.steadyState()
        except RuntimeError as e:
            return None
        ss_arr = np.array(rr.getFloatingSpeciesConcentrations())
        ss_arr = np.array([max(v, 1e-8) for v in ss_arr])  # Avoid division by zero in _isAtSteadyState.
        # Helper that determines if at steady state
        def _isAtSteadyState(end_time: float) -> bool:
            rr.reset()
            final_arr = rr.simulate(self.start_time, end_time, 2)[:, 1:]  # Exclude time column
            is_both_zero = all(np.isclose(ss_arr, 0.0)) and all(np.isclose(final_arr[1], 0.0))
            if is_both_zero:
                return True
            normalized_arr = final_arr[1] / ss_arr
            divergence = np.max(np.abs(normalized_arr - 1))
            return bool(divergence < THRESHOLD)
        #
        # Phase 1: find an end_time where the model is at steady state.
        end_time = 1.0
        while not _isAtSteadyState(end_time):
            end_time *= 2
            if end_time > 1e9:
                return None
        #
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
            if 1 - floor/ceiling < THRESHOLD:
                break
        #
        self._end_time = end_time
        return self._end_time
    
    def _calculateEndtimeSBML(self) -> Optional[float]:
        """
        Estimate an end time from the model's SBML string, if possible.

        This method looks for a SED-ML uniformTimeCourse task with an outputEndTime attribute, which is a common way to specify simulation end times in SBML files. If such a specification is found and valid, it is used as the end time.

        Returns
        -------
        Optional[float]
            The end time specified in the SBML string, or None if no valid specification is found.
        """
        if not self._sedml_str:
            return None
        #
        end_time_match = re.search(r'outputEndTime\s*=\s*"([0-9.]+)"', self._sedml_str)
        if end_time_match:
            end_time_match = re.search(r'outputEndTime\s*=\s*"([0-9.]+)"',
                    self._sedml_str)
            if end_time_match:
                end_time = float(end_time_match.group(1))
                if not DEFAULT_END_TIME_STR in self._sedml_str and end_time > 0:
                    self._end_time = end_time
        #
        return self._end_time
    
    def getSteadyState(self) -> np.ndarray:
        """
        Simulate the model to steady state and return the floating species concentrations.

        Returns
        -------
        np.ndarray
            1-D array of steady-state floating species concentrations.
        """
        rr = self.getRoadrunner()
        rr.reset()
        rr.steadyState()
        return np.array(rr.getFloatingSpeciesConcentrations())
    
    def simulate(self, is_with_timepoints: bool=False) -> np.ndarray:
        """
        Simulate the model for the timepoints defined by start_time, end_time, and num_points.
        Resets before simulating.

        Returns
        -------
        np.ndarray
            2-D array of shape (num_points, num_floating_species) containing the floating species concentrations at each timepoint.
        """
        idx = 1
        if is_with_timepoints:
            idx = 0
        result_arr = self.getRoadrunner().simulate(self.start_time, self.end_time, self.num_points)
        return np.array(result_arr[:, idx:])  # Exclude or include time column
    
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
        rr = self.getRoadrunner()
        if len(rr.getFloatingSpeciesIds()) == 0:
            raise ValueError("Model has no floating species; cannot compute Jacobian.")
        result_arr = rr.simulate(self.start_time, self.end_time, self.num_points)
        times_arr = np.array(result_arr["time"])  # copy before reset invalidates buffer
        #    
        jacobians = []
        for i, t in enumerate(times_arr):
            if i == 0:
                rr.simulate(self.start_time, t + 1e-10, 2)
            else:
                rr.simulate(times_arr[i - 1], t, 2)
            jacobians.append(np.array(rr.getFullJacobian()).copy())

        return np.array(jacobians), times_arr

    def _calculateEndtimeJacobian(self) -> Optional[float]:
        """
        Estimate a simulation end time from the Jacobian evaluated at t=0.

        The end time is estimated as the reciprocal of the smallest-magnitude
        non-zero eigenvalue of the Jacobian at t=0, but only when the rank of
        the left null space of the stoichiometry matrix is greater than or equal
        to the number of near-zero eigenvalues of the Jacobian.  This condition
        ensures the zero eigenvalues are attributable to conservation laws rather
        than dynamic degeneracies, so the smallest non-zero eigenvalue reliably
        sets the dominant timescale.

        Returns
        -------
        Optional[float]
            Estimated end time (1 / |λ_min|), or None if:
            - the condition is not satisfied, or
            - all eigenvalues are near-zero (no finite end time can be derived).

        Notes
        -----
        Zero eigenvalues are identified by |λ| < 1e-10.
        """
        ZERO_TOL = 1e-10

        # (a) Jacobian at t=0.
        # Use a freshly loaded instance to avoid state corruption left by a
        # failed steadyState() call (which can cause getFullJacobian to segfault
        # even after reset()).
        fresh_rr = self._loadModel(self.specification)
        fresh_rr.reset()
        fresh_rr.simulate(self.start_time, self.start_time + 1e-10, 2)
        jacobian_arr = np.array(fresh_rr.getFullJacobian())

        # (b) Rank of the left null space of the stoichiometry matrix.
        #     Left null space dimension = n_species - rank(N).
        #     Rate-rule-only models have no stoichiometry matrix; treat them as
        #     having no conservation laws (left_null_rank = 0).
        try:
            stoich_arr = np.array(fresh_rr.getFullStoichiometryMatrix()).copy()
            n_species = stoich_arr.shape[0]
            stoich_rank = np.linalg.matrix_rank(stoich_arr)
            left_null_rank = n_species - stoich_rank
        except RuntimeError:
            left_null_rank = 0

        # Eigenvalues of the Jacobian (may be complex).
        eigenvalues = np.linalg.eigvals(jacobian_arr)
        magnitudes = np.abs(eigenvalues)
        num_zero_eigs = int(np.sum(magnitudes < ZERO_TOL))

        # (c) Condition: conservation-law count covers all zero eigenvalues.
        if left_null_rank < num_zero_eigs:
            return None

        # (i)/(ii) Smallest non-zero magnitude → end time.
        nonzero_magnitudes = magnitudes[magnitudes >= ZERO_TOL]
        if nonzero_magnitudes.size == 0:
            return None

        min_magnitude = float(np.min(nonzero_magnitudes))
        return 1.0 / min_magnitude