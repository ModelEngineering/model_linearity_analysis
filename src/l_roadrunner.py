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

import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import regex as re # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
import tellurium as te  # type: ignore
from typing import Optional, Tuple, List

DEFAULT_END_TIME = 10.0
DEFAULT_END_TIME_STR = 'uniformTimeCourse id="auto_ten_seconds"'
SBML_DEFAULT_END_TIME = 10.0
MAX_ITERATOR_STEP = 50*int(1e6)


def getBiomodelsEndtimes(endtimes_csv_path: str=cn.CALCULATED_ENTIMES_PATH) -> dict[str, float]:
    """
    Load a mapping of BioModels IDs to end times from a CSV file.

    The CSV file should have two columns: "biomodel_id" and "end_time", where "biomodel_id" contains the BioModels ID (e.g., "BIOMD0000000001") and "end_time" contains the corresponding end time as a float.

    Parameters
    ----------
    endtimes_csv_path : str
        Path to the CSV file containing BioModels end times.

    Returns
    -------
    dict
        A dictionary mapping BioModels IDs (str) to end times (float).
    """
    result_dct : dict[str, float] = {}
    if os.path.exists(endtimes_csv_path):
        df = pd.read_csv(endtimes_csv_path)
        if not cn.COL_MODEL_NAME in df.columns or not cn.COL_ENDTIME in df.columns:
            #print(f"Warning: End times CSV file at {endtimes_csv_path} is missing required columns. Returning empty end times mapping.")
            result_dct = {}
            result_dct = {}
        else:
            result_dct = dict(zip(df[cn.COL_MODEL_NAME], df[cn.COL_ENDTIME]))
    return result_dct


class LRoadrunner(object):
    """Creates and manages the lifecycle of a RoadRunner simulation instance."""
    # Class variable to store the mapping of BioModels IDs to end times, loaded once when the class is defined.
    endtime_dct: dict[str, float] = getBiomodelsEndtimes()  # type: ignore

    def __init__(self, roadrunner_specification: str,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = None,
            num_point: int = cn.NUM_POINTS,
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
        self._start_time = start_time
        self.num_point = num_point
        self._end_time: float = end_time if end_time is not None else np.nan
        self._sedml_str = sedml_str
        self.end_time_source: Optional[str] = None
        self._species_names: List[str] = []
        self._timecourse = pd.DataFrame()

    @classmethod
    def makeBiomodel(cls, path:str = "", model_id: str = "",
            model_num: int = 0, **kwargs) -> "LRoadrunner":
        """
        Create an LRoadrunner instance from a BioModels SBML file.

        Parameters
        ----------
        path : str
            Path to the SBML file.
        model_id : str
            ID of the BioModel to load.
        model_num : int
            Number of the model to load.

        Returns
        -------
        LRoadrunner
            An instance of LRoadrunner initialized with the model from the specified SBML file.
        """
        if len(path) == 0:
            if len(model_id) == 0:
                if model_num <= 0:
                    raise ValueError("Must provide at least one of path, model_id, or model_num to load a BioModel.")
                else:
                    model_id = f"BIOMD{model_num:010d}"
            else:
                if not model_id.startswith("BIOMD"):
                    raise ValueError("model_id must start with 'BIOMD'.")
            path = os.path.join(cn.BIOMODELS_DIR, model_id, f"{model_id}_url.xml")
            if not os.path.exists(path):
                path = os.path.join(cn.BIOMODELS_DIR, model_id, "model.xml")
            if not os.path.exists(path):
                import pdb; pdb.set_trace()
                raise FileNotFoundError(f"Model file not found for model {model_id}")

        END_TIME = "end_time"
        with open(path, "r") as f:
            sbml_str = f.read()
        if not END_TIME in kwargs:
            end_time = cls.endtime_dct.get(os.path.basename(path), None)
        else:
            end_time = kwargs[END_TIME]
        kwargs[END_TIME] = end_time
        return cls(sbml_str, **kwargs)

    @property
    def species_names(self) -> list[str]:
        """Get the list of floating species names in the model."""
        if not self._species_names:
            rr = self.getRoadrunner()
            species_ids = rr.getFloatingSpeciesIds()
            self._species_names = [s[1:-1] if s.startswith("[") and s.endswith("]") else s for s in species_ids]
        return self._species_names
    
    @property
    def num_species(self) -> int:
        """Get the number of floating species in the model."""
        return len(self.species_names)

    @property
    def start_time(self) -> float:
        return self._start_time
    
    @start_time.setter
    def start_time(self, start_time: float) -> None:
        self._start_time = start_time

    @property
    def timecourse(self) -> pd.DataFrame:
        """
        Simulate the model and return the time course as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the time course of the simulation, with columns for time and each floating species.
        """
        if self._timecourse.empty:
            result_arr = self.simulate(is_with_timepoints=True)
            df = pd.DataFrame(result_arr, columns=["time"] + self.species_names)
            df = df.set_index("time")
            self._timecourse = df
        return self._timecourse

    def getInitialValues(self) -> np.ndarray:
        """
        Get the initial state of the model as a 1-D array of floating species concentrations.
        """
        rr = self.getRoadrunner()
        rr.simulate(0, self.start_time + 1e-6, 2)
        return np.array(rr.getFloatingSpeciesConcentrations())
    
    def getForcingInputs(self) -> np.ndarray:
        """
        Get the forced input timecourse of the model as a 1-D array of of size the number of floating species.
        Forced inputs are calculated as the difference between the rates of change and the Jacobian applied to the initial state, which is equivalent to the input that would need to be added to the system to make it perfectly linear with respect to the initial state.
        This is used in the linear predictor to extrapolate from the initial state.
        """
        rr = self.getRoadrunner()
        rr.reset()
        _ = rr.simulate(self.start_time, self.end_time, 2)
        jacobian_arr = np.array(rr.getFullJacobian())
        _ = rr.simulate(self.end_time, self.end_time*1.001, 2)
        f_arr = np.array(rr.getRatesOfChange())
        forced_input_arr = f_arr - jacobian_arr @ self.getInitialValues()
        return forced_input_arr

    def getRoadrunner(self) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
        if self.isNull():
            raise ValueError("Cannot get RoadRunner instance from a null LRoadrunner.")
        rr = self._loadModel(self.specification)
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', MAX_ITERATOR_STEP)
        return rr

    def isNull(self) -> bool:
        """Check if this LRoadrunner is a null model (i.e., has no specification)."""
        return self.specification == NULL_L_ROADRUNNER

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
        try:
            result = te.loada(model)
            return result
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

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
        if not np.isnan(self._end_time):
            self.end_time_source = cn.ENDTIME_SOURCE_USER_SPECIFIED
            return self._end_time
        #
        # Try to calculate from SED-ML string, if provided.
        self._end_time = self._calculateEndtimeSBML()
        if not np.isnan(self._end_time):
            self.end_time_source = cn.ENDTIME_SOURCE_SEDML
            return self._end_time
        # Try to calculate from steady state simulation.
        self._end_time = self._calculateEndtimeSteadystate()
        if not np.isnan(self._end_time):
            self.end_time_source = cn.ENDTIME_SOURCE_STEADYSTATE
            return self._end_time
        # Try to calculate by maximizing median CV over the time course.
        self._end_time = self._calculateEndtimeCV()
        if not np.isnan(self._end_time):
            self.end_time_source = cn.ENDTIME_SOURCE_MAX_MEDIAN_CV
            return self._end_time
        # Could not find end_time
        raise ValueError(
            "Could not determine an appropriate end time for this model. "
            "This may be because the model is invalid or unbounded."
        )

    def _calculateEndtimeSteadystate(self) -> float:
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
            return np.nan
        ss_arr = np.array(rr.getFloatingSpeciesConcentrations())
        ss_arr = np.array([max(v, 1e-8) for v in ss_arr])  # Avoid division by zero in _isAtSteadyState.
        if len(ss_arr) == 0:
            return np.nan
        if np.any(np.isnan(ss_arr)) or np.any(np.isinf(ss_arr)):
            return np.nan
        # Helper that determines if at steady state
        rr = self.getRoadrunner()  # Use a fresh instance to avoid state corruption left by a failed steadyState() call (which can cause simulate to segfault even after reset()).
        def _isAtSteadyState(end_time: float) -> bool:
            rr.reset()
            try:
                final_arr = rr.simulate(self.start_time, end_time, 2)[:, 1:]  # Exclude time column
            except Exception:
                return False
            is_both_zero = all(np.isclose(ss_arr, 0.0)) and all(np.isclose(final_arr[1], 0.0))
            if is_both_zero:
                return True
            try:
                normalized_arr = final_arr[1] / ss_arr
            except Exception as e:
                print(f"Exception occurred while normalizing concentrations: {e}")
                return False
            divergence = np.max(np.abs(normalized_arr - 1))
            return bool(divergence < THRESHOLD)
        #
        # Phase 1: find an end_time where the model is at steady state.
        end_time : float = 1.0
        while not _isAtSteadyState(end_time):
            end_time *= 2
            if end_time > 1e9:
                return np.nan
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
    
    def _msg(self, text: str) -> None:
        #print(text)
        pass
    
    def _calculateEndtimeSBML(self) -> float:
        """
        Estimate an end time from the model's SBML string, if possible.

        This method looks for a SED-ML uniformTimeCourse task with an outputEndTime attribute, which is a common way to specify simulation end times in SBML files. If such a specification is found and valid, it is used as the end time.

        Returns
        -------
        float or np.nan
            The end time specified in the SBML string, or np.nan if no valid specification is found.
        """
        if not self._sedml_str:
            self._msg("No SED-ML string provided, skipping SED-ML end time extraction.")
            return self._end_time
        #
        end_time_match = re.search(r'outputEndTime\s*=\s*"([0-9.]+)"', self._sedml_str)
        if end_time_match:
            end_time_match = re.search(r'outputEndTime\s*=\s*"([0-9.]+)"',
                    self._sedml_str)
            if end_time_match:
                end_time = float(end_time_match.group(1))
                if not DEFAULT_END_TIME_STR in self._sedml_str and end_time > 0:
                    self._end_time = end_time
                elif end_time != SBML_DEFAULT_END_TIME:
                    self._end_time = end_time
                else:
                    self._msg("Default end time specification found in SED-ML string, ignoring SED-ML end time.")
            else:
                self._msg("No valid outputEndTime attribute found in SED-ML string.")
        else:
            self._msg("No outputEndTime attribute found in SED-ML string.")
        """ # Validate the end time by simulating to it and checking for errors.
        if self._end_time is not np.nan:
            try:
                rr = self.getRoadrunner()
                rr.simulate(self.start_time, self._end_time, 2)
            except Exception:
                self._end_time = np.nan """
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
        try:
            result_arr = self.getRoadrunner().simulate(self.start_time, self.end_time, self.num_point)
        except Exception as e:
            print(f"Error occurred while simulating: {e}")
            return np.array([]).reshape(0, 0)
        return np.array(result_arr[:, idx:])  # Exclude or include time column
    
    def makeJacobians(self)->Tuple[np.ndarray, np.ndarray]:
        """
        Run simulations and collect the full Jacobian at each timepoint.

        The model is simulated once to obtain the output timepoints, reset,
        then stepped forward point-by-point so a Jacobian can be captured
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
        try:
            result_arr = rr.simulate(self.start_time, self.end_time, self.num_point)
        except RuntimeError as e:
            raise ValueError(f"CVODE failed during initial simulation: {e}") from e
        times_arr = np.array(result_arr["time"])  # copy before reset invalidates buffer

        # For BIOMD7, fails on the first simulation
        rr.reset()
        jacobians = []
        for i, t in enumerate(times_arr):
            try:
                if i == 0:
                    t2 = self.start_time + 1e-10
                    rr.simulate(self.start_time, t2, 2)
                else:
                    rr.simulate(times_arr[i - 1], t, 2)
            except RuntimeError as e:
                raise ValueError(f"CVODE failed at t={t}: {e}") from e
            jacobian_arr = np.array(rr.getFullJacobian()).copy()
            if np.all(np.isclose(jacobian_arr, 0.0)):
                raise ValueError(f"Jacobian at time {t} is all zeros, which may indicate an issue with the model or simulation. Setting Jacobian to all zeros for this timepoint.")
            jacobians.append(jacobian_arr)
        _ = self.timecourse  # Cache the timecourse DataFrame for later use in plotting, to avoid simulating again. This is done after the Jacobian collection loop to avoid state corruption that can cause getFullJacobian to segfault even after reset().
        return np.array(jacobians), times_arr

    def _calculateEndtimeJacobian(self) -> float:
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
        float
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
            return np.nan

        # (i)/(ii) Smallest non-zero magnitude → end time.
        nonzero_magnitudes = magnitudes[magnitudes >= ZERO_TOL]
        if nonzero_magnitudes.size == 0:
            return np.nan

        min_magnitude = float(np.min(nonzero_magnitudes))
        return 1.0 / min_magnitude

    def _calculateEndtimeCV(self) -> float:
        """
        Estimate a simulation end time by maximising the median coefficient of
        variation (CV) of the floating species concentrations over the time course.

        For a given candidate end time T the model is simulated from
        ``start_time`` to T and CV = std / |mean| is computed for each
        floating species across the output timepoints.  The median CV across
        all species is used as the objective.  The end time that maximises
        this objective (found via bounded scalar minimisation in log10-time
        space over the range [1e-3, 1e6]) is returned.

        The rationale is that very short end times capture little dynamics
        (low CV), very long end times show full convergence to steady state
        (low CV), and the optimal window maximises the observable variation.
        The method also works for oscillatory models, whose CV stays elevated
        once a full cycle has been captured.

        Returns
        -------
        float
            End time (in model time units) that maximises the median species
            CV, or ``np.nan`` if the model has no floating species, cannot be
            simulated, or all species maintain a zero mean concentration.
        """
        LOG_LOWER = -5.0   # 10^-3 = 0.001 s
        LOG_UPPER =  6.0   # 10^6  = 1 000 000 s

        rr = self._loadModel(self.specification)
        if len(rr.getFloatingSpeciesIds()) == 0:
            return np.nan

        def _negativeMedianCV(log_end_time: float) -> float:
            end_time = 10.0 ** log_end_time
            try:
                rr.reset()
                try:
                    result_arr = np.array(rr.simulate(self.start_time, end_time, self.num_point))
                except Exception:
                    return 0.0
                data_arr = result_arr[:, 1:]  # Exclude time column
                cvs = []
                for col in range(data_arr.shape[1]):
                    col_data = data_arr[:, col]
                    mean_val = float(np.mean(np.abs(col_data)))
                    if mean_val < 1e-10:
                        continue
                    cvs.append(float(np.std(col_data) / mean_val))
                if not cvs:
                    return 0.0
                return -float(np.median(cvs))
            except Exception:
                return 0.0

        opt = minimize_scalar(_negativeMedianCV, bounds=(LOG_LOWER, LOG_UPPER), method="bounded")
        if opt.fun < 0:  # A genuine maximum was found
            return float(10.0 ** opt.x)
        return np.nan


# Other constants
class DummyLRoadrunner(LRoadrunner):
    def __init__(self, roadrunner_specification: str = "") -> None:
        super().__init__(roadrunner_specification)

    @property
    def end_time(self) -> float:
        return 10.0
    

NULL_L_ROADRUNNER = DummyLRoadrunner()