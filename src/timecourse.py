'''Represents a time course and related properties.'''


import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.biomodels_iterator import getBiomodelsEndtimes  # type: ignore

from collections import namedtuple
import numpy as np  # type: ignore
import pickle
import os
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore
from typing import List, Optional, Tuple

MAX_ITERATOR_STEP = 50 * int(1e6)

SimulationResult = namedtuple('SimulationResult',
        ['timecourse_df', 'jacobian_collection_arr'])


class Timecourse(object):

    def __init__(self, model: Model,
        start_time: float = cn.START_TIME,
        end_time: Optional[float] = None,
        num_point: int = cn.NUM_POINTS,
        timecourse_df: pd.DataFrame = pd.DataFrame(),
        jacobian_collection_arr: np.ndarray = np.array([]),
        ) -> None:
        """ 
        Parameters
        ----------
        model : Model
            The model to simulate.
        start_time : float
            Time to start the simulation.
        end_time : float
            Time to end the simulation.
        num_points : int
            Number of time points to simulate.
        timecourse_df : pd.DataFrame
            Optional pre-computed timecourse DataFrame (index: time, columns: species).
        jacobian_collection_arr : np.ndarray
            Optional pre-computed Jacobian collection (shape: [num_time_points, num_species, num_species]).
        """
        self.model = model
        self.start_time = start_time
        self.end_time = self._updateEndtime(end_time)
        self.num_point = num_point
        #
        self._timecourse_df = timecourse_df
        self._jacobian_collection_arr = jacobian_collection_arr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timecourse):
            return NotImplemented
        return (self.model == other.model and
                bool(np.isclose(self.start_time, other.start_time)) and
                (self.end_time == other.end_time
                        if self.end_time is None or other.end_time is None
                        else bool(np.isclose(self.end_time, other.end_time))) and
                self.num_point == other.num_point and
                bool(np.allclose(self.timecourse_df.values,
                        other.timecourse_df.values)) and
                bool(np.allclose(self.jacobian_collection_arr,
                        other.jacobian_collection_arr)))

    def _updateEndtime(self, end_time: Optional[float]=None)->float | None:
        """Determine the end time and its source."""
        if end_time is not None:
            return end_time
        if self.model.model_name.startswith("BIOMD"):
            endtime_dct = getBiomodelsEndtimes()
            csv_end_time = endtime_dct.get(self.model.model_name, None)
            if csv_end_time is not None:
                return csv_end_time
        return end_time

    @property
    def timecourse_df(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        if self._timecourse_df.empty:
            simulation_result = self._simulate(is_jacobian_collection=False)
            self._timecourse_df = simulation_result.timecourse_df
        return self._timecourse_df
    
    @property
    def jacobian_collection_arr(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        if self._jacobian_collection_arr.size == 0:
            simulation_result = self._simulate(is_jacobian_collection=True)
            self._jacobian_collection_arr = simulation_result.jacobian_collection_arr
            self._timecourse_df = simulation_result.timecourse_df
        return self._jacobian_collection_arr
    
    def _checkSpeciesNames(self, names: List[str]) -> None:
        """Check that the species names in the simulation result match the model."""
        result_species = list(names)
        if result_species != self.model.species_names:
            raise ValueError(
                    f"Simulation species {result_species} do not match "
                    f"model species {self.model.species_names}.")

    def _simulate(self, is_jacobian_collection: bool = False) -> SimulationResult:
        """Create a Trajectory by running a simulation.

        This is the only method that uses RoadRunner.

        end_time resolution order:
            1. Caller-supplied value (source: user_specified).
            2. BioModels CSV lookup (source: sedml).
            3. Auto-detection via _makeEndtime (source: set by that method).

        Parameters
        ----------
        is_jacobian_collection : bool
            Whether to collect Jacobians at each time point.

        Returns
        -------
        SimulationResult
        """
        rr = te.loadSBMLModel(self.model.sbml_str)
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', MAX_ITERATOR_STEP)

        # Timecourse simulation
        rr.reset()
        if self.start_time > 0:
            rr.simulate(0, self.start_time, 2)
        try:
            rr_result = rr.simulate(self.start_time, self.end_time, self.num_point)
        except Exception as e:
            raise ValueError(f"Simulation failed: {e}")
        # Check column order before converting to ndarray (colnames lost after np.array).
        # Skip the leading 'time' column and strip brackets from species names.
        result_species = [
                c[1:-1] if c.startswith("[") and c.endswith("]") else c
                for c in rr_result.colnames[1:]  # type: ignore
        ]
        self._checkSpeciesNames(result_species)
        result_arr = np.array(rr_result)
        timepoint_arr = result_arr[:, 0]
        timecourse_df = pd.DataFrame(
                result_arr[:, 1:],
                index=timepoint_arr,
                columns=self.model.species_names,
        )
        timecourse_df.index.name = "time"

        # Step-by-step simulation to collect Jacobians and forcing inputs
        if is_jacobian_collection:
            rr.reset()
            if self.start_time > 0:
                rr.simulate(0, self.start_time, 2)
            jacobian_collection: List[np.ndarray] = []
            for i, t in enumerate(timepoint_arr):
                if i == 0:
                    rr.simulate(self.start_time, self.start_time + 1e-10, 2)
                else:
                    rr.simulate(timepoint_arr[i - 1], t, 2)
                jacobian_arr = rr.getFullJacobian()
                self._checkSpeciesNames(jacobian_arr.rownames)
                self._checkSpeciesNames(jacobian_arr.colnames)
                jacobian_arr = np.array(jacobian_arr).copy()
                if np.all(np.isclose(jacobian_arr, 0.0)):
                    raise ValueError(
                            f"Jacobian at t={t} is all zeros; model may be degenerate.")
                jacobian_collection.append(jacobian_arr)
        else:
            jacobian_collection = []

        return SimulationResult(
                jacobian_collection_arr=np.array(jacobian_collection),
                timecourse_df=timecourse_df,
        )
    
    def serialize(self) -> str:
        """
        Serialize the Timecourse to a file

        Returns:
            str: The path to the serialized file. 
        """
        if not self.model.model_name:
            raise ValueError("Model must have a name to serialize Timecourse.")
        path = self.makeBiomodelSerializePath(self.model.model_name)
        dct = {
            "model": self.model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_point": self.num_point,
            "timecourse_df": self.timecourse_df,
            "jacobian_collection_arr": self.jacobian_collection_arr,}
        with open(path, 'wb') as f:
            pickle.dump(dct, f)
        return path
    
    @staticmethod
    def makeBiomodelSerializePath(model_name: str) -> str:
        """
        Get the expected path for a serialized Timecourse of a BioModel.

        Parameters:
            model_name (str): The name of the BioModel.
        """
        return os.path.join(cn.TIMECOURSE_SERIALIZATION_DIR, f"{model_name}_timecourse.pkl")

    @classmethod
    def deserialize(cls, path: str = "", model_name: str = "") -> 'Timecourse':
        """
        Deserialize a Timecourse from a file
        At least one of `path` or `model_name` must be provided.
        If both are provided, `path` takes precedence.

        Parameters:
            path (str): The path to the serialized file.
            model_name (str): The name of the BioModel (used if path is not specified).

        Returns:
            Timecourse: The deserialized Timecourse object.
        """
        if not path and not model_name:
            raise ValueError("At least one of `path` or `model_name` must be provided.")
        if not path:
            path = cls.makeBiomodelSerializePath(model_name)
        # Check if the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No serialized Timecourse found at {path}")    
        # Deserialize
        with open(path, 'rb') as f:
            dct = pickle.load(f)
        return cls(
            model=dct['model'],
            start_time=dct['start_time'],
            end_time=dct['end_time'],
            num_point=dct['num_point'],
            timecourse_df=dct['timecourse_df'],
            jacobian_collection_arr=dct['jacobian_collection_arr']
        )