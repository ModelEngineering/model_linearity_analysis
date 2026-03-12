"""
Manages creation and lifecycle of a RoadRunner simulation instance.

Usage::

    maker = RoadRunnerMaker(model_string)
    rr = maker.getRoadRunner()
    maker.deleteRoadRunner()
    maker.restoreRoadRunner()
    rr = maker.getRoadRunner()
"""
import src.constants as cn  # type: ignore
import tellurium as te  # type: ignore

NO_SPECIFICATION = "no_specification"


class RoadRunnerMaker:
    """Creates and manages the lifecycle of a RoadRunner simulation instance."""

    def __init__(self, roadrunner_specification,
            start_time: float = cn.START_TIME,
            end_time: float = cn.END_TIME,
            num_points: int = cn.NUM_POINTS,
            is_keep_roadrunner: bool = True,
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

        if isinstance(roadrunner_specification, str):
            self._roadrunner_specification = roadrunner_specification
        else:
            self._roadrunner_specification = NO_SPECIFICATION

        self._roadrunner = self._loadRoadRunner(roadrunner_specification)
        if not is_keep_roadrunner:
            self.deleteRoadRunner()

    @property
    def roadrunner(self) -> "te.roadrunner.ExtendedRoadRunner":  # type: ignore
        if not hasattr(self, "_roadrunner"):
            self._loadModel(self._roadrunner_specification)
        return self._roadrunner

    def deleteRoadRunner(self) -> None:
        """Delete the RoadRunner instance to free memory."""
        if hasattr(self, "_roadrunner"):
            del self._roadrunner

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
