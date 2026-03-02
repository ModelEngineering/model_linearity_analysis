"""
Module for analyzing linearity of SBML and Antimony models via Jacobian analysis.
"""

import os
import glob
from typing import List, Optional, Tuple

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import tellurium as te # type: ignore


class LinearAnalyzer:
    """Analyzes linearity of SBML or Antimony models by collecting Jacobians over time."""

    def __init__(
        self,
        model: str,
        start: float = 0,
        end: float = 10,
        num_point: int = 100,
    ) -> None:
        """
        Initialize a LinearAnalyzer with a model and simulation parameters.

        Parameters
        ----------
        model : str
            SBML XML string or Antimony model string.
        start : float
            Simulation start time (default: 0).
        end : float
            Simulation end time (default: 10).
        num_points : int
            Number of simulation timepoints (default: 100).
        """
        self.model = model
        self.start = start
        self.end = end
        self.num_point = num_point
        self._rr = self._loadModel(model)
        self._jacobian_arr: Optional[np.ndarray] = None
        self._species_ids: List[str] = self._rr.getFloatingSpeciesIds()

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
        if stripped.startswith("<?xml") or stripped.startswith("<sbml"):
            return te.loadSBMLModel(model)
        return te.loada(model)

    def collectJacobians(self) -> np.ndarray:
        """
        Run the model and collect the full Jacobian at each simulation timepoint.

        The Jacobian contains only floating species. The model is reset before
        simulation and stepped forward timepoint-by-timepoint so a Jacobian can
        be captured at each output time.

        Returns
        -------
        np.ndarray
            Array of shape (num_points, n_species, n_species) containing the
            Jacobian matrix at each timepoint.
        """
        self._rr.reset()
        result = self._rr.simulate(self.start, self.end, self.num_point)
        times = result["time"]

        self._rr.reset()
        jacobians = []
        for i, t in enumerate(times):
            if i == 0:
                self._rr.simulate(self.start, t + 1e-10, 2)
            else:
                self._rr.simulate(times[i - 1], t, 2)
            jacobians.append(np.array(self._rr.getFullJacobian()))

        self._jacobian_arr = np.array(jacobians)
        assert(isinstance(self._jacobian_arr, np.ndarray))
        return self._jacobian_arr

    def plot(self) -> plt.Figure:  # type: ignore
        """
        Create a heatmap of the coefficient of variation for each Jacobian entry.

        The coefficient of variation (CV = |std / mean|) is computed across all
        timepoints for each matrix entry. Entries where the mean is zero are shown
        as NaN. If Jacobians have not been collected yet, collectJacobians() is
        called automatically.

        Returns
        -------
        plt.Figure
            The matplotlib Figure containing the heatmap.
        """
        if self._jacobian_arr is None:
            self.collectJacobians()
        assert self._jacobian_arr is not None

        mean = np.mean(self._jacobian_arr, axis=0)
        std = np.std(self._jacobian_arr, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(mean != 0, np.abs(std / mean), np.nan)

        species = self._species_ids
        n = len(species)
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        sns.heatmap(
            cv,
            ax=ax,
            xticklabels=species,
            yticklabels=species,
            #cmap="viridis",
            cmap="Reds",
            annot=n <= 10,
            vmin=0,
            vmax=1,
        )
        ax.set_title("Coefficient of Variation of Jacobian Entries")
        ax.set_xlabel("Species")
        ax.set_ylabel("Species")
        plt.tight_layout()
        return fig

    @classmethod
    def processBioModels(
        cls,
        directory: str = "/Users/jlheller/home/Technical/repos/temp-biomodels/final",
    ) -> List[Tuple[str, "LinearAnalyzer"]]:
        """
        Process all SBML models found in subdirectories of the given directory.

        Each subdirectory is expected to contain one or more XML files. The first
        non-manifest XML file found is loaded as the SBML model. Models that fail
        to load or simulate are skipped with a warning.

        Parameters
        ----------
        directory : str
            Path to the directory containing BioModel subdirectories. Defaults to
            the local temp-biomodels/final directory.

        Returns
        -------
        List[Tuple[str, LinearAnalyzer]]
            List of (model_id, LinearAnalyzer) tuples for successfully processed
            models, where model_id is the subdirectory name.
        """
        results = []
        for model_dir in sorted(os.listdir(directory)):
            model_path = os.path.join(directory, model_dir)
            if not os.path.isdir(model_path):
                continue
            sbml_files = [
                f
                for f in glob.glob(os.path.join(model_path, "*.xml"))
                if not f.endswith("manifest.xml")
            ]
            if not sbml_files:
                continue
            sbml_file = sbml_files[0]
            try:
                with open(sbml_file, "r") as f:
                    sbml_str = f.read()
                analyzer = cls(sbml_str)
                analyzer.collectJacobians()
                results.append((model_dir, analyzer))
            except Exception as e:
                print(f"Warning: skipping {model_dir}: {e}")
        return results
