"""
Module for analyzing linearity of SBML and Antimony models via Jacobian analysis.
"""
import src.constants as cn

import collections
import os
import pandas as pd # type: ignore
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np # type: ignore
import tellurium as te # type: ignore
from sklearn.cluster import KMeans # type: ignore

OUTPUT_DATA_FILE = os.path.join(cn.DATA_DIR, "model_linearity_analysis_data.csv")
ClusterResult = collections.namedtuple("ClusterResult", ["clusters", "max_cv"])


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
        num_point : int
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
        if "<?xml" in stripped or "<sbml" in stripped:
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
        if len(self._rr.getFloatingSpeciesIds()) == 0:
            raise ValueError("Model species changed after reset, cannot collect Jacobians.")
        result_arr = self._rr.simulate(self.start, self.end, self.num_point)
        times_arr = np.array(result_arr["time"])  # copy before reset invalidates C++ buffer

        self._rr.reset()
        jacobians = []
        for i, t in enumerate(times_arr):
            if i == 0:
                self._rr.simulate(self.start, t + 1e-10, 2)
            else:
                self._rr.simulate(times_arr[i - 1], t, 2)
            jacobian = np.array(self._rr.getFullJacobian())
            jacobians.append(jacobian.copy())
        # Processed all times
        self._jacobian_arr = np.array(jacobians)
        assert(isinstance(self._jacobian_arr, np.ndarray))
        return self._jacobian_arr

    def makeJacobianCVs(self) -> np.ndarray:
        """
        Compute the coefficient of variation for each cell in the Jacobian.

        The coefficient of variation (CV = |std / mean|) is computed across all
        simulation timepoints for each (i, j) entry of the Jacobian matrix.
        Entries where the mean is zero are set to 0. If Jacobians have not
        been collected yet, collectJacobians() is called automatically.

        Returns
        -------
        np.ndarray
            Array of shape (n_species, n_species) containing the CV for each
            Jacobian cell.
        """
        if self._jacobian_arr is None:
            self.collectJacobians()
        assert self._jacobian_arr is not None

        mean_arr = np.mean(self._jacobian_arr, axis=0)
        std_arr = np.std(self._jacobian_arr, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_arr = np.where(mean_arr != 0, np.abs(std_arr / mean_arr), 0)
        return cv_arr

    def partitionJacobians(
        self, n_cluster: int, max_iter: int = 300
    ) -> ClusterResult:
        """
        Partition the collected Jacobians into n_cluster clusters using
        scikit-learn KMeans. Clusters need not consist of
        contiguous timepoints.

        Each Jacobian matrix is flattened to a feature vector and clustered
        with KMeans (k-means++ init). If Jacobians have not been collected yet,
        collectJacobians() is called automatically.

        Parameters
        ----------
        n_cluster : int
            Number of clusters to partition the Jacobians into.
        max_iter : int
            Maximum number of k-means iterations (default: 300).

        Returns
        -------
        ClusterResult
            Named tuple containing the clustered Jacobians and the maximum CV.

        Raises
        ------
        ValueError
            n_cluster exceeds the number of timepoints
        """
        if self._jacobian_arr is None:
            self.collectJacobians()
        assert self._jacobian_arr is not None
        jacobian_arr = self._jacobian_arr
        n_points = jacobian_arr.shape[0]

        if n_cluster > n_points:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_points})."
            )

        # Flatten each Jacobian to a 1-D feature vector for clustering
        flat_arr = jacobian_arr.reshape(n_points, -1).astype(float)

        kmeans = KMeans(
            n_clusters=n_cluster, init="k-means++", max_iter=max_iter,
            n_init=1, random_state=0,
        )
        labels_arr = kmeans.fit_predict(flat_arr)

        # Compute the max CV within each cluster
        def _cluster_max_cv(indices: np.ndarray) -> float:
            """Return the max CV across all Jacobian entries for the given indices."""
            seg = jacobian_arr[indices]
            mean_arr = np.mean(seg, axis=0)
            std_arr = np.std(seg, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cv_arr = np.where(mean_arr != 0, std_arr / np.abs(mean_arr), 0.0)
            finite_cv = cv_arr[np.isfinite(cv_arr)]
            return float(np.max(finite_cv)) if len(finite_cv) > 0 else 0.0

        cluster_indices = [np.where(labels_arr == c)[0] for c in range(n_cluster)]
        worst_cv = max(
            _cluster_max_cv(idx) for idx in cluster_indices if len(idx) > 0
        )

        return ClusterResult(
            clusters=[jacobian_arr[idx] for idx in cluster_indices],
            max_cv=worst_cv
        )

    def partitionJacobiansSequentially(
        self, n_cluster: int, max_iter: int = 300
    ) -> ClusterResult:
        """
        Partition the collected Jacobians into n_cluster contiguous time segments.

        Unlike partitionJacobians (which uses KMeans and may produce non-contiguous
        clusters), this method constrains each cluster to be a contiguous run of
        timepoints. Dynamic programming is used to find the partition into exactly
        n_cluster contiguous segments that minimises the maximum within-segment CV.

        Parameters
        ----------
        n_cluster : int
            Number of contiguous segments to partition the Jacobians into.
        max_iter : int
            Unused; present for signature compatibility with partitionJacobians.

        Returns
        -------
        ClusterResult
            Named tuple containing the clustered Jacobians (in time order) and the
            maximum CV across all segments.

        Raises
        ------
        ValueError
            n_cluster exceeds the number of timepoints.
        """
        if self._jacobian_arr is None:
            self.collectJacobians()
        assert self._jacobian_arr is not None
        jacobian_arr = self._jacobian_arr
        n_point = jacobian_arr.shape[0]

        if n_cluster > n_point:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_point})."
            )

        def _segment_max_cv(i: int, j: int) -> float:
            """Return the max CV for the contiguous segment [i, j] inclusive."""
            seg = jacobian_arr[i : j + 1]
            if seg.shape[0] == 1:
                return 0.0
            mean_arr = np.mean(seg, axis=0)
            std_arr = np.std(seg, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cv_arr = np.where(mean_arr != 0, std_arr / np.abs(mean_arr), 0.0)
            finite_cv = cv_arr[np.isfinite(cv_arr)]
            return float(np.max(finite_cv)) if len(finite_cv) > 0 else 0.0

        # Precompute cost[i][j] = max CV for segment [i, j]
        cost = np.zeros((n_point, n_point))
        for i in range(n_point):
            for j in range(i, n_point):
                cost[i][j] = _segment_max_cv(i, j)

        # DP: dp[k][i] = min possible max-segment-CV when partitioning
        #     the first i timepoints into k contiguous segments.
        INF = float("inf")
        dp = [[INF] * (n_point + 1) for _ in range(n_cluster + 1)]
        split = [[0] * (n_point + 1) for _ in range(n_cluster + 1)]
        dp[0][0] = 0.0
        for k in range(1, n_cluster + 1):
            for i in range(k, n_point + 1):
                for j in range(k - 1, i):
                    val = max(dp[k - 1][j], cost[j][i - 1])
                    if val < dp[k][i]:
                        dp[k][i] = val
                        split[k][i] = j

        # Reconstruct contiguous segment boundaries from the split table
        boundaries = []
        i = n_point
        for k in range(n_cluster, 0, -1):
            j = split[k][i]
            boundaries.append((j, i))
            i = j
        boundaries.reverse()

        clusters = [jacobian_arr[start:end] for start, end in boundaries]
        worst_cv = dp[n_cluster][n_point]

        return ClusterResult(clusters=clusters, max_cv=worst_cv)

    @classmethod
    def processBioModels(
        cls,
        directory: str = cn.BIOMODELS_DIR,
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

    @classmethod
    def processBioModelsCVs(
        cls,
        directory: str = cn.BIOMODELS_DIR,
        output_data_file: str = OUTPUT_DATA_FILE,
        excluded_models: Optional[List[str]] = None,
        n_cluster: int = 3,
        is_sequential_partition: bool = False,
    ) -> pd.Series:
        """
        Compute Jacobian CVs for all SBML models in subdirectories of directory.

        Parameters
        ----------
        directory : str
            Path to the directory containing BioModel subdirectories. Defaults to
            the local temp-biomodels/final directory.
        output_data_file : str  where the CSV file containing the CV results will be saved.
        excluded_models : Optional[List[str]]
            List of model identifiers to exclude from processing.
        n_cluster : int
            Number of clusters of Jacobians for timepoints to use for k-means clustering.
        is_sequential_partition : bool
            Whether to use sequential partitioning instead of k-means clustering.

        Returns
        -------
        pd.Series
            Series containing the max_CV of the clusters
        """
        if excluded_models is None:
            excluded_models = []

        # Load existing CSV once to identify already-processed models
        existing_ser = pd.Series(dtype=float)
        if os.path.isfile(output_data_file) and os.path.getsize(output_data_file) > 0:
            try:
                df = pd.read_csv(output_data_file, header=None,
                    names=['value'], index_col=0)
                existing_ser = df['value']
            except pd.errors.EmptyDataError:
                pass
        processed_model_ids = set(existing_ser.index.astype(str)) if not existing_ser.empty else set()
        ##
        def _write_csv(result_dct: Dict[str, float]) -> pd.Series:
            """Write the given results to the output CSV, appending to existing data."""
            ser = pd.Series(result_dct)
            write_ser = pd.concat([existing_ser, ser[~ser.index.isin(existing_ser.index)]])
            write_ser.to_csv(output_data_file, index=True)
            return ser
        ##
        # Iterate over models and append results to CSV after each model is processed
        result_dct: Dict[str, float] = {}
        ser = pd.Series(dtype=float)
        for model_dir in sorted(os.listdir(directory)):
            model_dir = model_dir.strip()
            print(model_dir)
            if model_dir in excluded_models:
                print(f"Skipping excluded model: {model_dir}")
                continue
            if model_dir in processed_model_ids:
                print(f"Skipping already-processed model: {model_dir}")
                continue
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
            # Write the CSV on each iteration since this can take a long time
            try:
                with open(sbml_file, "r") as f:
                    sbml_str = f.read()
                analyzer = cls(sbml_str)
                if is_sequential_partition:
                    cluster_result = analyzer.partitionJacobiansSequentially(n_cluster=n_cluster)
                else:
                    cluster_result = analyzer.partitionJacobians(n_cluster=n_cluster)
                max_cv = cluster_result.max_cv
                result_dct[model_dir] = max_cv
                ser = _write_csv(result_dct)
            except Exception as e:
                print(f"Warning: skipping {model_dir}: {e}")
        #
        return ser
