"""
Module for analyzing linearity of SBML and Antimony models via Jacobian analysis.
"""
import src.constants as cn
from jacobian_collection import JacobianCollection # type: ignore
from clustered_jacobian_collection import ClusteredJacobianCollection  # type: ignore
from biomodels_iterator import BiomodelsIterator  # type: ignore

import os
import pandas as pd # type: ignore
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np # type: ignore
import tellurium as te # type: ignore
from sklearn.cluster import KMeans # type: ignore

from src.l_roadrunner import LRoadrunner # type: ignore

OUTPUT_DATA_FILE = os.path.join(cn.DATA_DIR, "model_linearity_analysis_data.csv")


class LinearAnalyzer:
    """Analyzes linearity of SBML or Antimony models by collecting Jacobians over time."""

    def __init__(
        self,
        model: str,
        start_time: float = 0,
        end_time: Optional[float] = None,
        num_point: int = 100,
        sedml_str: Optional[str] = None,
    ) -> None:
        """
        Initialize a LinearAnalyzer with a model and simulation parameters.

        Parameters
        ----------
        model : str
            SBML XML string or Antimony model string.
        start_time : float
            Simulation start time (default: 0).
        end_time : float, optional
            Simulation end time (default: 10).
        num_point : int
            Number of simulation timepoints (default: 100).
        sedml_str : str, optional
            SED-ML string for simulation setup (default: None).
        """
        self.start_time = start_time
        self.end_time = end_time
        self.num_point = num_point
        self.sedml_str = sedml_str
        self.l_roadrunner = LRoadrunner(model, start_time=start_time,
                end_time=end_time, num_points=num_point)
        self._jacobian_collection = JacobianCollection(self.l_roadrunner)
        self.model = model

    @staticmethod
    def _report(text: str):
        """Print a report message if reporting is enabled."""
        print(text)

    def partitionJacobians(
        self, n_cluster: int, max_iter: int = 300
    ) -> ClusteredJacobianCollection:
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
        jacobian_arr = self._jacobian_collection.jacobian_arr
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

        cluster_indices = [np.where(labels_arr == c)[0] for c in range(n_cluster)]
        # FIXME: Construct with l_roadrunner for collections
        return ClusteredJacobianCollection(
                jacobian_collections=[JacobianCollection.fromArrays(jacobian_arr[idx],
                self._jacobian_collection.timepoint_arr[idx],
                l_roadrunner=self.l_roadrunner)
                for idx in cluster_indices]
        )

    def partitionJacobiansSequentially(self,
            n_cluster: int) -> ClusteredJacobianCollection:
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

        Returns
        -------
        ClusteredJacobianCollection
            Collection of clustered Jacobians (in time order) and the
            maximum CV across all segments.

        Raises
        ------
        ValueError
            n_cluster exceeds the number of timepoints.
        """
        jacobian_arr = self._jacobian_collection.jacobian_arr
        timepoint_arr = self._jacobian_collection.timepoint_arr
        n_point = jacobian_arr.shape[0]

        if n_cluster > n_point:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_point})."
            )

        # Precompute cost[i][j] = max CV for segment [i, j]
        cost = np.zeros((n_point, n_point))
        for i in range(n_point):
            for j in range(i, n_point):
                jacobian_collection = JacobianCollection.fromArrays(jacobian_arr[i : j + 1],
                        timepoint_arr[i : j + 1], self.l_roadrunner)
                cost[i][j] = jacobian_collection.max_cv

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
        timepoint_arr = self._jacobian_collection.timepoint_arr
        timespans = [timepoint_arr[start:end] for start, end in boundaries]
        jacobian_collections = [JacobianCollection.fromArrays(cluster, timespan,
                l_roadrunner=self.l_roadrunner)
                for cluster, timespan in zip(clusters, timespans)] 
        clustered_jacobian_collection = ClusteredJacobianCollection(jacobian_collections)
        #
        return clustered_jacobian_collection
    
    @classmethod
    def _getSedml(cls, directory: str) -> Optional[str]:
        """Search for a SED-ML file in the given directory and return its contents as a string."""
        sedml_files = glob.glob(os.path.join(directory, "*.sedml"))
        if not sedml_files:
            return None
        with open(sedml_files[0], "r") as f:
            return f.read()

    @classmethod
    def partitionBiomodelsJacobians(
        cls,
        directory: str = cn.BIOMODELS_DIR,
        output_data_file: str = OUTPUT_DATA_FILE,
        excluded_models: Optional[List[str]] = None,
        n_cluster: int = 3,
        is_sequential_partition: bool = False,
        is_report: bool = True,
    ) -> pd.DataFrame:
        """
        For each model in BioModels, partition its Jacobians into n_cluster clusters and save
        the max CV of the clusters to a CSV.
        Two partitionation methods are available:
            k-means clustering (partitionJacobians) and sequential partitioning

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
        is_report : bool
            Whether to report progress during processing.

        Returns
        -------
        pd.DataFrame
            DataFrame containing
                index: model identifiers (subdirectory names)
                max_cv: max CV of the Jacobian clusters for each model.
                end_time: end time used for each model's simulation.
        """
        if excluded_models is None:
            excluded_models = []
        iterator = BiomodelsIterator(
            biomodels_dir=directory,
            excluded_models=excluded_models,
            existing_csv_path=output_data_file,
            is_report=is_report,
        )
        existing_df = iterator._existing_df
        ##
        def _write_csv(result_dct: Dict[str, float]) -> pd.DataFrame:
            """Write the given results to the output CSV, appending to existing data."""
            df = pd.DataFrame(result_dct)
            df = pd.concat([existing_df, df], ignore_index=False) if not existing_df.empty else df
            df.set_index(cn.COL_MODEL_NAME, inplace=True)
            df.to_csv(output_data_file, header=True, index=True)
            return df
        ##
        # Iterate over models and append results to CSV after each model is processed
        col_names = list(set(cn.COL_NAMES) - {cn.COL_ENDTIME_SOURCE})
        result_dct: dict = {c: [] for c in col_names}
        for item in iterator:

            clustered_jacobian_collection = cls.makeBiomodelCusteredJacobianCollection(
                item.model_name,
                directory=directory,
                n_cluster=n_cluster,
                is_sequential_partition=is_sequential_partition,
                is_report=is_report,
            )
            result_dct[cn.COL_MODEL_NAME].append(item.model_name)
            result_dct[cn.COL_MAXCV].append(clustered_jacobian_collection.max_cv)
            result_dct[cn.COL_ENDTIME].append(clustered_jacobian_collection.end_time)
            result_df = _write_csv(result_dct)
        #
        result_df = _write_csv(result_dct)
        return result_df
    
    @classmethod
    def makeBiomodelCusteredJacobianCollection(cls,
        model_name: str,
        directory: str = cn.BIOMODELS_DIR,
        start_time: float = cn.START_TIME,
        end_time: float = np.nan,
        num_point: int = cn.NUM_POINTS,
        n_cluster: int = 1,
        is_sequential_partition: bool = True,
        is_report: bool = True,
    ) -> ClusteredJacobianCollection:
        """
        Create a ClusteredJacobianCollection for a single Biomodel.

        Parameters
        ----------
        model_name : str
            The BioModel identifier (e.g. 'BIOMD0000000001').
        directory : str
            Path to the directory containing BioModel subdirectories.
            Defaults to cn.BIOMODELS_DIR.
        start_time : float
            The start time for the simulation.
        num_points : int
            The number of time points to simulate.
        n_cluster : int
            The number of clusters to partition the Jacobians into.
        is_sequential_partition : bool
            Whether to use sequential partitioning instead of k-means clustering.
        is_report : bool
            Whether to print report messages during processing.

        Returns
        -------
        ClusteredJacobianCollection
            A ClusteredJacobianCollection instance containing the clustered Jacobian collections for the model.
        """
        clustered_jacobian_collection = ClusteredJacobianCollection([])
        model_dir = os.path.join(directory, model_name)
        item = BiomodelsIterator.getBiomodelInfo(model_dir)
        sbml_file = item.sbml_paths[0]
        if np.isnan(end_time):
            end_time = LRoadrunner.endtime_dct.get(item.model_name, np.nan) # type: ignore
        if np.isnan(end_time): # type: ignore
            if is_report:
                cls._report(f"Error processing {model_name}: skipping due to missing end time.")
            return clustered_jacobian_collection
        try:
            with open(sbml_file, "r") as f:
                sbml_str = f.read()
            analyzer = cls(sbml_str, start_time=start_time, num_point=num_point, end_time=end_time)
            if is_sequential_partition:
                clustered_jacobian_collection = analyzer.partitionJacobiansSequentially(n_cluster=n_cluster)
            else:
                clustered_jacobian_collection= analyzer.partitionJacobians(n_cluster=n_cluster)
        except Exception as e:
            if is_report:
                cls._report(f"Error processing {model_name}: {e}")
            return clustered_jacobian_collection
        #
        return clustered_jacobian_collection