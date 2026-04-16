'''Container of a collection of Jacobian matrices and their timepoints and utilities.'''
import src.constants as cn
from src.l_roadrunner import LRoadrunner, NULL_L_ROADRUNNER  # type: ignore

import collections
import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from typing import List, Optional, Tuple, cast


PlotInfo = collections.namedtuple("PlotInfo",
        ["top_ax", "bottom_ax", "fig"])

class JacobianCollection(object):
    """A collection of Jacobian matrices over one or more simulation timepoints."""

    def __init__(self, l_roadrunner: LRoadrunner=NULL_L_ROADRUNNER,
            diameter_metric: str = cn.DIAMETER_WEIGHTED_EIGENVECTORS) -> None:
        """
        Parameters
        ----------
        l_roadrunner : LRoadrunner
            An LRoadrunner instance used to simulate the model and collect Jacobians.
            Jacobians and timepoints are obtained by calling makeJacobians() on this object.
        diameter_metric : str
            The metric to use for calculating the diameter of the Jacobian collection.
            Options are "weighted_eigenvectors" (default) or "max_cv".
        """
        self._initialize(l_roadrunner)
        try:
            self.jacobian_arr, self.timepoint_arr = l_roadrunner.makeJacobians()
            self._sortArrays()
        except Exception as e:
            raise ValueError(f"Failed to create JacobianCollection for {l_roadrunner.specification[0:200]}") from e
        self._diameter_metric = diameter_metric

    def _initialize(self, l_roadrunner: LRoadrunner) -> None:
        """Initialize the JacobianCollection with a new LRoadrunner instance."""
        self.l_roadrunner = l_roadrunner
        self._jacobian_mean_arr = np.array([])
        self._jacobian_std_arr = np.array([])
        self._diameter = np.nan
        self._diameter_metric = cn.DIAMETER_WEIGHTED_EIGENVECTORS
    
    def _sortArrays(self) -> None:
        """Sort the jacobian_arr and timepoint_arr by timepoint."""
        sort_indices = np.argsort(self.timepoint_arr)
        self.timepoint_arr = self.timepoint_arr[sort_indices]
        self.jacobian_arr = self.jacobian_arr[sort_indices]
    
    @classmethod
    def fromArrays(cls, jacobian_arr: np.ndarray, timepoint_arr: np.ndarray,
            l_roadrunner: LRoadrunner = NULL_L_ROADRUNNER):
        """Create a JacobianCollection from explicit arrays."""
        jc = object.__new__(cls)
        jc._initialize(l_roadrunner)
        jc.jacobian_arr = jacobian_arr
        jc.timepoint_arr = timepoint_arr
        jc._sortArrays()
        return jc

    def getTimes(self) -> np.ndarray:
        """Return the sorted unique timepoints in this collection."""
        return np.unique(self.timepoint_arr)
    
    @property
    def jacobian_mean_arr(self) -> np.ndarray:
        """Compute the element-wise mean Jacobian across all timepoints."""
        if self._jacobian_mean_arr.size == 0:
            self._jacobian_mean_arr = np.mean(self.jacobian_arr, axis=0)
        return self._jacobian_mean_arr

    @property
    def max_cv(self) -> float:
        """Compute the maximum coefficient of variation (CV = |std/mean|) across all Jacobian entries."""
        if self.jacobian_arr.size == 0:
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_arr = np.abs(self.jacobian_std_arr / self.jacobian_mean_arr)
            cv_arr[~np.isfinite(cv_arr)] = 0.0
        return np.max(cv_arr)
    
    @staticmethod
    def _calculateWeightedEigenvectors(jacobian_arr: np.ndarray) -> np.ndarray:
        """
        Calculate the eigenvectors of each Jacobian weighted by their eigenvalues.
        This is essentially the solution to an initial value problems
        at time t=1 with initial conditions that yield constants c_i=1 for each eigenvector.
        The result is a vector that captures the dominant modes of variation in the
        Jacobian, weighted by their growth rates (eigenvalues).
        """
        eigvals, eigvecs = np.linalg.eig(jacobian_arr)
        result_arr = eigvecs @ np.exp(eigvals)
        return result_arr

    @property
    def diameter(self) -> float:
        if self._diameter_metric == cn.DIAMETER_MAX_CV:
            return self.max_cv
        return self.weighted_eigenvector_diameter

    @property
    def weighted_eigenvector_diameter(self) -> float:
        """
        Compute the maximum distance between the centroid weighted eigenvector
        (the element-wise mean of the weighted eigenvectors) and any individual weighted eigenvector in the collection.
        (the element-wise mean) and any individual Jacobian in the collection.   
        The distance is the L2 norm.
        """
        if not np.isnan(self._diameter):
            return cast(float, self._diameter)
        if self.jacobian_arr.size == 0:
            return 0.0
        weighted_arrs: list[np.ndarray] = []
        for jacobian_arr in self.jacobian_arr:
            weighted_arr = self._calculateWeightedEigenvectors(jacobian_arr)
            weighted_arrs.append(weighted_arr)
        mean_weighted_arr = np.mean(weighted_arrs, axis=0)
        max_distance = 0.0
        for weighted_arr in weighted_arrs:
            distance = np.linalg.norm(weighted_arr - mean_weighted_arr)
            max_distance = max(max_distance, distance)
        self._diameter = max_distance
        return cast(float, self._diameter)

    @property
    def jacobian_std_arr(self) -> np.ndarray:
        """Compute the element-wise standard deviation of Jacobians across all timepoints."""
        if self._jacobian_std_arr.size == 0:
            self._jacobian_std_arr = np.std(self.jacobian_arr, axis=0)
        return self._jacobian_std_arr

    def _calculateDeviation(self) -> np.ndarray:
        """
        Calculate the Frobenius-norm distance of each Jacobian from the centroid.

        The centroid is the element-wise mean of all Jacobians in jacobian_arr.
        For each timepoint the deviation is ||J(t) - centroid||_F.

        Returns
        -------
        np.ndarray
            1-D array of shape (num_points,) containing the deviation at each timepoint.
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            diff_arr = np.abs(self.jacobian_arr - self.jacobian_mean_arr)/np.abs(self.jacobian_mean_arr)
        diff_arr[:, np.abs(self.jacobian_mean_arr) == 0] = 0.0
        result = np.sqrt(np.sum(diff_arr**2, axis=(1,2)))
        return result

    def plot(self,
            top_ax: Optional[plt.Axes] = None,   # type: ignore
            bottom_ax: Optional[plt.Axes] = None, # type: ignore
            fig: Optional[plt.Figure] = None,  # type: ignore
            is_legend: bool = True,
            ylim: Tuple[float, float] = (0.0, 1.0),
            ) -> PlotInfo:
        """
        Constructs a figure with two plots with time on the x-axis: (1) the Frobenius-norm distance of each Jacobian from the centroid, and
        (2) the timecourse of simulation species concentrations.
        The first plot shows how the Jacobian changes over time relative to the centroid.
        The second plot shows the dynamics of the model's species concentrations
        over time.

        Parameters
        ----------
        top_ax : Optional[plt.Axes]
            An optional matplotlib Axes object to use for the top plot. If None, a new figure and axes will be created.
        bottom_ax : Optional[plt.Axes]
            An optional matplotlib Axes object to use for the bottom plot. If None, a new figure and axes will be created.
        fig : Optional[plt.Figure]
            An optional matplotlib Figure object to use. If None, a new figure will be created.
        """

        if hasattr(self.l_roadrunner, "getRoadrunner"):
            roadrunner = self.l_roadrunner.getRoadrunner()
            species_ids = roadrunner.getFloatingSpeciesIds()
            data_arr = self.l_roadrunner.simulate(is_with_timepoints=True)
            species_data = data_arr[:, 1:]  # Exclude time column
            species_times = data_arr[:, 0]  # Extract time column
        else:
            raise ValueError("Cannot plot species timecourse has a NULL LRoadrunner instance.") 
        jacobian_times = self.getTimes()
        deviation_arr = self._calculateDeviation()

        if top_ax is None or bottom_ax is None or fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
        else:
            ax1 = top_ax
            ax2 = bottom_ax

        ax1.plot(jacobian_times, deviation_arr, marker="o")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Normalized distance")
        ax1.set_title("Normalized Distance of Jacobian to Centroid")
        ax1.set_ylim(ylim)

        for i, species_id in enumerate(species_ids):
            ax2.plot(species_times, species_data[:, i], label=species_id)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Concentration")
        ax2.set_title("Species Timecourse")
        if is_legend:
            ax2.legend()

        fig.tight_layout()
        plt.show()
        return PlotInfo(top_ax=ax1, bottom_ax=ax2, fig=fig)

    def nonsequentialPartition(self, n_cluster: int, max_iter: int = 300) -> List["JacobianCollection"]:
        """Partition the Jacobians into n_cluster clusters using KMeans.

        Clusters need not consist of contiguous timepoints. Each Jacobian
        matrix is flattened to a feature vector and clustered with KMeans
        (k-means++ init).

        Parameters
        ----------
        n_cluster : int
            Number of clusters to partition the Jacobians into.
        max_iter : int
            Maximum number of k-means iterations (default: 300).

        Returns
        -------
        List[JacobianCollection]
            One JacobianCollection per cluster.

        Raises
        ------
        ValueError
            If n_cluster exceeds the number of timepoints.
        """
        n_points = self.jacobian_arr.shape[0]
        if n_cluster > n_points:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_points})."
            )
        flat_arr = self.jacobian_arr.reshape(n_points, -1).astype(float)
        kmeans = KMeans(
            n_clusters=n_cluster, init="k-means++", max_iter=max_iter,
            n_init=1, random_state=0,
        )
        labels_arr = kmeans.fit_predict(flat_arr)
        cluster_indices = [np.where(labels_arr == c)[0] for c in range(n_cluster)]
        return [
            JacobianCollection.fromArrays(
                self.jacobian_arr[idx], self.timepoint_arr[idx], self.l_roadrunner)
            for idx in cluster_indices
        ]

    def sequentialPartition(self, n_cluster: int,
            cost_criteria: str = "expo_eigen") -> List["JacobianCollection"]:
        """Partition the Jacobians into n_cluster contiguous time segments.

        Dynamic programming finds the partition into exactly n_cluster
        contiguous segments that minimises the maximum within-segment cost.

        Parameters
        ----------
        n_cluster : int
            Number of contiguous segments.
        cost_criteria : str
            Cost metric per segment: ``"expo_eigen"`` (default) or ``"max_cv"``.

        Returns
        -------
        List[JacobianCollection]
            One JacobianCollection per segment, in time order.

        Raises
        ------
        ValueError
            If n_cluster exceeds the number of timepoints.
        """
        CRITERIA_MAX_CV = "max_cv"
        CRITERIA_EXPO_EIGEN = "expo_eigen"
        n_point = self.jacobian_arr.shape[0]
        if n_cluster > n_point:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_point})."
            )
        cost = np.zeros((n_point, n_point))
        for i in range(n_point):
            for j in range(i, n_point):
                jc = JacobianCollection.fromArrays(
                        self.jacobian_arr[i:j + 1],
                        self.timepoint_arr[i:j + 1],
                        self.l_roadrunner)
                if cost_criteria == CRITERIA_MAX_CV:
                    cost[i][j] = jc.max_cv
                elif cost_criteria == CRITERIA_EXPO_EIGEN:
                    cost[i][j] = jc.diameter
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
        boundaries = []
        i = n_point
        for k in range(n_cluster, 0, -1):
            j = split[k][i]
            boundaries.append((j, i))
            i = j
        boundaries.reverse()
        return [
            JacobianCollection.fromArrays(
                self.jacobian_arr[start:end],
                self.timepoint_arr[start:end],
                self.l_roadrunner)
            for start, end in boundaries
        ]

    def heatmap(self, ax: Optional[maxes.Axes] = None) -> mfigure.Figure:
        """Construct a heatmap of the Jacobian where cells are colored by their
        coefficient of variation (CV = |std/mean|) and labelled with their mean value.

        Zero-mean entries (where CV is undefined) are shown with CV=0 and labelled
        with their mean (0.0).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created when omitted.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the heatmap.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_arr = np.abs(self.jacobian_std_arr / self.jacobian_mean_arr)
        cv_arr = np.where(np.isfinite(cv_arr), cv_arr, 0.0)

        mean_arr = self.jacobian_mean_arr
        annot_arr = np.vectorize(lambda v: f"{v:.2g}")(mean_arr)

        rr = self.l_roadrunner.getRoadrunner() if hasattr(self, "l_roadrunner") else None
        species_ids = rr.getFloatingSpeciesIds() if rr is not None else None

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
            assert isinstance(fig, mfigure.Figure)

        sns.heatmap(
            cv_arr,
            annot=annot_arr,
            fmt="",
            cmap="coolwarm",
            cbar_kws={"label": "CV (|std/mean|)"},
            xticklabels=species_ids if species_ids is not None else "auto",
            yticklabels=species_ids if species_ids is not None else "auto",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_xlabel("Species")
        ax.set_ylabel("Species")
        ax.set_title("Jacobian: CV (color) and mean (label)")
        assert isinstance(fig, mfigure.Figure)
        fig.tight_layout()
        return fig