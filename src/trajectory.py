'''Represents a trajectory, the results of a model simulation.'''

import src.constants as cn
from src.l_roadrunner import LRoadrunner, NULL_L_ROADRUNNER  # type: ignore

import collections
import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from lmfit import Parameters, minimize as lmfit_minimize  # type: ignore
from typing import List, Optional, Tuple, cast


# Times relative to the characteristic time at which to evaluate the weighted eigenvectors
IVP_RELATIVE_TIMES = np.array([0, 0.1, 0.3])
NULL_COST = -1


PlotInfo = collections.namedtuple("PlotInfo",
        ["top_ax", "bottom_ax", "fig"])

class Trajectory(object):
    """A collection of Jacobian matrices over one or more simulation timepoints."""

    def __init__(self, l_roadrunner: LRoadrunner=NULL_L_ROADRUNNER,
            diameter_metric: str = cn.DIAMETER_IVP,
            eigenvalues_collection_arr: np.ndarray = np.array([]),
            eigenvector_collection_arr: np.ndarray = np.array([])) -> None:
        """
        Parameters
        ----------
        l_roadrunner : LRoadrunner
            An LRoadrunner instance used to simulate the model and collect Jacobians.
            Jacobians and timepoints are obtained by calling makeJacobians() on this object.
        diameter_metric : str
            The metric to use for calculating the diameter of the Jacobian collection.
            Options are "weighted_eigenvectors" (default) or "max_cv".
        eigenvalues_collection : List[np.ndarray]
            An optional list of eigenvalue arrays corresponding to the Jacobians, used for diameter calculations.
        eigenvector_collection : List[np.ndarray]
            An optional list of eigenvector arrays corresponding to the Jacobians, used for diameter calculations.
        """
        self.jacobian_collection_arr, self.timepoint_arr = l_roadrunner.makeJacobians()
        self._initialize(l_roadrunner, eigenvalues_collection_arr,
                eigenvector_collection_arr)
        self._sortArrays()
        self._diameter_metric = diameter_metric

    def _initialize(self, l_roadrunner: LRoadrunner,
                eigenvalues_collection_arr: np.ndarray = np.array([]),
                eigenvector_collection_arr: np.ndarray = np.array([])) -> None:
        """Initialize the JacobianCollection with a new LRoadrunner instance."""
        self.l_roadrunner = l_roadrunner
        self.num_species = self.l_roadrunner.num_species
        self.num_point = self.l_roadrunner.num_point
        self._jacobian_mean_arr = np.array([])
        self._jacobian_std_arr = np.array([])
        self._diameter = np.nan
        self._eigenvalues_collection_arr = eigenvalues_collection_arr
        self._eigenvectors_collection_arr = eigenvector_collection_arr
        self._diameter_metric = cn.DIAMETER_IVP
        self.num_jacobian = self.jacobian_collection_arr.shape[0] if hasattr(self, "jacobian_collection_arr") else 0
        if self.num_jacobian == 0:
            raise ValueError("JacobianCollection initialized with no Jacobians.")
        self._cost_mat = NULL_COST*np.ones((self.num_jacobian, self.num_jacobian))  # Placeholder value indicating uninitialized cost matrix

    def getCost(self, istart: int, iend: int) -> float:
        """Get the cost of a segment of Jacobians from index i to j (inclusive).

        The cost is calculated as the diameter of the segment, which is determined
        by the specified diameter_metric. Costs are cached in a cost matrix to
        avoid redundant calculations.

        Parameters
        ----------
        istart : int
            Starting index of the segment
        iend : int
            Ending index of the segment (inclusive).

        Returns
        -------
        float
            The cost of the segment from index i to j.
        """
        iendplus = iend + 1
        if self._cost_mat[istart][iend] == NULL_COST:
            jc = Trajectory.fromArrays(
                    self.jacobian_collection_arr[istart:iendplus],
                    self.timepoint_arr[istart:iendplus],
                    self.l_roadrunner,
                    self.eigenvalues_collection_arr[istart:iendplus],
                    self.eigenvectors_collection_arr[istart:iendplus],
                    )
            self._cost_mat[istart][iend] = jc.diameter
        return self._cost_mat[istart][iend]

    def _makeEigenvaluesAndVectors(self) -> None:
        """Calculate the eigenvalues and eigenvectors for each Jacobian in the collection."""
        eigenvalues_collection:list = []
        eigenvectors_collection: list = []
        for jacobian_arr in self.jacobian_collection_arr:
            eigval, eigvec = np.linalg.eig(jacobian_arr)
            eigenvalues_collection.append(eigval)
            eigenvectors_collection.append(eigvec)
        self._eigenvalues_collection_arr = np.array(eigenvalues_collection)
        self._eigenvectors_collection_arr = np.array(eigenvectors_collection)

    @property
    def eigenvalues_collection_arr(self) -> np.ndarray:
        """Get the of eigenvalues for all Jacobians."""
        if len(self._eigenvalues_collection_arr) == 0 and self.jacobian_collection_arr.size > 0:
            self._makeEigenvaluesAndVectors()
        return self._eigenvalues_collection_arr

    @property
    def eigenvectors_collection_arr(self) -> np.ndarray:
        """Get the list of eigenvectors for all Jacobians."""
        if len(self._eigenvectors_collection_arr) == 0 and self.jacobian_collection_arr.size > 0:
            self._makeEigenvaluesAndVectors()
        return self._eigenvectors_collection_arr

    def _sortArrays(self) -> None:
        """Sort the jacobian_arr and timepoint_arr by timepoint."""
        sort_indices = np.argsort(self.timepoint_arr)
        self.timepoint_arr = self.timepoint_arr[sort_indices]
        self.jacobian_collection_arr = self.jacobian_collection_arr[sort_indices]
    
    @classmethod
    def fromArrays(cls, jacobian_arr: np.ndarray, timepoint_arr: np.ndarray,
            l_roadrunner: LRoadrunner = NULL_L_ROADRUNNER,
            eigenvalues_collection_arr: np.ndarray = np.array([]),
            eigenvector_collection_arr: np.ndarray = np.array([])) -> 'Trajectory':
        """Create a JacobianCollection from explicit arrays."""
        jc = object.__new__(cls)
        jc.jacobian_collection_arr = jacobian_arr.copy()
        jc.timepoint_arr = timepoint_arr.copy()
        jc._initialize(l_roadrunner, eigenvalues_collection_arr, eigenvector_collection_arr)
        jc._sortArrays()
        return jc

    def getTimes(self) -> np.ndarray:
        """Return the sorted unique timepoints in this collection."""
        return np.unique(self.timepoint_arr)
    
    @property
    def jacobian_mean_arr(self) -> np.ndarray:
        """Compute the element-wise mean Jacobian across all timepoints."""
        if self._jacobian_mean_arr.size == 0:
            self._jacobian_mean_arr = np.mean(self.jacobian_collection_arr, axis=0)
        return self._jacobian_mean_arr

    @property
    def max_cv(self) -> float:
        """Compute the maximum coefficient of variation (CV = |std/mean|) across all Jacobian entries."""
        if self.jacobian_collection_arr.size == 0:
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_arr = np.abs(self.jacobian_std_arr / self.jacobian_mean_arr)
            cv_arr[~np.isfinite(cv_arr)] = 0.0
        return np.max(cv_arr)
    
    @staticmethod
    def _ivp(_: float, x: np.ndarray, jacobian_arr: np.ndarray) -> np.ndarray:
        # Calculate the derivative of x with respect to time given the Jacobian and current state x
        return jacobian_arr @ x

    @property
    def ivp_solutions(self) -> np.ndarray:
        """
        Calculate the solution to the linear IVP problem at multiple time points for each Jacobian
            in the collection.
        For each Jacobian, a vector is constructed that is 
            the for multiple time points for each state variable
        Return an array of these vectors.
        """
        if self.jacobian_collection_arr.size == 0:
            return np.array([])
        timepoint_arr = self.l_roadrunner.end_time * IVP_RELATIVE_TIMES
        solutions: list = []
        for jacobian_arr in self.jacobian_collection_arr:
            sol = solve_ivp(self._ivp, (timepoint_arr[0], timepoint_arr[-1]),
                    np.ones(jacobian_arr.shape[0]), t_eval=timepoint_arr,
                    args=(jacobian_arr,))
            solutions.append(np.concatenate(sol.y))
        return np.array(solutions)

    @property
    def diameter(self) -> float:
        if self._diameter_metric == cn.DIAMETER_MAX_CV:
            return self.max_cv
        return self.diameter_ivp

    @property
    def diameter_ivp(self) -> float:
        """
        Represents each Jacobian by a vector of IVP solutions, and calculates the diameter
        of the collection as the maximum distance of any solution vector to the centroid of the solution vectors.
        """
        if not np.isnan(self._diameter):
            return cast(float, self._diameter)
        # Exclude trivial case of empty collection to avoid NaN issues with mean and distance calculations
        if len(self.jacobian_collection_arr) <= 1:
            return 0.0
        # Calculate the solution to the IVP for each Jacobian in the collection
        solution_collection_arr = self.ivp_solutions
        solution_median_arr = np.median(solution_collection_arr, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_solution_collection_arr = np.where(
                    solution_median_arr == 0,
                    0.0,
                    solution_collection_arr / solution_median_arr)
        diameter = 0.0
        middle_arr = np.ones(solution_median_arr.shape)
        for solution_arr in normalized_solution_collection_arr:
            distance = np.linalg.norm(solution_arr - middle_arr)
            diameter = max(diameter, distance)
        self._diameter = diameter
        return cast(float, self._diameter)

    @property
    def jacobian_std_arr(self) -> np.ndarray:
        """Compute the element-wise standard deviation of Jacobians across all timepoints."""
        if self._jacobian_std_arr.size == 0:
            self._jacobian_std_arr = np.std(self.jacobian_collection_arr, axis=0)
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
            diff_arr = np.abs(self.jacobian_collection_arr - self.jacobian_mean_arr)/np.abs(self.jacobian_mean_arr)
        diff_arr[:, np.abs(self.jacobian_mean_arr) == 0] = 0.0
        result = np.sqrt(np.sum(diff_arr**2, axis=(1,2)))
        return result
    

    def predictLinear(self, **kwargs) -> pd.DataFrame:
        """
        Predict the timecourse of species concentrations given an initial state and a timecourse of Jacobians
        Using linear prediction. If no values are given for initial state or forced input,
        these are obtained from LRoadrunner.

        Parameters
        ----------
        initial_state_arr : np.ndarray
            1-D array of shape (num_species,) containing the initial concentrations of each species.
        forcing_input_arr : np.ndarray
            1-D array of shape (num_species,) containing constant forcing inputs for each species.

        Returns
        -------
        np.ndarray
            2-D array of shape (num_timepoints, num_species) containing the predicted concentrations.
        """
        arr = self._predictLinear(**kwargs)
        df = pd.DataFrame(arr, columns=self.l_roadrunner.species_names)
        df = df.set_index(self.timepoint_arr)
        return df

    def _predictLinear(self,
            initial_state_arr: np.ndarray=cn.NULL_ARRAY,
            forcing_input_arr: np.ndarray = cn.NULL_ARRAY) -> np.ndarray:
        """
        Predict the timecourse of species concentrations given an initial state and a timecourse of Jacobians
        Using linear prediction. If no values are given for initial state or forced input,
        these are obtained from LRoadrunner.

        Parameters
        ----------
        initial_state_arr : np.ndarray
            1-D array of shape (num_species,) containing the initial concentrations of each species.
        forcing_input_arr : np.ndarray
            1-D array of shape (num_species,) containing constant forcing inputs for each species.

        Returns
        -------
        np.ndarray
            2-D array of shape (num_timepoints, num_species) containing the predicted concentrations.
        """
        if np.array_equal(initial_state_arr, cn.NULL_ARRAY):
            initial_state_arr = self.l_roadrunner.getInitialValues()
        if np.array_equal(forcing_input_arr, cn.NULL_ARRAY):
            forcing_input_arr = self.fitForcingInputs(initial_state_arr)
        ##
        def ode(t: float, x: np.ndarray) -> np.ndarray:
            return self.jacobian_mean_arr @ x + forcing_input_arr
        ##
        sol = solve_ivp(ode,
                (self.timepoint_arr[0], self.timepoint_arr[-1]),
                initial_state_arr,
                t_eval=self.timepoint_arr,
                method='LSODA')
        return sol.y.T

    def fitForcingInputs(self,
            initial_state_arr: np.ndarray = cn.NULL_ARRAY) -> np.ndarray:
        """
        Find constant forcing inputs that minimise the prediction error of predictLinear
        against the actual nonlinear simulation timecourse.

        Uses lmfit least-squares optimisation to find a forcing input vector u of shape
        (num_species,) such that predictLinear(initial_state_arr, u) is as close as
        possible to l_roadrunner.simulate().

        Parameters
        ----------
        initial_state_arr : np.ndarray
            1-D array of shape (num_species,) of initial concentrations.
            If omitted, l_roadrunner.getInitialValues() is used.

        Returns
        -------
        np.ndarray
            1-D array of shape (num_species,) containing the optimal forcing inputs.
        """
        if np.array_equal(initial_state_arr, cn.NULL_ARRAY):
            initial_state_arr = self.l_roadrunner.getInitialValues()
        actual_arr = self.l_roadrunner.simulate()
        params = Parameters()
        initial_forcing_input_arr = self.l_roadrunner.getForcingInputs()
        for i in range(self.num_species):
            params.add(f'u{i}', value=initial_forcing_input_arr[i])
            #params.add(f'u{i}', value=0)
        ##
        def _residuals(params: Parameters) -> np.ndarray:
            forcing_input_arr = np.array([params[f'u{i}'].value
                    for i in range(self.num_species)])
            predicted_arr = self._predictLinear(initial_state_arr, forcing_input_arr)
            denom = np.abs(actual_arr.flatten())
            numerator = (predicted_arr - actual_arr).flatten()
            numerator = np.where(denom == 0, 0, numerator)
            denom = np.where(denom == 0, 1, denom)
            normalized_residuals = numerator / denom
            return normalized_residuals
        ##
        result = lmfit_minimize(_residuals, params, method='leastsq')
        return np.array([result.params[f'u{i}'].value for i in range(self.num_species)])  # type: ignore

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

    def nonsequentialPartition(self, n_cluster: int, max_iter: int = 300) -> List["Trajectory"]:
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
        n_points = self.jacobian_collection_arr.shape[0]
        if n_cluster > n_points:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_points})."
            )
        flat_arr = self.jacobian_collection_arr.reshape(n_points, -1).astype(float)
        kmeans = KMeans(
            n_clusters=n_cluster, init="k-means++", max_iter=max_iter,
            n_init=1, random_state=0,
        )
        labels_arr = kmeans.fit_predict(flat_arr)
        cluster_indices = [np.where(labels_arr == c)[0] for c in range(n_cluster)]
        return [
            Trajectory.fromArrays(
                self.jacobian_collection_arr[idx], self.timepoint_arr[idx], self.l_roadrunner)
            for idx in cluster_indices
        ]

    def sequentialPartition(self, n_cluster: int) -> List["Trajectory"]:
        """Partition the Jacobians into n_cluster contiguous time segments.
            Uses a greedy heuristic to find a partition into exactly n_cluster contiguous segments
            that reduces the maximum within-segment diameter.

        Parameters
        ----------
        n_cluster : int
            Number of contiguous segments.

        Returns
        -------
        List[JacobianCollection]
            One JacobianCollection per segment, in time order.

        Raises
        ------
        ValueError
            If n_cluster exceeds the number of timepoints.
        """
        # Check for trivial case of 1 cluster to avoid unnecessary cost calculations
        if n_cluster == 1:
            return [self]
        #
        if n_cluster > self.num_jacobian:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({self.num_jacobian})."
            )
        ##
        def split(istart: int, iend: int) -> int:
            """
            The minimum cost split is the maximum cost of the two resulting segments
            The upper end of the interval (RHS) is exclusive, so the last index is iend-1

            Parameters
            ----------
            istart : int
                Starting index of the segment to split (inclusive).
            iend : int
                Ending index of the segment to split (exclusive).

            Returns
            -------
            int: The index at which to split the segment, 
                    where the left segment is [istart, idx) and the right segment is [idx, iend].
            """
            costs = [max(self.getCost(istart, i),  self.getCost(i, iend)) for i in range(istart, iend)]
            idx = np.argmin(costs) + istart
            return cast(int, idx)
        ##
        first_interval = (0, self.num_jacobian-1)  # All indices are inclusive, so the last index is num_jacobian-1
        intervals: List[Tuple[int, int]] = [first_interval]
        interval_costs: List[float] = [self.getCost(*first_interval)]
        for _ in range(n_cluster - 1):
            highest_cost_idx = np.argmax(interval_costs)
            lower_idx = intervals[highest_cost_idx][0]
            upper_idx = intervals[highest_cost_idx][1]
            split_idx = split(lower_idx, upper_idx)
            sub_interval1 = (lower_idx, split_idx)
            sub_interval2 = (split_idx, upper_idx)
            # Update the intervals
            intervals.remove(intervals[highest_cost_idx])
            intervals.append(sub_interval1)
            intervals.append(sub_interval2)
            # Update the costs
            interval_costs.remove(interval_costs[highest_cost_idx])
            interval_costs.append(self.getCost(*sub_interval1))
            interval_costs.append(self.getCost(*sub_interval2))
        # Construct the JacobianCollections for each cluster
        jcs: List[Trajectory] = []
        for istart, iend in intervals:
            if iend == self.num_jacobian - 1:
                iend = self.num_jacobian # Get all of the Jacobians in the last segment
            jcs.append(
                Trajectory.fromArrays(
                self.jacobian_collection_arr[istart:iend],
                self.timepoint_arr[istart:iend],
                self.l_roadrunner,
                self.eigenvalues_collection_arr[istart:iend],
                self.eigenvectors_collection_arr[istart:iend],
                )
            )
        return jcs

    def deprecatedsequentialPartition(self, n_cluster: int) -> List["Trajectory"]:
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
        # Check for trivial case of 1 cluster to avoid unnecessary cost calculations
        if n_cluster == 1:
            return [self]
        #
        n_point = self.jacobian_collection_arr.shape[0]
        if n_cluster > n_point:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_point})."
            )
        cost = np.zeros((n_point, n_point))
        # FIXME: Including timepoint in JacobianCollection.fromArrays() is inefficient since it will be repeated for each cost calculation --- IGNORE ---
        for i in range(n_point):
            for j in range(i, n_point):
                jc = Trajectory.fromArrays(
                        self.jacobian_collection_arr[i:j + 1],
                        self.timepoint_arr[i:j + 1],
                        self.l_roadrunner,
                        self.eigenvalues_collection_arr[i:j],
                        self.eigenvectors_collection_arr[i:j],
                        )
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
            Trajectory.fromArrays(
                self.jacobian_collection_arr[start:end],
                self.timepoint_arr[start:end],
                self.l_roadrunner,
                self.eigenvalues_collection_arr[start:end],
                self.eigenvectors_collection_arr[start:end],
                )
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