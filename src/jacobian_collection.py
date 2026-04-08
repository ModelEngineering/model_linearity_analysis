'''Container of a collection of Jacobian matrices and their timepoints and utilities.'''
import src.constants as cn
from src.l_roadrunner import LRoadrunner, NULL_L_ROADRUNNER  # type: ignore

import collections
import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from typing import Optional, Tuple, cast



PlotInfo = collections.namedtuple("PlotInfo",
        ["top_ax", "bottom_ax", "fig"])

class JacobianCollection(object):
    """A collection of Jacobian matrices over one or more simulation timepoints."""

    def __init__(self, l_roadrunner: LRoadrunner=NULL_L_ROADRUNNER) -> None:
        """
        Parameters
        ----------
        l_roadrunner : LRoadrunner
            An LRoadrunner instance used to simulate the model and collect Jacobians.
            Jacobians and timepoints are obtained by calling makeJacobians() on this object.
        """
        self._initialize(l_roadrunner)
        try:
            self.jacobian_arr, self.timepoint_arr = l_roadrunner.makeJacobians()
            self._sortArrays()
        except Exception as e:
            raise ValueError(f"Failed to create JacobianCollection for {l_roadrunner.specification[0:200]}") from e
        
    def _initialize(self, l_roadrunner: LRoadrunner) -> None:
        """Initialize the JacobianCollection with a new LRoadrunner instance."""
        self.l_roadrunner = l_roadrunner
        self._jacobian_mean_arr = np.array([])
        self._jacobian_std_arr = np.array([])
        self._max_expo_eigen_distance = np.nan
    
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
    
    @property
    def max_eigen_expo_distance(self) -> float:
        """
        Compute the maximum distance of the eigenvalues of each Jacobian 
        from the eigenvalues of the mean Jacobian, where distance is defined as the L2 norm of the difference 
        of their exponentials.
        """
        if not np.isnan(self._max_expo_eigen_distance):
            return cast(float, self._max_expo_eigen_distance)
        if self.jacobian_arr.size == 0:
            return 0.0
        mean_eigvals = np.linalg.eigvals(self.jacobian_mean_arr)
        max_distance = 0.0
        for jacobian_arr in self.jacobian_arr:
            eigvals = np.linalg.eigvals(jacobian_arr)
            distance = np.linalg.norm(np.exp(eigvals) - np.exp(mean_eigvals))
            max_distance = max(max_distance, distance)
        self._max_expo_eigen_distance = max_distance
        return cast(float, self._max_expo_eigen_distance)

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

        if hasattr(self, 'l_roadrunner'):
            roadrunner = self.l_roadrunner.getRoadrunner()
            species_ids = roadrunner.getFloatingSpeciesIds()
            data_arr = self.l_roadrunner.simulate(is_with_timepoints=True)
            species_data = data_arr[:, 1:]  # Exclude time column
            species_times = data_arr[:, 0]  # Extract time column
        else:
            raise ValueError("Cannot plot species timecourse without an LRoadrunner instance.") 
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