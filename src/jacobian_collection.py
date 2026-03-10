'''Container of a collection of Jacobian matrices and their timepoints and utilities.'''
import src.constants as cn
from roadrunner_maker import RoadRunnerMaker  # type: ignore

import collections
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, Set


PlotInfo = collections.namedtuple("PlotInfo",
        ["top_ax", "bottom_ax", "fig"])

class JacobianCollection(object):
    """A collection of Jacobian matrices over one or more simulation timepoints."""

    def __init__(self, jacobian_arr: np.ndarray, timepoints: np.ndarray) -> None:
        """
        Parameters
        ----------
        jacobian_arr : np.ndarray
            Array of shape (num_point, n_species, n_species) containing Jacobian matrices.
        timepoints : np.ndarray
            Array of timepoints corresponding to each Jacobian matrix in jacobian_arr.
        """
        sort_indices = np.argsort(timepoints)
        self.timepoints = timepoints[sort_indices]
        self.jacobian_arr = jacobian_arr[sort_indices]

    def getTimes(self) -> Set[float]:
        """Return the unique set of timepoints in this collection."""
        return set(self.timepoints)

    @property
    def max_cv(self) -> float:
        """Compute the maximum coefficient of variation (CV = |std/mean|) across all Jacobian entries."""
        if self.jacobian_arr.size == 0:
            return 0.0
        mean_arr = np.mean(self.jacobian_arr, axis=0)
        std_arr = np.std(self.jacobian_arr, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_arr = np.abs(std_arr / mean_arr)
            cv_arr[~np.isfinite(cv_arr)] = 0.0
        return np.max(cv_arr)

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
        centroid_arr = np.mean(self.jacobian_arr, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            diff_arr = np.abs(self.jacobian_arr - centroid_arr)/np.abs(centroid_arr)
        diff_arr[:, np.abs(centroid_arr) == 0] = 0.0
        result = np.sqrt(np.sum(diff_arr**2, axis=(1,2)))
        return result

    def plot(self, roadrunner_maker: RoadRunnerMaker,
            top_ax: Optional[plt.Axes] = None,   # type: ignore
            bottom_ax: Optional[plt.Axes] = None, # type: ignore
            fig: Optional[plt.Figure] = None  # type: ignore
            ) -> PlotInfo:  
        """
        Constructs a figure with two plots with time on the x-axis: (1) the Frobenius-norm distance of each Jacobian from the centroid, and
        (2) the timecourse of simulation species concentrations.
        The first plot shows how the Jacobian changes over time relative to the centroid.
        The second plot shows the dynamics of the model's species concentrations
        over time.

        Parameters
        ----------
        roadrunner_maker : RoadRunnerMaker
            A RoadRunnerMaker instance containing the model to simulate for the second plot.
        top_ax : Optional[plt.Axes]
            An optional matplotlib Axes object to use for the top plot. If None, a new figure and axes will be created.
        bottom_ax : Optional[plt.Axes]
            An optional matplotlib Axes object to use for the bottom plot. If None, a new figure and axes will be created.
        fig : Optional[plt.Figure]
            An optional matplotlib Figure object to use. If None, a new figure will be created.
        """

        rr = roadrunner_maker.roadrunner
        rr.reset()
        species_ids = rr.getFloatingSpeciesIds()
        rows = []
        for i, t in enumerate(self.timepoints):
            if i == 0:
                rr.simulate(self.timepoints[0], t + 1e-10, 2)
            else:
                rr.simulate(self.timepoints[i - 1], t, 2)
            rows.append(list(rr.getFloatingSpeciesConcentrations()))
        species_data = np.array(rows)

        times = np.array(sorted(self.getTimes()))
        deviation_arr = self._calculateDeviation()

        if top_ax is None or bottom_ax is None or fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
        else:
            ax1 = top_ax
            ax2 = bottom_ax

        ax1.plot(times, deviation_arr)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Normalized distance")
        ax1.set_title("Normalized Distance of Jacobian to Centroid")

        for i, species_id in enumerate(species_ids):
            ax2.plot(self.timepoints, species_data[:, i], label=species_id)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Concentration")
        ax2.set_title("Species Timecourse")
        ax2.legend()

        fig.tight_layout()
        plt.show()
        return PlotInfo(top_ax=ax1, bottom_ax=ax2, fig=fig)  