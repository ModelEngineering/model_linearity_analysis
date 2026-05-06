'''A collection of trajectories that constitute a larger timecourse.'''

import src.constants as cn
from trajectory import Trajectory  # type: ignore
from src.l_roadrunner import NULL_L_ROADRUNNER  # type: ignore

import matplotlib.cm as mcm  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.gridspec as mgridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from typing import List

# TODO: Implement a plot that shows the timecourse of the simulation in combination
# TODO: predict uses Trajector to predict for each segment. Accumulates the predictions and timepoints.
#   with the clustered Jacobian information

class TrajectoryCollection(object):
    """A collection of trajectories that constitute a larger timecourse."""

    def __init__(self, trajectory_collection: List[Trajectory]) -> None:
        """
        Parameters
        ----------
        trajectory_collections : list[Trajectory]
            List of trajectories for each cluster.
        """
        self.jacobian_collections = trajectory_collection
        if len(trajectory_collection) > 0:
            self.l_roadrunner = trajectory_collection[0].l_roadrunner
        else:
            self.l_roadrunner = NULL_L_ROADRUNNER

    @property
    def max_cv(self) -> float:
        """Compute the maximum CV across all clusters."""
        if len(self.jacobian_collections) == 0:
            return np.nan
        return max(c.max_cv for c in self.jacobian_collections)

    @property
    def score(self) -> float:
        """Return the clustering score, defined as max_cv across all clusters.

        Lower is better: a perfect clustering would have zero within-cluster
        variation (CV = 0).
        """
        return self.max_cv

    def heatmaps(self) -> mfigure.Figure:
        """Plot the CV heatmap for each JacobianCollection in a single row.

        Each subplot title shows the start and end timepoints of its cluster.
        A single shared horizontal colorbar is drawn in a dedicated row below
        the heatmaps so it does not overlap them.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing one heatmap per cluster and a shared colorbar.
        """
        jcs = self.jacobian_collections
        n = len(jcs)

        # Two rows: heatmaps on top, colorbar on bottom (small height ratio).
        fig = plt.figure(figsize=(5 * n, 5))
        gs = mgridspec.GridSpec(2, n, figure=fig, height_ratios=[10, 1], hspace=0.4)

        heatmap_axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
        cbar_ax = fig.add_subplot(gs[1, :])

        for ax, jc in zip(heatmap_axes, jcs):
            t_start = float(jc.timepoint_arr[0])
            t_end = float(jc.timepoint_arr[-1])
            with np.errstate(divide="ignore", invalid="ignore"):
                cv_arr = np.abs(jc.jacobian_std_arr / jc.jacobian_median_arr)
            cv_arr = np.where(np.isfinite(cv_arr), cv_arr, 0.0)
            annot_arr = np.vectorize(lambda v: f"{v:.2g}")(jc.jacobian_median_arr)
            rr = jc.l_roadrunner.getRoadrunner() if hasattr(jc, "l_roadrunner") else None
            species_ids = rr.getFloatingSpeciesIds() if rr is not None else None
            sns.heatmap(
                cv_arr,
                annot=annot_arr,
                fmt="",
                cmap="coolwarm",
                cbar=False,
                xticklabels=species_ids if species_ids is not None else "auto",
                yticklabels=species_ids if species_ids is not None else "auto",
                ax=ax,
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xlabel("Species")
            ax.set_ylabel("Species")
            ax.set_title(f"t={t_start:.3g} to {t_end:.3g}")

        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        sm = mcm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("CV (|std/mean|)")

        return fig