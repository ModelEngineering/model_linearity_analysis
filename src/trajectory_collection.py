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
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from typing import List
import src.utils as utils  # type: ignore

# TODO: Implement a plot that shows the timecourse of the simulation in combination
# TODO: predict uses Trajector to predict for each segment. Accumulates the predictions and timepoints.
#   with the clustered Jacobian information

class TrajectoryCollection(object):
    """A collection of trajectories that constitute a larger timecourse."""

    def __init__(self, trajectories: List[Trajectory]) -> None:
        """
        Parameters
        ----------
        trajectory_collections : list[Trajectory]
            List of trajectories for each cluster.
        """
        self.trajectories = trajectories
        if len(trajectories) > 0:
            self.l_roadrunner = trajectories[0].l_roadrunner
        else:
            self.l_roadrunner = NULL_L_ROADRUNNER

    @classmethod
    def split(cls,
                trajectory: Trajectory,
                split_times: List[float],
                **kwargs
            ) -> 'TrajectoryCollection':
        """Creates a trajectory collection by splitting a single trajectory
            at specified timepoints.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to be split into a collection.
        split_times : list[float]
            A list of timepoints at which to split the trajectory.
                Must be within the bounds of the trajectory's timecourse.        
        **kwargs
            Additional keyword arguments to pass to the Trajectory.fromArrays method
        """
        split_times = sorted(split_times)
        start_times = [float(trajectory.timepoint_arr[0])] + list(split_times)
        end_times = list(split_times) + [float(trajectory.timepoint_arr[-1])]
        trajectories: List[Trajectory] = []
        for start_time, end_time in zip(start_times, end_times):
            if start_time < trajectory.timepoint_arr[0] or start_time > trajectory.timepoint_arr[-1]:
                raise ValueError(f"Split time {start_time} is out of bounds for the trajectory timecourse.")
            if end_time < trajectory.timepoint_arr[0] or end_time > trajectory.timepoint_arr[-1]:
                raise ValueError(f"Split time {end_time} is out of bounds for the trajectory timecourse.")
            istart = utils.findFloatIndex(trajectory.timepoint_arr, start_time)
            iend = utils.findFloatIndex(trajectory.timepoint_arr, end_time)
            jacobian_arr = trajectory.jacobian_collection_arr[istart:iend + 1]  # type: ignore
            timepoint_arr = trajectory.timepoint_arr[istart:iend + 1]  # type: ignore
            trajectories.append(
                    Trajectory.fromArrays(jacobian_arr, timepoint_arr,
                    trajectory.l_roadrunner, **kwargs))
        trajectory_collection = TrajectoryCollection(trajectories)
        return trajectory_collection
    
    @staticmethod
    def _findFloatIndex(arr: np.ndarray, value: float) -> int:
        """Find the index of a float value in an array, allowing for a small tolerance.

        Parameters
        ----------
        arr : np.ndarray
            The array to search.
        value : float
            The value to find.

        Returns
        -------
        int
            The index of the value in the array.

        Raises
        ------
        ValueError
            If the value is not found within the tolerance.
        """
        idx = np.where(np.isclose(arr, value, atol=1e-8))[0]
        if len(idx) == 0:
            raise ValueError(f"Value {value} not found in array within tolerance.")
        return idx[0]

    def predict(self)-> pd.DataFrame:
        """Predicts each trajectory and concatenate the results.

        Returns
        -------
        pd.DataFrame
            The predicted trajectory for the entire timecourse, concatenated from each cluster's prediction.
        """
        predicted_trajectories = []
        for idx, trajectory in enumerate(self.trajectories):
            pred_df = trajectory.predict()
            if idx > 0:
                pred_df = pred_df.iloc[1:]  # Remove the first timepoint to avoid duplication
            predicted_trajectories.append(pred_df)
        return pd.concat(predicted_trajectories, axis=0)   

    @property
    def max_cv(self) -> float:
        """Compute the maximum CV across all clusters."""
        if len(self.trajectories) == 0:
            return np.nan
        return max(c.max_cv for c in self.trajectories)

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
        jcs = self.trajectories
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