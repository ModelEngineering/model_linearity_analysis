"""Collection of Trajectory objects sharing the same Model."""

import src.constants as cn  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.trajectory import Trajectory  # type: ignore

import numpy as np  # type: ignore
from typing import List


class TrajectoryCollection(object):
    """An ordered, non-overlapping collection of Trajectory sharing one Model.

    Use split() to partition a single Trajectory into a TrajectoryCollection.
    """

    def __init__(self, trajectories: List[Trajectory]) -> None:
        """
        Parameters
        ----------
        trajectories : List[Trajectory]
            All must share the same Model. Sorted by Trajectory.__lt__;
            overlapping time ranges raise ValueError.
        """
        # Error checking
        if not trajectories:
            raise ValueError("trajectories must not be empty.")
        self.trajectories = sorted(trajectories)
        self.model = self.trajectories[0].model
        for traj in trajectories[1:]:
            if traj.model != self.model:
                raise ValueError(
                        "All trajectories must share the same Model.")
        self.start_time = self.trajectories[0].start_time
        self.end_time = self.trajectories[-1].end_time

    def isConsecutive(self) -> bool:
        """True iff trajectories are consecutive (no gaps or overlaps)."""
        for t1, t2 in zip(self.trajectories, self.trajectories[1:]):
            if not np.isclose(t1.end_time, t2.start_time):
                return False
        return True 

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryCollection):
            raise ValueError("Can only compare TrajectoryCollection to another ")
        if len(self.trajectories) != len(other.trajectories):
            return False
        return all(t1 == t2 for t1, t2 in zip(self.trajectories, other.trajectories))

    def plotTimecourse(self, **kwargs) -> PlotOptions:
        """Plot the pieced-together timecourse with dashed vertical separators.

        Each trajectory's timecourse is drawn continuously; a vertical dashed
        line marks the boundary between adjacent trajectories.  One legend
        entry per species.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions. Supported keys: ax, fig, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.

        Returns
        -------
        PlotOptions
        """
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax
        for i, name in enumerate(self.model.species_names):
            color = f"C{i}"
            for j, traj in enumerate(self.trajectories):
                label = name if j == 0 else None
                ax.plot(
                        traj.timecourse_df.index,
                        traj.timecourse_df[name],
                        color=color,
                        label=label,
                )
        for traj in self.trajectories[:-1]:
            ax.axvline(x=traj.end_time, color="black", linestyle="--",
                    linewidth=0.8)
        plot_options.apply()
        return plot_options

    @classmethod
    def split(cls,
            trajectory: Trajectory,
            timepoints: List[float]) -> "TrajectoryCollection":
        """Partition a Trajectory at the given split times.

        Each split time becomes the shared end/start boundary between adjacent
        sub-trajectories.  Times not exactly in timepoint_arr snap to the
        nearest available timepoint.  Split times that snap to start_time or
        end_time are ignored.

        Parameters
        ----------
        trajectory : Trajectory
        timepoints : List[float]

        Returns
        -------
        TrajectoryCollection
        """
        tp_arr = trajectory.timepoint_arr
        snapped = []
        for t in timepoints:
            idx = int(np.argmin(np.abs(tp_arr - t)))
            snapped.append(float(tp_arr[idx]))
        snapped = sorted({
                t for t in snapped
                if trajectory.start_time < t < trajectory.end_time
        })
        boundaries = [trajectory.start_time] + snapped + [trajectory.end_time]
        sub_trajectories = [
                trajectory.makeSubmodel(boundaries[i], boundaries[i + 1])
                for i in range(len(boundaries) - 1)
        ]
        return cls(sub_trajectories)
