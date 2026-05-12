"""Tests for TrajectoryCollection."""
import unittest

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.linear_predictor import LinearPredictor  # type: ignore
from model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from trajectory import Trajectory  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore

IGNORE_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_SPECIES = 2


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL)


def _makeTrajectoryWithRange(start: float, end: float,
        num_point: int = 11, seed: int = -1) -> Trajectory:
    """Build a synthetic Trajectory covering [start, end]."""
    actual_seed = seed if seed >= 0 else int(start * 100 + end)
    rng = np.random.default_rng(actual_seed)
    tp = np.linspace(start, end, num_point)
    jc = rng.standard_normal((num_point, NUM_SPECIES, NUM_SPECIES))
    fi = rng.standard_normal((num_point, NUM_SPECIES))
    tc = pd.DataFrame(
            rng.standard_normal((num_point, NUM_SPECIES)),
            index=tp,
            columns=["S1", "S2"],
    )
    tc.index.name = "time"
    return Trajectory(
            model=_makeModel(),
            jacobian_collection_arr=jc,
            timepoint_arr=tp,
            forcing_input_collection_arr=fi,
            timecourse_df=tc,
    )


def _makeTwoNonOverlapping():
    """Return two non-overlapping trajectories: [0,5] and [6,10]."""
    return _makeTrajectoryWithRange(0.0, 5.0), _makeTrajectoryWithRange(6.0, 10.0)


def _makeCollection() -> TrajectoryCollection:
    t1, t2 = _makeTwoNonOverlapping()
    return TrajectoryCollection([t1, t2])


def _makeWideTrajectory(num_point: int = 21) -> Trajectory:
    """Return a synthetic Trajectory covering [0, 10] suitable for split()."""
    return _makeTrajectoryWithRange(0.0, 10.0, num_point)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestTrajectoryCollectionInit(unittest.TestCase):

    def test_constructs_with_valid_args(self) -> None:
        """Constructor accepts a list of non-overlapping same-model trajectories."""
        if IGNORE_TESTS:
            return
        tc = _makeCollection()
        self.assertIsInstance(tc, TrajectoryCollection)

    def test_single_trajectory(self) -> None:
        """Constructor accepts a single-element list."""
        if IGNORE_TESTS:
            return
        traj = _makeTrajectoryWithRange(0.0, 10.0)
        tc = TrajectoryCollection([traj])
        self.assertEqual(len(tc.trajectories), 1)

    def test_empty_list_raises(self) -> None:
        """Empty list raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            TrajectoryCollection([])

    def test_different_models_raises(self) -> None:
        """Trajectories with different Model instances raise ValueError."""
        if IGNORE_TESTS:
            return
        t1 = _makeTrajectoryWithRange(0.0, 5.0)
        t2 = _makeTrajectoryWithRange(6.0, 10.0)
        # Replace t2's model with a fresh, distinct Model object
        t2_modified = Trajectory(
                model=_makeModel(),
                jacobian_collection_arr=t2.jacobian_collection_arr,
                timepoint_arr=t2.timepoint_arr + 100,  # distinct time range too
                forcing_input_collection_arr=t2.forcing_input_collection_arr,
                timecourse_df=t2.timecourse_df,
        )
        # Both use the same ANTIMONY_MODEL so Model.__eq__ would say they're equal;
        # use a different antimony string to force inequality.
        different_antimony = """
S1 -> S2; k1*S1
k1 = 0.5; S1 = 1; S2 = 0
"""
        different_model = Model(different_antimony)
        t2_diff_model = Trajectory(
                model=different_model,
                jacobian_collection_arr=t2.jacobian_collection_arr,
                timepoint_arr=t2.timepoint_arr,
                forcing_input_collection_arr=t2.forcing_input_collection_arr,
                timecourse_df=t2.timecourse_df,
        )
        with self.assertRaises(ValueError):
            TrajectoryCollection([t1, t2_diff_model])

    def test_isConsecutive(self) -> None:
        if IGNORE_TESTS:
            return
        t1 = _makeTrajectoryWithRange(0.0, 7.0)
        t2 = _makeTrajectoryWithRange(5.0, 10.0)
        self.assertFalse(TrajectoryCollection([t1, t2]).isConsecutive())
        #
        t2 = _makeTrajectoryWithRange(7.0, 10.0)
        self.assertTrue(TrajectoryCollection([t1, t2]).isConsecutive())

    def test_trajectories_are_sorted(self) -> None:
        """Constructor sorts trajectories by time order regardless of input order."""
        if IGNORE_TESTS:
            return
        t1 = _makeTrajectoryWithRange(0.0, 5.0)
        t2 = _makeTrajectoryWithRange(6.0, 10.0)
        tc = TrajectoryCollection([t2, t1])  # reversed input
        self.assertLess(tc.trajectories[0].start_time, tc.trajectories[1].start_time)

    def test_adjacent_shared_boundary_allowed(self) -> None:
        """Trajectories whose end_time == next start_time are accepted."""
        if IGNORE_TESTS:
            return
        t1 = _makeTrajectoryWithRange(0.0, 5.0)
        t2 = _makeTrajectoryWithRange(5.0, 10.0)
        tc = TrajectoryCollection([t1, t2])
        self.assertEqual(len(tc.trajectories), 2)


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

class TestTrajectoryCollectionEq(unittest.TestCase):

    def test_equal_collections(self) -> None:
        """Two collections built from identical trajectories are equal."""
        if IGNORE_TESTS:
            return
        t1, t2 = _makeTwoNonOverlapping()
        tc1 = TrajectoryCollection([t1, t2])
        tc2 = TrajectoryCollection([t1, t2])
        self.assertEqual(tc1, tc2)

    def test_different_number_of_trajectories_not_equal(self) -> None:
        """Collections with different lengths are not equal."""
        if IGNORE_TESTS:
            return
        t1, t2 = _makeTwoNonOverlapping()
        tc1 = TrajectoryCollection([t1, t2])
        tc2 = TrajectoryCollection([t1])
        self.assertNotEqual(tc1, tc2)

    def test_different_trajectories_not_equal(self) -> None:
        """Collections with different trajectories are not equal."""
        if IGNORE_TESTS:
            return
        t1, t2 = _makeTwoNonOverlapping()
        t3 = _makeTrajectoryWithRange(6.0, 10.0, seed=999)  # same range, different data
        tc1 = TrajectoryCollection([t1, t2])
        tc2 = TrajectoryCollection([t1, t3])
        self.assertNotEqual(tc1, tc2)

    def test_not_equal_to_non_collection(self) -> None:
        """Comparing with a non-TrajectoryCollection returns NotImplemented."""
        if IGNORE_TESTS:
            return
        tc = _makeCollection()
        with self.assertRaises(ValueError):
            _ = tc.__eq__("not a collection")


# ---------------------------------------------------------------------------
# plotTimecourse
# ---------------------------------------------------------------------------

class TestTrajectoryCollectionPlotTimecourse(unittest.TestCase):

    def test_returns_plot_options(self) -> None:
        """plotTimecourse returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        tc = _makeCollection()
        result = tc.plotTimecourse()
        self.assertIsInstance(result, PlotOptions)

    def test_vertical_lines_count(self) -> None:
        """One vertical separator line is drawn between each pair of trajectories."""
        if IGNORE_TESTS:
            return
        t1, t2 = _makeTwoNonOverlapping()
        t3 = _makeTrajectoryWithRange(12.0, 20.0)
        tc = TrajectoryCollection([t1, t2, t3])
        plot_options = tc.plotTimecourse()
        ax = plot_options.ax
        # axvline adds a Line2D; count lines whose xdata is a scalar (vertical)
        vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]  # type: ignore
        self.assertEqual(len(vlines), 2)

    def test_line_count_matches_species(self) -> None:
        """Number of non-vertical lines equals num_species * num_trajectories."""
        if IGNORE_TESTS:
            return
        t1, t2 = _makeTwoNonOverlapping()
        tc = TrajectoryCollection([t1, t2])
        plot_options = tc.plotTimecourse()
        ax = plot_options.ax
        non_vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]  # type: ignore
        self.assertEqual(len(non_vlines), NUM_SPECIES * 2)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

class TestTrajectoryCollectionSplit(unittest.TestCase):

    def setUp(self) -> None:
        self.trajectory = _makeWideTrajectory(num_point=21)  # [0, 10], 21 pts

    def test_returns_trajectory_collection(self) -> None:
        """split returns a TrajectoryCollection."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        self.assertIsInstance(tc, TrajectoryCollection)

    def test_single_split_gives_two_trajectories(self) -> None:
        """One split time produces two sub-trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        self.assertEqual(len(tc.trajectories), 2)

    def test_two_splits_give_three_trajectories(self) -> None:
        """Two split times produce three sub-trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [3.0, 7.0])
        self.assertEqual(len(tc.trajectories), 3)

    def test_empty_timepoints_gives_one_trajectory(self) -> None:
        """No split times returns a single-trajectory collection."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [])
        self.assertEqual(len(tc.trajectories), 1)

    def test_split_covers_full_time_range(self) -> None:
        """Sub-trajectories together span the original start and end times."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [4.0, 7.0])
        self.assertAlmostEqual(tc.trajectories[0].start_time,
                self.trajectory.start_time)
        self.assertAlmostEqual(tc.trajectories[-1].end_time,
                self.trajectory.end_time)

    def test_snap_to_nearest_timepoint(self) -> None:
        """A split time between grid points snaps to the nearest timepoint."""
        if IGNORE_TESTS:
            return
        # timepoint_arr has points at 0, 0.5, 1.0, ... (step 0.5 for 21 pts over [0,10])
        # Split at 4.9 — nearest grid point is 5.0
        tc = TrajectoryCollection.split(self.trajectory, [4.9])
        boundary = tc.trajectories[0].end_time
        self.assertIn(boundary, self.trajectory.timepoint_arr)

    def test_boundary_timepoint_shared(self) -> None:
        """The end_time of the first sub-trajectory equals the start_time of the second."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        self.assertAlmostEqual(
                tc.trajectories[0].end_time,
                tc.trajectories[1].start_time,
        )

    def test_split_time_at_start_ignored(self) -> None:
        """A split time that snaps to start_time does not create an empty sub-trajectory."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [0.0])
        self.assertEqual(len(tc.trajectories), 1)

    def test_split_time_at_end_ignored(self) -> None:
        """A split time that snaps to end_time does not create an empty sub-trajectory."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [10.0])
        self.assertEqual(len(tc.trajectories), 1)

    def test_duplicate_split_times_collapsed(self) -> None:
        """Duplicate split times that snap to the same point give one split."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0, 5.0])
        self.assertEqual(len(tc.trajectories), 2)

    def test_sub_trajectory_timepoints_in_range(self) -> None:
        """Each sub-trajectory only contains timepoints within its own range."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        for traj in tc.trajectories:
            self.assertTrue(np.all(traj.timepoint_arr >= traj.start_time))
            self.assertTrue(np.all(traj.timepoint_arr <= traj.end_time))

    def test_all_trajectories_share_same_model(self) -> None:
        """All sub-trajectories reference the original trajectory's model."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [3.0, 7.0])
        for traj in tc.trajectories:
            self.assertEqual(traj.model, self.trajectory.model)



# ---------------------------------------------------------------------------
# autoSplit
# ---------------------------------------------------------------------------

class TestTrajectoryCollectionAutoSplit(unittest.TestCase):

    def setUp(self) -> None:
        self.trajectory = _makeWideTrajectory(num_point=11)  # [0, 10], 11 pts

    def test_returns_trajectory_collection(self) -> None:
        """autoSplit returns a TrajectoryCollection."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=1)
        self.assertIsInstance(tc, TrajectoryCollection)

    def test_zero_splits_gives_one_trajectory(self) -> None:
        """num_split=0 returns a single-trajectory collection."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=0)
        self.assertEqual(len(tc.trajectories), 1)

    def test_one_split_gives_two_trajectories(self) -> None:
        """num_split=1 returns two sub-trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=1)
        self.assertEqual(len(tc.trajectories), 2)

    def test_two_splits_give_three_trajectories(self) -> None:
        """num_split=2 returns three sub-trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=2)
        self.assertEqual(len(tc.trajectories), 3)

    def test_split_reduces_cost(self) -> None:
        """autoSplit with num_split=1 finds a split no worse than splitting at the midpoint."""
        if IGNORE_TESTS:
            return
        model = _makeModel()
        trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=10.0, num_point=11)
        jac_sel = cn.JAC_MEDIAN
        num_step = 1
        tc_auto = TrajectoryCollection.autoSplit(
                trajectory, num_split=1)
        auto_cost = sum(
                LinearPredictor(t, jacobian_selection=jac_sel, num_step=num_step).cost
                for t in tc_auto.trajectories)
        mid_idx = len(trajectory.timepoint_arr) // 2
        mid_time = float(trajectory.timepoint_arr[mid_idx])
        tc_mid = TrajectoryCollection.split(trajectory, [mid_time])
        mid_cost = sum(
                LinearPredictor(t, jacobian_selection=jac_sel, num_step=num_step).cost
                for t in tc_mid.trajectories)
        self.assertLessEqual(auto_cost, mid_cost + 1e-9)

    def test_covers_full_time_range(self) -> None:
        """autoSplit result spans the original start and end times."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=2)
        self.assertAlmostEqual(tc.trajectories[0].start_time,
                self.trajectory.start_time)
        self.assertAlmostEqual(tc.trajectories[-1].end_time,
                self.trajectory.end_time)

    def test_split_times_are_valid_timepoints(self) -> None:
        """Every boundary between sub-trajectories is in the original timepoint_arr."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=2)
        for traj in tc.trajectories:
            self.assertIn(traj.start_time, self.trajectory.timepoint_arr)
            self.assertIn(traj.end_time, self.trajectory.timepoint_arr)

    def test_excessive_splits_clamped(self) -> None:
        """num_split larger than n-2 is clamped to n-2."""
        if IGNORE_TESTS:
            return
        n = len(self.trajectory.timepoint_arr)  # 11
        tc = TrajectoryCollection.autoSplit(self.trajectory, num_split=100)
        self.assertEqual(len(tc.trajectories), n - 1)  # at most n-1 segments

    def test_finds_optimal_single_split(self) -> None:
        """autoSplit result has cost <= every explicit single split."""
        if IGNORE_TESTS:
            return
        from src.linear_predictor import LinearPredictor

        model = _makeModel()
        trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=10.0, num_point=11)

        tc_auto = TrajectoryCollection.autoSplit(trajectory, num_split=1)
        auto_cost = sum(
                LinearPredictor(t, num_step=1).cost
                for t in tc_auto.trajectories)

        tp_arr = trajectory.timepoint_arr
        for idx in range(1, len(tp_arr) - 1):
            split_time = float(tp_arr[idx])
            tc_manual = TrajectoryCollection.split(trajectory, [split_time])
            manual_cost = sum(
                    LinearPredictor(t, num_step=1).cost
                    for t in tc_manual.trajectories)
            self.assertLessEqual(auto_cost, manual_cost + 1e-9)


if __name__ == "__main__":
    unittest.main()
