"""Tests and analysis for BIOMD0000000599 with MultipleLinearPredictor / autoSplit.

# Why autoSplit does poorly on BIOMD0000000599

## Model stiffness (the root cause)

The median Jacobian has a maximum absolute eigenvalue of ~47.7 (time^-1).
The characteristic timescale is 1/47.7 ≈ 0.02 time units, but the simulation
step size is 2400/100 = 24 time units — roughly 1143× the characteristic
timescale.

The linear ODE solution x(t+dt) = exp(J * dt) * x(t) amplifies errors by
exp(47.7 * 24) ≈ exp(1144).  Even with num_step=1 (restart from exact observed
state every step), the prediction diverges catastrophically within a single
24-unit step.  Splitting cannot fix this: any segment spans at least one step,
and one step is already 1143× the Jacobian timescale.

## Edge-clustering of splits

autoSplit repeatedly peels off single-point edge segments.  Observed result
for num_split=5: ['0-24', '24-2064', '2064-2328', '2328-2352', '2352-2376',
'2376-2400'] — most splits cluster at the edges of the time range.

This happens because stiffness makes any tiny segment very cheap: predicting
one step from an exact initial condition is easy regardless of nonlinearity.
Both the current 1/max formula AND the alternative sum formula prefer the edge
split for this model:

  k=1  (edge):    1/max(0.18, 2.21) = 0.45;  sum = 2.39
  k=50 (middle):  1/max(1.66, 1.23) = 0.60;  sum = 2.89

Both metrics rank the edge split lower (better), so edge-clustering is driven
by stiffness, not by which formula is used.

## Note on the 1/max formula

The formula c = 1/max(left, right) minimizes 1/max, i.e., maximizes max,
choosing the most unbalanced split.  For most models this would diverge from
the sum formula's answer.  For BIOMD0000000599 both formulas accidentally
agree because stiffness dominates: the edge split is cheapest under every
cost metric.
"""

import unittest
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib  # type: ignore

import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from trajectory import Trajectory  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from src.linear_predictor import LinearPredictor  # type: ignore
from multiple_linear_predictor import MultipleLinearPredictor  # type: ignore

IGNORE_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

MODEL_NAME = "BIOMD0000000599"
NUM_POINT = 101
NUM_SPECIES = 30
# Step size in time units: end_time / (num_point - 1) = 2400 / 100 = 24
STEP_SIZE = 24.0
# Max absolute eigenvalue of median Jacobian
MAX_EIGVAL = 47.7
# Ratio: step_size / (1/max_eigval) — must be >> 1 for stiffness
STIFFNESS_RATIO = MAX_EIGVAL * STEP_SIZE  # ~1145


class TestBiomd599Setup(unittest.TestCase):
    """Verify model and trajectory properties."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model.makeBiomodel(MODEL_NAME)
        cls.trajectory = Trajectory.makeFromSimulation(
                cls.model, start_time=0.0, end_time=None, num_point=NUM_POINT)

    def test_model_name(self) -> None:
        """Model loads with expected name."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.model.model_name, MODEL_NAME)

    def test_num_species(self) -> None:
        """Model has expected number of species."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.model.num_species, NUM_SPECIES)

    def test_end_time(self) -> None:
        """Simulation end_time is 2400 (from SEDML)."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.trajectory.end_time, 2400.0, places=0)

    def test_step_size(self) -> None:
        """Time step between consecutive timepoints is approximately 24 time units."""
        if IGNORE_TESTS:
            return
        dts = np.diff(self.trajectory.timepoint_arr)
        self.assertTrue(np.allclose(dts, STEP_SIZE, rtol=0.01))


class TestBiomd599Stiffness(unittest.TestCase):
    """Verify the stiffness diagnosis: step size >> Jacobian timescale."""

    @classmethod
    def setUpClass(cls) -> None:
        model = Model.makeBiomodel(MODEL_NAME)
        cls.trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=None, num_point=NUM_POINT)
        J = cls.trajectory.jacobian_median_arr
        eigvals = np.linalg.eigvals(J)
        cls.max_eigval = float(np.max(np.abs(eigvals.real)))

    def test_large_max_eigenvalue(self) -> None:
        """Max absolute Jacobian eigenvalue is large (>> 1/step_size)."""
        if IGNORE_TESTS:
            return
        self.assertGreater(self.max_eigval, 10.0)

    def test_stiffness_ratio_exceeds_threshold(self) -> None:
        """max_eigval * step_size >> 1 (step is much longer than Jacobian timescale)."""
        if IGNORE_TESTS:
            return
        ratio = self.max_eigval * STEP_SIZE
        # A ratio > 100 indicates severe stiffness: linear prediction within one
        # step amplifies errors by exp(ratio), making accurate prediction impossible.
        self.assertGreater(ratio, 100.0)

    def test_outlier_species_dominate_cost(self) -> None:
        """A subset of species have catastrophically higher cost than the median."""
        if IGNORE_TESTS:
            return
        lp = LinearPredictor(self.trajectory,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)
        actual_arr = self.trajectory.timecourse_df.values[1:]
        predicted_arr = lp.predict().values[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_arr = np.where(actual_arr == 0, np.nan,
                    (predicted_arr - actual_arr) / actual_arr)
        species_costs = np.nanmean(rel_arr ** 2, axis=0)
        max_cost = float(np.nanmax(species_costs))
        median_cost = float(np.nanmedian(species_costs))
        # At least some species must have cost orders of magnitude above median
        self.assertGreater(max_cost / median_cost, 1000.0)


class TestBiomd599BaselineCost(unittest.TestCase):
    """Verify that single-segment linear prediction is poor."""

    @classmethod
    def setUpClass(cls) -> None:
        model = Model.makeBiomodel(MODEL_NAME)
        cls.trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=None, num_point=NUM_POINT)
        cls.lp = LinearPredictor(cls.trajectory,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)

    def test_baseline_cost_is_high(self) -> None:
        """Single-segment cost is well above 1.0 (median species ARE² > 100%)."""
        if IGNORE_TESTS:
            return
        self.assertGreater(self.lp.cost, 1.0)

    def test_predict_returns_correct_shape(self) -> None:
        """predict() returns a DataFrame with NUM_POINT rows and NUM_SPECIES columns."""
        if IGNORE_TESTS:
            return
        pred_df = self.lp.predict()
        self.assertEqual(pred_df.shape, (NUM_POINT, NUM_SPECIES))


class TestBiomd599AutoSplit(unittest.TestCase):
    """Verify that autoSplit provides minimal improvement and clusters splits at edges.

    The inverted cost formula (1/max rather than sum) causes the greedy
    algorithm to prefer edge splits: peeling off a single-point segment
    (cost ≈ 0) maximizes max(left, right), minimizing 1/max(...) and thus
    being ranked as the "best" split at every iteration.
    """

    @classmethod
    def setUpClass(cls) -> None:
        model = Model.makeBiomodel(MODEL_NAME)
        cls.trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=None, num_point=NUM_POINT)
        cls.baseline_lp = LinearPredictor(cls.trajectory,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)
        cls.baseline_cost = cls.baseline_lp.cost

        cls.tc_5 = TrajectoryCollection.autoSplit(cls.trajectory, num_split=5)
        cls.mlp_5 = MultipleLinearPredictor(cls.tc_5,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)

    def test_baseline_cost_high(self) -> None:
        """Confirms baseline cost exceeds 1.0 before splitting."""
        if IGNORE_TESTS:
            return
        self.assertGreater(self.baseline_cost, 1.0)

    def test_autosplit_5_still_poor(self) -> None:
        """After 5 splits, cost remains above 1.0 (stiffness cannot be split away)."""
        if IGNORE_TESTS:
            return
        self.assertGreater(self.mlp_5.cost, 1.0)

    def test_autosplit_5_improvement_is_small(self) -> None:
        """5 splits reduce cost by less than 40% (from ~1.78 to ~1.25)."""
        if IGNORE_TESTS:
            return
        improvement = (self.baseline_cost - self.mlp_5.cost) / self.baseline_cost
        self.assertLess(improvement, 0.40)

    def test_splits_cluster_at_edges(self) -> None:
        """autoSplit(5) produces at least 2 segments spanning only the first or last step.

        The inverted 1/max formula rewards the most unbalanced splits, so the
        algorithm repeatedly peels single-point segments from the edges rather
        than cutting through the interior.
        """
        if IGNORE_TESTS:
            return
        trajs = self.tc_5.trajectories
        # An edge segment spans exactly one simulation step (≈ STEP_SIZE)
        edge_segs = [t for t in trajs
                if np.isclose(t.end_time - t.start_time, STEP_SIZE, rtol=0.01)]
        self.assertGreaterEqual(len(edge_segs), 2)

    def test_both_formulas_prefer_edge_split(self) -> None:
        """Both the 1/max and sum cost formulas rank the edge split over the midpoint split.

        For this stiff model, stiffness — not formula choice — drives edge-clustering.
        A single-step edge segment is cheap under every metric because we restart from
        the exact observed state and predict only one step forward.
        """
        if IGNORE_TESTS:
            return
        tp_arr = self.trajectory.timepoint_arr

        def _segcost(i: int, j: int) -> float:
            sub = self.trajectory.makeSubmodel(float(tp_arr[i]), float(tp_arr[j]))
            return LinearPredictor(sub, jacobian_selection=cn.JAC_MEDIAN,
                    num_step=-1).cost

        edge_left = _segcost(0, 1)
        edge_right = _segcost(1, len(tp_arr) - 1)
        mid = len(tp_arr) // 2
        mid_left = _segcost(0, mid)
        mid_right = _segcost(mid, len(tp_arr) - 1)

        # 1/max formula: lower = better
        self.assertLess(1.0 / max(edge_left, edge_right),
                1.0 / max(mid_left, mid_right),
                msg="1/max formula prefers edge split for this stiff model")

        # sum formula: lower = better
        self.assertLess(edge_left + edge_right, mid_left + mid_right,
                msg="sum formula also prefers edge split — stiffness drives clustering")


if __name__ == "__main__":
    unittest.main()
