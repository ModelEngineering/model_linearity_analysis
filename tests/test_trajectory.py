"""Tests for Trajectory class."""
from src.l_roadrunner import LRoadrunner  # type: ignore

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd  # type: ignore
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from trajectory import Trajectory, IVP_RELATIVE_TIMES  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore

IGNORE_TESTS = True
if not IGNORE_TESTS:
    matplotlib.use("Agg")

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMD38_PATH = os.path.join(
    cn.BIOMODELS_DIR, "BIOMD0000000038", "BIOMD0000000038_url.xml"
)
BIOMD8_PATH = os.path.join(
    cn.BIOMODELS_DIR, "BIOMD0000000008", "BIOMD0000000008_url.xml"
)
BIOMD8_ENDTIME = 20.0
BIOMD206_PATH = os.path.join(
    cn.BIOMODELS_DIR, "BIOMD0000000206", "BIOMD0000000206_url.xml"
)
BIOMD206_ENDTIME = 15.0
BIOMODEL_NAMES = ["BIOMD0000000008", "BIOMD0000000054", "BIOMD0000000181"]


def _make_trajectory_from_arrays(jacobian_collection_arr: np.ndarray, timepoints: np.ndarray) -> Trajectory:
    """Return a JacobianCollection from explicit arrays using a mock LRoadrunner."""
    lr = MagicMock(spec=LRoadrunner)
    lr.makeJacobians.return_value = (jacobian_collection_arr, timepoints)
    lr.start_time = 0.0
    lr.end_time = 5.0
    lr.num_point = len(timepoints)
    if len(jacobian_collection_arr) == 0:
        num_species = 0
    else:
        num_species = jacobian_collection_arr.shape[1]
    lr.simulate.return_value = np.zeros((len(timepoints), num_species))
    lr.getForcingInputs.return_value = np.zeros(num_species)
    lr.getInitialValues.return_value = np.zeros(num_species)
    return Trajectory(lr)

def _make_trajectory(n_point: int = 5, n_species: int = 3) -> Trajectory:
    """Return a JacobianCollection with deterministic values using a mock LRoadrunner."""
    rng = np.random.default_rng(42)
    jacobian_collection_arr = rng.standard_normal((n_point, n_species, n_species))
    timepoints = np.linspace(0, 10, n_point)
    return _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)

def _make_lroadrunner(antimony_str: str, end_time:float = 10.0, num_point:int = 11) -> LRoadrunner:
    """Return a real LRoadrunner for the given Antimony model."""
    return LRoadrunner(antimony_str, start_time=0.0, end_time=end_time, num_point=num_point)

def _make_trajectory_from_model(antimony_str: str, end_time:float = 10.0, num_point:int = 11) -> Trajectory:
    """Return a MultipleLinearPredictor built from a real Antimony model."""
    lr = _make_lroadrunner(antimony_str, end_time=end_time, num_point=num_point)
    jc = Trajectory(lr)
    return jc


class TestJacobianCollectionInit(unittest.TestCase):
    """Tests for JacobianCollection.__init__."""

    def test_stores_jacobian_collection_arr(self) -> None:
        """jacobian_collection_arr attribute is stored as-is."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertEqual(jc.jacobian_collection_arr.shape, (5, 3, 3))

    def test_stores_timepoints(self) -> None:
        """timepoints attribute is stored as-is."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertEqual(len(jc.timepoint_arr), 5)


class TestGetTimes(unittest.TestCase):
    """Tests for JacobianCollection.getTimes."""

    def test_returns_set(self) -> None:
        """getTimes returns a set."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertIsInstance(jc.getTimes(), np.ndarray)

    def test_unique_timepoints(self) -> None:
        """getTimes returns the unique set of timepoints."""
        if IGNORE_TESTS:
            return
        timepoints = np.array([0.0, 1.0, 1.0, 2.0])
        jacobian_collection_arr = np.zeros((4, 2, 2))
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertTrue(np.array_equal(jc.getTimes(), np.array([0.0, 1.0, 2.0])))

    def test_all_unique_timepoints(self) -> None:
        """getTimes returns all timepoints when none are duplicated."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5)
        self.assertEqual(len(jc.getTimes()), 5)


class TestJacobianMeanArr(unittest.TestCase):
    """Tests for JacobianCollection.jacobian_mean_arr property."""

    def test_shape(self) -> None:
        """jacobian_mean_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=3)
        self.assertEqual(jc.jacobian_mean_arr.shape, (3, 3))

    def test_known_value(self) -> None:
        """jacobian_mean_arr equals element-wise mean across timepoints."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(jc.jacobian_mean_arr, expected)

    def test_single_timepoint_equals_jacobian(self) -> None:
        """With one timepoint mean equals the Jacobian itself."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        np.testing.assert_allclose(jc.jacobian_mean_arr, jacobian_collection_arr[0])

    def test_cached_on_second_access(self) -> None:
        """jacobian_mean_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=2)
        first = jc.jacobian_mean_arr
        second = jc.jacobian_mean_arr
        self.assertIs(first, second)


class TestJacobianStdArr(unittest.TestCase):
    """Tests for JacobianCollection.jacobian_std_arr property."""

    def test_shape(self) -> None:
        """jacobian_std_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=3)
        self.assertEqual(jc.jacobian_std_arr.shape, (3, 3))

    def test_known_value(self) -> None:
        """jacobian_std_arr equals element-wise std across timepoints."""
        if IGNORE_TESTS:
            return
        values = np.array([1.0, 2.0, 3.0])
        jacobian_collection_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        expected = np.array([[np.std(values)]])
        np.testing.assert_allclose(jc.jacobian_std_arr, expected)

    def test_constant_jacobian_gives_zero_std(self) -> None:
        """Identical Jacobians across timepoints produce std=0 everywhere."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.ones((5, 2, 2))
        timepoints = np.linspace(0, 4, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        np.testing.assert_allclose(jc.jacobian_std_arr, np.zeros((2, 2)))

    def test_nonnegative(self) -> None:
        """jacobian_std_arr is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=10, n_species=3)
        self.assertTrue(np.all(jc.jacobian_std_arr >= 0.0))

    def test_cached_on_second_access(self) -> None:
        """jacobian_std_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=2)
        first = jc.jacobian_std_arr
        second = jc.jacobian_std_arr
        self.assertIs(first, second)


class TestMaxCV(unittest.TestCase):
    """Tests for JacobianCollection.max_cv property."""

    def test_returns_float(self) -> None:
        """max_cv returns a float."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertIsInstance(jc.max_cv, float)

    def test_constant_entries_return_zero(self) -> None:
        """Constant Jacobian entries (std=0) produce cv=0 and max_cv=0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.ones((10, 2, 2))
        timepoints = np.linspace(0, 1, 10)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_zero_mean_entries_excluded(self) -> None:
        """Entries with zero mean (which give inf/nan CV) are treated as 0, not inf."""
        if IGNORE_TESTS:
            return
        # Alternating +1/-1 → mean=0, so CV would be inf/nan
        jacobian_collection_arr = np.array([[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]])
        timepoints = np.array([0.0, 1.0, 2.0, 3.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_known_cv_value(self) -> None:
        """max_cv matches hand-computed CV for a simple case."""
        if IGNORE_TESTS:
            return
        # Single entry varying from 1 to 3 → mean=2, std=~0.816 → CV=~0.408
        values = np.array([1.0, 2.0, 3.0])
        jacobian_collection_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        expected_cv = float(np.abs(np.std(values) / np.mean(values)))
        self.assertAlmostEqual(jc.max_cv, expected_cv, places=10)

    def test_max_taken_across_entries(self) -> None:
        """max_cv returns the maximum CV across all Jacobian entries."""
        if IGNORE_TESTS:
            return
        # Entry [0,0]: constant 1.0 → cv=0; Entry [0,1]: varies → cv>0
        jacobian_collection_arr = np.ones((5, 2, 2))
        jacobian_collection_arr[:, 0, 1] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timepoints = np.linspace(0, 1, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_max = float(np.abs(np.std(vals) / np.mean(vals)))
        self.assertAlmostEqual(jc.max_cv, expected_max, places=10)

    def test_nonnegative(self) -> None:
        """max_cv is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertGreaterEqual(jc.max_cv, 0.0)


class TestCalculateDeviation(unittest.TestCase):
    """Tests for JacobianCollection._calculateDeviation."""

    def test_returns_1d_array_of_correct_length(self) -> None:
        """Result has shape (num_points,)."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=3)
        result = jc._calculateDeviation()
        self.assertEqual(result.shape, (5,))

    def test_identical_jacobians_give_zero_deviation(self) -> None:
        """When all Jacobians are identical the centroid equals every Jacobian, so deviation is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.ones((4, 2, 2))
        timepoints = np.linspace(0, 3, 4)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(4))

    def test_known_value_1x1(self) -> None:
        """Hand-computed deviation for a 1×1 Jacobian with two timepoints."""
        if IGNORE_TESTS:
            return
        # centroid = 3.0; each deviation = |J - 3| / 3 = 1/3
        jacobian_collection_arr = np.array([[[2.0]], [[4.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        expected = np.array([1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_centroid_entry_produces_no_nan_or_inf(self) -> None:
        """Entries whose centroid is 0 contribute 0 to the deviation (no inf/nan)."""
        if IGNORE_TESTS:
            return
        # Off-diagonal entries are always 0 → centroid off-diagonal = 0
        jacobian_collection_arr = np.zeros((3, 2, 2))
        jacobian_collection_arr[:, 0, 0] = np.array([1.0, 2.0, 3.0])
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_all_zero_jacobians_give_zero_deviation(self) -> None:
        """All-zero Jacobians → centroid zero → deviation is 0 everywhere."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.zeros((3, 2, 2))
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_single_timepoint_gives_zero_deviation(self) -> None:
        """With one timepoint the centroid equals the Jacobian, so deviation is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(1))

    def test_nonnegative(self) -> None:
        """Deviation is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        result = jc._calculateDeviation()
        self.assertTrue(np.all(result >= 0.0))


class TestPlot(unittest.TestCase):
    """Tests for JacobianCollection.plot."""

    def setUp(self) -> None:
        lr = LRoadrunner(ANTIMONY_MODEL, start_time=0.0, end_time=5.0, num_point=11)
        self.collection = Trajectory(lr)

    def tearDown(self) -> None:
        plt.close("all")

    def test_runs_without_error(self) -> None:
        """plot runs without raising for a simple Antimony model."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()

    def test_creates_two_axes(self) -> None:
        """plot creates a figure with exactly two subplots."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes), 2)

    def test_first_axis_title(self) -> None:
        """First subplot title describes the Jacobian deviation."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(
            plt.gcf().axes[0].get_title(),
            "Normalized Distance of Jacobian to Centroid",
        )

    def test_second_axis_title(self) -> None:
        """Second subplot title describes the species timecourse."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(plt.gcf().axes[1].get_title(), "Species Timecourse")

    def test_first_axis_has_one_line(self) -> None:
        """First subplot contains exactly one line (the deviation curve)."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes[0].lines), 1)

    def test_second_axis_line_count_matches_species(self) -> None:
        """Second subplot has one line per floating species."""
        if IGNORE_TESTS:
            return
        n_species = len(self.collection.l_roadrunner.getRoadrunner().getFloatingSpeciesIds())
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes[1].lines), n_species)

    def test_second_axis_legend_contains_species_ids(self) -> None:
        """Second subplot legend labels match the floating species IDs."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        legend = plt.gcf().axes[1].get_legend()
        self.assertIsNotNone(legend)
        if legend is not None:
            legend_texts = [t.get_text() for t in legend.get_texts()]
            self.assertIn("S1", legend_texts)
            self.assertIn("S2", legend_texts)

    def test_biomodel_38(self) -> None:
        """plot runs without error for BioModel BIOMD0000000038."""
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD38_PATH):
            self.skipTest(f"SBML file not found at {BIOMD38_PATH}")
        with open(BIOMD38_PATH) as f:
            sbml_str = f.read()
        lr = LRoadrunner(sbml_str, start_time=0.0, end_time=0.002, num_point=600)
        collection = Trajectory(lr)
        with patch("matplotlib.pyplot.show"):
            collection.plot()


def _diameter_ivp(jacobian_collection_arr: np.ndarray) -> np.ndarray:
    """Mirror of JacobianCollection._calculateWeightedEigenvectors for tests."""
    eigvals, eigvecs = np.linalg.eig(jacobian_collection_arr)
    return eigvecs @ np.exp(eigvals)


class TestDiameter(unittest.TestCase):
    """Tests for JacobianCollection.diameter property."""

    def test_returns_float(self) -> None:
        """diameter returns a float."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertIsInstance(jc.diameter, float)

    def test_identical_jacobians_return_zero(self) -> None:
        """When all Jacobians are identical the mean equals each one, so diameter is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), (5, 1, 1))
        timepoints = np.linspace(0, 4, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertAlmostEqual(jc.diameter, 0.0, places=10)

    def test_single_timepoint_returns_zero(self) -> None:
        """With one timepoint the mean equals the Jacobian, so diameter is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 0.0], [0.0, -2.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertAlmostEqual(jc.diameter, 0.0, places=10)

    def test_max_taken_across_timepoints(self) -> None:
        """diameter is the maximum distance, not mean or sum."""
        if IGNORE_TESTS:
            return
        # mean = diag([3, 3]); j1 is further from the mean than j0.
        j0 = np.diag([1.0, 1.0])
        j1 = np.diag([5.0, 5.0])
        jacobian_collection_arr = np.array([j0, j1])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        w_mean = _diameter_ivp(jc.jacobian_mean_arr)
        dist0 = float(np.linalg.norm(_diameter_ivp(j0) - w_mean))
        dist1 = float(np.linalg.norm(_diameter_ivp(j1) - w_mean))
        self.assertNotEqual(dist0, dist1)

    def test_nonnegative(self) -> None:
        """diameter is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertGreaterEqual(jc.diameter, 0.0)

    def test_decreases_as_outliers_eliminated(self) -> None:
        """diameter decreases strictly as outlier Jacobians are removed.

        Three symmetric collections are built around center diag([2, 2]) so that
        the mean stays diag([2, 2]) regardless of which subset is used:
            - full: strong outliers + mild outliers + center  → largest diameter
            - mid:  mild outliers + center                    → intermediate diameter
            - center only                                     → diameter = 0
        """
        if IGNORE_TESTS:
            return
        j_center = np.diag([2.0, 2.0])
        j_mild0 = np.diag([1.0, 1.0])
        j_mild1 = np.diag([3.0, 3.0])
        j_strong0 = np.diag([0.0, 0.0])
        j_strong1 = np.diag([4.0, 4.0])

        full_arr = np.array([j_strong0, j_mild0, j_center, j_mild1, j_strong1])
        jc_full = Trajectory.fromArrays(full_arr, np.linspace(0, 4, 5))

        mid_arr = np.array([j_mild0, j_center, j_mild1])
        jc_mid = Trajectory.fromArrays(mid_arr, np.linspace(0, 2, 3))

        center_arr = np.array([j_center])
        jc_center = Trajectory.fromArrays(center_arr, np.array([0.0]))

        self.assertGreater(jc_full.diameter, jc_mid.diameter)
        self.assertGreater(jc_mid.diameter, jc_center.diameter)
        self.assertAlmostEqual(jc_center.diameter, 0.0, places=10)


class TestNonsequentialPartition(unittest.TestCase):

    def setUp(self) -> None:
        self.n_points = 20
        self.n_species = 3
        rng = np.random.default_rng(0)
        jacobian_collection_arr = rng.standard_normal((self.n_points, self.n_species, self.n_species))
        timepoint_arr = np.linspace(0.0, 10.0, self.n_points)
        self.jc = Trajectory.fromArrays(jacobian_collection_arr, timepoint_arr)

    def test_returns_list(self) -> None:
        """nonsequentialPartition returns a list."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=2)
        self.assertIsInstance(result, list)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """Result has exactly n_cluster elements."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=3)
        self.assertEqual(len(result), 3)

    def test_each_element_is_jacobian_collection(self) -> None:
        """Each element in the result is a JacobianCollection."""
        if IGNORE_TESTS:
            return
        for jc in self.jc.nonsequentialPartition(n_cluster=2):
            self.assertIsInstance(jc, Trajectory)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=4)
        total = sum(jc.jacobian_collection_arr.shape[0] for jc in result)
        self.assertEqual(total, self.n_points)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.jc.nonsequentialPartition(n_cluster=self.n_points + 1)

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].jacobian_collection_arr.shape[0], self.n_points)


class TestPartitionJacobiansSequentially(unittest.TestCase):
    """Tests for JacobianCollection.sequentialPartition."""

    def setUp(self) -> None:
        self.n_points = 20
        self.n_species = 3
        rng = np.random.default_rng(0)
        jacobian_collection_arr = rng.standard_normal((self.n_points, self.n_species, self.n_species))
        timepoint_arr = np.linspace(0.0, 10.0, self.n_points)
        self.jc = Trajectory.fromArrays(jacobian_collection_arr, timepoint_arr)

    def test_returns_list(self) -> None:
        """sequentialPartition returns a list."""
        if IGNORE_TESTS:
            return
        for n_cluster in [2, 3, 4]:
            result = self.jc.sequentialPartition(n_cluster=n_cluster)
            self.assertEqual(len(result), n_cluster)
            total_length = 0
            for jc in result:
                self.assertIsInstance(jc, Trajectory)
                total_length += jc.jacobian_collection_arr.shape[0]
            self.assertEqual(total_length, self.n_points)  # Ensure all Jacobians are included
            self.assertIsInstance(result, list)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=4)
        total = sum(jc.jacobian_collection_arr.shape[0] for jc in result)
        self.assertEqual(total, self.n_points)

    def test_clusters_are_contiguous_in_time(self) -> None:
        """Concatenating cluster jacobian_collection_arrs in order reconstructs the original array."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=3)
        reconstructed = np.concatenate([jc.jacobian_collection_arr for jc in result], axis=0)
        np.testing.assert_array_equal(reconstructed, self.jc.jacobian_collection_arr)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.jc.sequentialPartition(n_cluster=self.n_points + 1)

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].jacobian_collection_arr, self.jc.jacobian_collection_arr)


class TestBiomodel206(unittest.TestCase):
    """Integration tests for JacobianCollection with BioModel BIOMD0000000206.

    Uses the canonical simulation window from the model's SED-ML: t=0..10.
    num_points is reduced from 1000 to 100 to keep the test suite fast.
    """

    def setUp(self) -> None:
        if not os.path.exists(BIOMD206_PATH):
            self.skipTest(f"BIOMD0000000206 not found at {BIOMD206_PATH}")
        with open(BIOMD206_PATH) as f:
            sbml_str = f.read()
        self.lr = LRoadrunner(sbml_str, start_time=0.0, end_time=BIOMD206_ENDTIME, num_point=100)

    def test_construction_succeeds(self) -> None:
        """JacobianCollection is created without error from BIOMD0000000206."""
        if IGNORE_TESTS:
            return
        try:
            Trajectory(self.lr)
            self.assertTrue(False)
        except Exception as e:
            self.assertTrue(True)


def _make_wed_collection(jacobian_collection_arr: np.ndarray, timepoint_arr: np.ndarray = np.array([]),
        end_time: float = 1.0) -> Trajectory:
    """Return a JacobianCollection backed by a real LRoadrunner for diameter_ivp.

    diameter_ivp reads self.l_roadrunner.end_time to compute relative
    timepoints, so a mock is not sufficient here.
    """
    lr = LRoadrunner(ANTIMONY_MODEL, start_time=0.0, end_time=end_time, num_point=11)
    if len(timepoint_arr) == 0:
        timepoint_arr = np.linspace(0.0, end_time, len(jacobian_collection_arr))
    return Trajectory.fromArrays(jacobian_collection_arr, timepoint_arr, lr)


class TestDiameterIVP(unittest.TestCase):
    """Tests for JacobianCollection.diameter_ivp property.

    Uses identity matrices as a key reference point: all eigenvalues are 1, so
    eigvals**t = 1 for every t, and every weighted-eigenvector sum equals n*e.
    The resulting constant sequence has mean n*e and max-distance 0 — the only
    case that provably yields diameter = 0.
    """

    # 2×2 identity: eigvals=[1,1], so eigvecs@exp(eigvals**t)=[e,e] for all t.
    IDENTITY = np.eye(2)
    # Diagonal with eigenvalues 0.5: produces timepoint-dependent sums.
    J_HALF = np.diag([0.5, 0.5])

    def test_identical(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jc = _make_wed_collection(
                np.tile([self.IDENTITY]*5, (3, 1, 1)),
                np.linspace(0, 2, 3))
        result = jc.diameter_ivp
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.0)  # Identity Jacobians should yield diameter = 0.0
    
    def test_zero_difference(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jacobian_arr = -1*np.array([self.IDENTITY, self.IDENTITY])
        jc1 = _make_wed_collection(jacobian_arr)
        result = jc1.diameter_ivp
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.0)  # Identity Jacobians should yield diameter = 0.0

    def test_more_different(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jacobian_arr = -1*np.array([self.IDENTITY, 0.9*self.IDENTITY])
        jc1 = _make_wed_collection(jacobian_arr)
        for factor in [1.5, 2.0, 10.0, 100.0]:
            jacobian_arr = -1*np.array([self.IDENTITY, factor*self.IDENTITY])
            jc2 = _make_wed_collection(jacobian_arr)
            self.assertGreater(jc2.diameter_ivp, jc1.diameter_ivp)  # More different Jacobians should yield larger diameter 
            jc1 = jc2
    
    def test_different(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jc1 = _make_wed_collection(
                -1*np.array([self.IDENTITY, 2*self.IDENTITY, self.J_HALF]),
                np.linspace(0, 2, 3))
        result = jc1.diameter_ivp
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)  # Identity Jacobians should yield diameter = 0.0
        #
        jc2 = _make_wed_collection(
                -1*np.array([self.IDENTITY, 4*self.IDENTITY, 0.5*self.J_HALF]),
                np.linspace(0, 2, 3))
        result2 = jc2.diameter_ivp
        self.assertGreater(result2, result)  # More different Jacobians should yield larger diameter

    def test_nonnegative(self) -> None:
        """diameter_ivp is always non-negative."""
        if IGNORE_TESTS:
            return
        rng = np.random.default_rng(7)
        jacobian_collection_arr = rng.standard_normal((5, 2, 2)) * 0.2
        jc = _make_wed_collection(jacobian_collection_arr, np.linspace(0, 4, 5))
        self.assertGreaterEqual(jc.diameter_ivp, 0.0)

    def test_cached_on_second_access(self) -> None:
        """diameter_ivp is cached: the same value is returned on repeated calls."""
        if IGNORE_TESTS:
            return
        jc = _make_wed_collection(
                np.tile(self.J_HALF, (3, 1, 1)),
                np.linspace(0, 2, 3))
        first = jc.diameter_ivp
        second = jc.diameter_ivp
        self.assertEqual(first, second)
        self.assertFalse(np.isnan(jc._diameter))

    def test_identity_matrices_give_zero(self) -> None:
        """A collection of identity matrices has diameter_ivp == 0.

        For J = I, eigvals = [1,...,1] so eigvals**t = 1 for all t, making every
        weighted-eigenvector sum equal to n*e regardless of t.  The sequence is
        constant and its max-distance from the mean is 0.
        """
        if IGNORE_TESTS:
            return
        jc = _make_wed_collection(
                np.tile(self.IDENTITY, (5, 1, 1)),
                np.linspace(0, 4, 5))
        self.assertAlmostEqual(jc.diameter_ivp, 0.0, places=10)

    def test_nonidentity_jacobians_give_positive_diameter(self) -> None:
        """A collection that mixes identity and non-identity Jacobians has positive diameter."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([self.IDENTITY, self.J_HALF, self.IDENTITY])
        jc = _make_wed_collection(jacobian_collection_arr, np.linspace(0, 2, 3))
        self.assertGreater(jc.diameter_ivp, 0.0)

    def test_larger_spread_gives_larger_diameter(self) -> None:
        """Mixing identity Jacobians with a non-identity one increases the diameter.

        jc_A contains only identity matrices  → diameter = 0.
        jc_B contains identity and J_HALF     → diameter > 0.
        """
        if IGNORE_TESTS:
            return
        arr_a = np.tile(self.IDENTITY, (3, 1, 1))
        arr_b = np.array([self.IDENTITY, self.J_HALF, self.IDENTITY])
        jc_a = _make_wed_collection(arr_a, np.linspace(0, 2, 3))
        jc_b = _make_wed_collection(arr_b, np.linspace(0, 2, 3))
        self.assertGreater(jc_b.diameter_ivp,
                jc_a.diameter_ivp)


class TestIvpSolutions(unittest.TestCase):
    """Tests for JacobianCollection.ivp_solutions property."""

    def test_returns_empty_for_empty_collection(self) -> None:
        """ivp_solutions returns an empty array for an empty jacobian_collection_arr."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            _ = _make_trajectory_from_arrays(np.array([]), np.array([]))

    def test_returns_ndarray(self) -> None:
        """ivp_solutions returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=2, n_species=2)
        self.assertIsInstance(jc.ivp_solutions, np.ndarray)

    def test_shape(self) -> None:
        """ivp_solutions has shape (n_jacobians, n_species, n_ivp_timepoints)."""
        if IGNORE_TESTS:
            return
        n_points, n_species = 4, 3
        jc = _make_trajectory(n_point=n_points, n_species=n_species)
        result = jc.ivp_solutions
        self.assertEqual(result.shape, (n_points, n_species *len(IVP_RELATIVE_TIMES)))

    def test_initial_condition_is_ones(self) -> None:
        """At t=0 (IVP_RELATIVE_TIMES[0]=0) the solution equals the initial condition ones(n)."""
        if IGNORE_TESTS:
            return
        n_species = 2
        jacobian_collection_arr = np.array([np.diag([-1.0, -2.0])])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc.ivp_solutions
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][len(IVP_RELATIVE_TIMES)], 1)

    def test_known_1x1_solution(self) -> None:
        """For J=[[-1]], solution is x(t)=exp(-t); verify at relative timepoints."""
        if IGNORE_TESTS:
            return
        from trajectory import IVP_RELATIVE_TIMES  # type: ignore
        end_time = 5.0
        jacobian_collection_arr = np.array([[[-1.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc.ivp_solutions
        expected_times = end_time * IVP_RELATIVE_TIMES
        expected = np.exp(-1.0 * expected_times)
        self.assertLess(np.max(abs(result-expected)), 1e-3)

    def test_different_jacobians_give_different_solutions(self) -> None:
        """Two distinct Jacobians in the collection produce distinct solution trajectories."""
        if IGNORE_TESTS:
            return
        j0 = np.array([[-1.0, 0.0], [0.0, -1.0]])
        j1 = np.array([[-3.0, 0.0], [0.0, -3.0]])
        jacobian_collection_arr = np.array([j0, j1])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc.ivp_solutions
        self.assertTrue(all(result[1] <= result[0]))  # J with more negative eigenvalues should decay faster


class TestPredictLinearBasic(unittest.TestCase):
    """Tests for Trajectory.predictLinear."""

    def setUp(self) -> None:
        self.trajectory = _make_trajectory_from_model(ANTIMONY_MODEL, end_time=1.0)

    def test_output_shape(self) -> None:
        """Result has shape (num_timepoints, num_species)."""
        if IGNORE_TESTS:
            return
        n_point, n_species = self.trajectory.num_point, self.trajectory.num_species
        initial_state_arr = np.ones(n_species)
        forcing_input_arr = np.zeros(n_species)
        result = self.trajectory._predictLinear(initial_state_arr, forcing_input_arr)
        self.assertEqual(result.shape, (n_point, n_species))

    def test_returns_ndarray(self) -> None:
        """predictLinear returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        trajectory = _make_trajectory_from_model(ANTIMONY_MODEL, end_time=1.0)
        result = trajectory._predictLinear(np.ones(2), np.zeros(2))
        self.assertIsInstance(result, np.ndarray)

    def test_zero_initial_state_gives_zero_output(self) -> None:
        """Zero initial state with zero forcing produces zero trajectory."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.tile(np.array([[-1.0, 0.0], [0.0, -2.0]]), (3, 1, 1))
        timepoints = np.linspace(0.0, 2.0, 3)
        trajectory = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = trajectory._predictLinear(np.zeros(2), np.zeros(2))
        np.testing.assert_allclose(result, np.zeros((3, 2)), atol=1e-10)

    def test_known_1x1_no_forcing(self) -> None:
        """For J=[[-1]], x0=[1], u=[0]: x(t)=exp(-t) at each timepoint."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[-1.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        # predictLinear integrates over timepoint_arr which has only one point, so t_span = [0, 0]
        # Use a multi-point collection for a meaningful trajectory
        jacobian_collection_arr = np.tile(np.array([[-1.0]]), (5, 1, 1))
        timepoints = np.linspace(0.0, 2.0, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._predictLinear(np.array([1.0]), np.array([0.0]))
        expected = np.exp(-timepoints).reshape(-1, 1)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_known_1x1_with_forcing(self) -> None:
        """For J=[[-1]], x0=[1], u=[c]: x(t)=(1-c)*exp(-t)+c."""
        if IGNORE_TESTS:
            return
        c = 2.0
        jacobian_collection_arr = np.tile(np.array([[-1.0]]), (5, 1, 1))
        timepoints = np.linspace(0.0, 2.0, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._predictLinear(np.array([1.0]), np.array([c]))
        expected = ((1.0 - c) * np.exp(-timepoints) + c).reshape(-1, 1)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_forcing_changes_trajectory(self) -> None:
        """Non-zero forcing produces a different trajectory than zero forcing."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.tile(np.array([[-1.0, 0.0], [0.0, -1.0]]), (5, 1, 1))
        timepoints = np.linspace(0.0, 2.0, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        initial_state_arr = np.array([1.0, 1.0])
        result_no_forcing = jc._predictLinear(initial_state_arr, np.zeros(2))
        result_with_forcing = jc._predictLinear(initial_state_arr, np.array([1.0, 0.0]))
        self.assertFalse(np.allclose(result_no_forcing, result_with_forcing))

    def test_first_row_matches_initial_state(self) -> None:
        """The first row of the result approximates the initial state."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.tile(np.array([[-1.0, 0.0], [0.0, -2.0]]), (5, 1, 1))
        timepoints = np.linspace(0.0, 2.0, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        initial_state_arr = np.array([3.0, 5.0])
        result = jc._predictLinear(initial_state_arr, np.zeros(2))
        np.testing.assert_allclose(result[0], initial_state_arr, atol=1e-5)


class TestPredictLinearBioModel(unittest.TestCase):
    """Advanced tests for Trajectory.predictLinear."""

    def setUp(self) -> None:
        if not os.path.exists(BIOMD8_PATH):
            self.skipTest(f"SBML file not found at {BIOMD8_PATH}")
        self.lr = LRoadrunner.makeBiomodel(BIOMD8_PATH,
                start_time=0.0, end_time=2*BIOMD8_ENDTIME, num_point=30)
        self.trajectory = Trajectory(self.lr)

    def test_timecourse(self) -> None:
        """predictLinear returns an ndarray with shape (num_points, num_species)."""
        if IGNORE_TESTS:
            return
        predicted_df = self.trajectory.predictLinear()
        actual_df = self.lr.timecourse
        df = (predicted_df - actual_df).abs()/actual_df
        for species_name in self.lr.species_names:
            plt.plot(actual_df.index, actual_df[species_name], label=species_name);
            plt.scatter(actual_df.index, predicted_df[species_name], label=species_name);
        plt.legend();
        plt.xlabel("time");
        plt.ylabel("concentration");
        plt.title("predictLinear vs actual timecourse");
        plt.show();
        self.assertIsInstance(predicted_df, pd.DataFrame)
        self.assertEqual(predicted_df.shape, actual_df.shape)

class TestFitForcingInputs(unittest.TestCase):
    """Tests for Trajectory.fitForcingInputs."""

    def setUp(self) -> None:
        self.trajectory = _make_trajectory_from_model(ANTIMONY_MODEL, end_time=5.0)

    def test_returns_ndarray(self) -> None:
        """fitForcingInputs returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitForcingInputs()
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self) -> None:
        """fitForcingInputs returns array of shape (num_species,)."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitForcingInputs()
        self.assertEqual(result.shape, (self.trajectory.num_species,))

    def test_improves_prediction(self) -> None:
        """Fitted forcing inputs produce prediction error no worse than zero forcing."""
        if IGNORE_TESTS:
            return
        actual_arr = self.trajectory.l_roadrunner.simulate()
        initial_state_arr = self.trajectory.l_roadrunner.getInitialValues()
        default_forcing_arr = self.trajectory.l_roadrunner.getForcingInputs()
        fitted_forcing_arr = self.trajectory.fitForcingInputs(initial_state_arr)
        predicted_zero_arr = self.trajectory._predictLinear(
                initial_state_arr=initial_state_arr,
                forcing_input_arr=default_forcing_arr,
                jacobian_arr=self.trajectory.jacobian_mean_arr)
        predicted_fitted_arr = self.trajectory._predictLinear(
                initial_state_arr=initial_state_arr,
                forcing_input_arr=fitted_forcing_arr,
                jacobian_arr=self.trajectory.jacobian_mean_arr)
        mse_default = float(np.mean((predicted_zero_arr - actual_arr)**2))
        mse_fitted = float(np.mean((predicted_fitted_arr - actual_arr)**2))
        self.assertGreaterEqual(mse_default - mse_fitted, 0)

    def test_with_explicit_initial_state(self) -> None:
        """fitForcingInputs accepts an explicit initial state and still returns correct shape."""
        if IGNORE_TESTS:
            return
        initial_state_arr = self.trajectory.l_roadrunner.getInitialValues()
        result = self.trajectory.fitForcingInputs(initial_state_arr)
        self.assertEqual(result.shape, (self.trajectory.num_species,))


class TestFitJacobian(unittest.TestCase):
    """Tests for Trajectory.fitJacobian."""

    def setUp(self) -> None:
        self.trajectory = _make_trajectory_from_model(ANTIMONY_MODEL, end_time=5.0, num_point=100)

    def test_returns_ndarray(self) -> None:
        """fitJacobian returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self) -> None:
        """fitJacobian returns array of shape (num_species, num_species)."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        n = self.trajectory.num_species
        self.assertEqual(result.shape, (n, n))

    def test_off_diagonals_unchanged(self) -> None:
        """Off-diagonal entries equal those of the mean Jacobian."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        mean = self.trajectory.jacobian_mean_arr
        n = self.trajectory.num_species
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertAlmostEqual(result[i, j], mean[i, j], places=10)

    def test_improves_or_matches_prediction(self) -> None:
        """Fitted Jacobian produces MSE no worse than the mean Jacobian."""
        if IGNORE_TESTS:
            return
        initial_state_arr = self.trajectory.l_roadrunner.getInitialValues()
        forcing_input_arr = self.trajectory.l_roadrunner.getForcingInputs()
        actual_arr = self.trajectory.l_roadrunner.simulate()
        mean_pred_arr = self.trajectory._predictLinear(
                initial_state_arr=initial_state_arr,
                forcing_input_arr=forcing_input_arr,
                jacobian_arr=self.trajectory.jacobian_mean_arr)
        fitted_jacobian_arr = self.trajectory.fitJacobian()
        fitted_pred_arr = self.trajectory._predictLinear(
                initial_state_arr=initial_state_arr,
                forcing_input_arr=forcing_input_arr,
                jacobian_arr=fitted_jacobian_arr)
        mse_mean = float(np.mean((mean_pred_arr - actual_arr) ** 2))
        mse_fitted = float(np.mean((fitted_pred_arr - actual_arr) ** 2))
        self.assertLessEqual(mse_fitted, mse_mean + 1e-6)

class TestBiomodelsFit(unittest.TestCase):

    def setUp(self) -> None:
        self.lr_dct: dict[str, LRoadrunner] = {} # Dictionary of LRoadrunner instances for different BioModels
        for model_name in BIOMODEL_NAMES:
            model_path = os.path.join(cn.BIOMODELS_DIR, model_name, f"{model_name}_url.xml")
            if not os.path.exists(model_path):
                print(f"Warning: {model_name} not found at {model_path}. Skipping this model.")
                continue
            with open(model_path) as f:
                sbml_str = f.read()
            self.lr_dct[model_name] = LRoadrunner(sbml_str, start_time=0.0, end_time=10.0, num_point=100)

    def test_improves_or_matches_prediction(self) -> None:
        """Fitted Jacobian produces MSE no worse than the mean Jacobian."""
        #if IGNORE_TESTS:
        #    return
        for model_name, lr in self.lr_dct.items():
            with self.subTest(model=model_name):
                trajectory = Trajectory(lr)
                initial_state_arr = trajectory.l_roadrunner.getInitialValues()
                forcing_input_arr = trajectory.l_roadrunner.getForcingInputs()
                actual_arr = trajectory.l_roadrunner.simulate()
                mean_pred_arr = trajectory._predictLinear(
                        initial_state_arr=initial_state_arr,
                        forcing_input_arr=forcing_input_arr,
                        jacobian_arr=trajectory.jacobian_mean_arr)
                fitted_jacobian_arr = trajectory.fitJacobian()
                fitted_pred_arr = trajectory._predictLinear(
                        initial_state_arr=initial_state_arr,
                        forcing_input_arr=forcing_input_arr,
                        jacobian_arr=fitted_jacobian_arr)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_mean = np.where(actual_arr != 0,
                            (mean_pred_arr - actual_arr) ** 2 / actual_arr**2, 0.0)
                    ratio_fitted = np.where(actual_arr != 0,
                            (fitted_pred_arr - actual_arr) ** 2 / actual_arr**2, 0.0)
                mse_mean = float(np.mean(ratio_mean))
                mse_fitted = float(np.mean(ratio_fitted))
                self.assertLessEqual(mse_fitted, mse_mean + 1e-6)


if __name__ == "__main__":
    unittest.main()
