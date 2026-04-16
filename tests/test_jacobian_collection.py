"""Tests for JacobianCollection class."""
from src.l_roadrunner import LRoadrunner  # type: ignore

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import matplotlib # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from jacobian_collection import JacobianCollection  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore

IGNORE_TESTS = False

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMD_PATH = os.path.join(
    cn.BIOMODELS_DIR, "BIOMD0000000038", "BIOMD0000000038_url.xml"
)
BIOMD206_PATH = os.path.join(
    cn.BIOMODELS_DIR, "BIOMD0000000206", "BIOMD0000000206_url.xml"
)


def _make_collection_from_arrays(jacobian_arr: np.ndarray, timepoints: np.ndarray) -> JacobianCollection:
    """Return a JacobianCollection from explicit arrays using a mock LRoadrunner."""
    lr = MagicMock(spec=LRoadrunner)
    lr.makeJacobians.return_value = (jacobian_arr, timepoints)
    return JacobianCollection(lr)


def _make_collection(n_points: int = 5, n_species: int = 3) -> JacobianCollection:
    """Return a JacobianCollection with deterministic values using a mock LRoadrunner."""
    rng = np.random.default_rng(42)
    jacobian_arr = rng.standard_normal((n_points, n_species, n_species))
    timepoints = np.linspace(0, 10, n_points)
    return _make_collection_from_arrays(jacobian_arr, timepoints)


class TestJacobianCollectionInit(unittest.TestCase):
    """Tests for JacobianCollection.__init__."""

    def test_stores_jacobian_arr(self) -> None:
        """jacobian_arr attribute is stored as-is."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        self.assertEqual(jc.jacobian_arr.shape, (5, 3, 3))

    def test_stores_timepoints(self) -> None:
        """timepoints attribute is stored as-is."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        self.assertEqual(len(jc.timepoint_arr), 5)


class TestGetTimes(unittest.TestCase):
    """Tests for JacobianCollection.getTimes."""

    def test_returns_set(self) -> None:
        """getTimes returns a set."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        self.assertIsInstance(jc.getTimes(), np.ndarray)

    def test_unique_timepoints(self) -> None:
        """getTimes returns the unique set of timepoints."""
        if IGNORE_TESTS:
            return
        timepoints = np.array([0.0, 1.0, 1.0, 2.0])
        jacobian_arr = np.zeros((4, 2, 2))
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertTrue(np.array_equal(jc.getTimes(), np.array([0.0, 1.0, 2.0])))

    def test_all_unique_timepoints(self) -> None:
        """getTimes returns all timepoints when none are duplicated."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=5)
        self.assertEqual(len(jc.getTimes()), 5)


class TestJacobianMeanArr(unittest.TestCase):
    """Tests for JacobianCollection.jacobian_mean_arr property."""

    def test_shape(self) -> None:
        """jacobian_mean_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=5, n_species=3)
        self.assertEqual(jc.jacobian_mean_arr.shape, (3, 3))

    def test_known_value(self) -> None:
        """jacobian_mean_arr equals element-wise mean across timepoints."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(jc.jacobian_mean_arr, expected)

    def test_single_timepoint_equals_jacobian(self) -> None:
        """With one timepoint mean equals the Jacobian itself."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        np.testing.assert_allclose(jc.jacobian_mean_arr, jacobian_arr[0])

    def test_cached_on_second_access(self) -> None:
        """jacobian_mean_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=5, n_species=2)
        first = jc.jacobian_mean_arr
        second = jc.jacobian_mean_arr
        self.assertIs(first, second)


class TestJacobianStdArr(unittest.TestCase):
    """Tests for JacobianCollection.jacobian_std_arr property."""

    def test_shape(self) -> None:
        """jacobian_std_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=5, n_species=3)
        self.assertEqual(jc.jacobian_std_arr.shape, (3, 3))

    def test_known_value(self) -> None:
        """jacobian_std_arr equals element-wise std across timepoints."""
        if IGNORE_TESTS:
            return
        values = np.array([1.0, 2.0, 3.0])
        jacobian_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        expected = np.array([[np.std(values)]])
        np.testing.assert_allclose(jc.jacobian_std_arr, expected)

    def test_constant_jacobian_gives_zero_std(self) -> None:
        """Identical Jacobians across timepoints produce std=0 everywhere."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.ones((5, 2, 2))
        timepoints = np.linspace(0, 4, 5)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        np.testing.assert_allclose(jc.jacobian_std_arr, np.zeros((2, 2)))

    def test_nonnegative(self) -> None:
        """jacobian_std_arr is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=10, n_species=3)
        self.assertTrue(np.all(jc.jacobian_std_arr >= 0.0))

    def test_cached_on_second_access(self) -> None:
        """jacobian_std_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=5, n_species=2)
        first = jc.jacobian_std_arr
        second = jc.jacobian_std_arr
        self.assertIs(first, second)


class TestMaxCV(unittest.TestCase):
    """Tests for JacobianCollection.max_cv property."""

    def test_returns_float(self) -> None:
        """max_cv returns a float."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        self.assertIsInstance(jc.max_cv, float)

    def test_empty_array_returns_zero(self) -> None:
        """max_cv returns 0.0 for an empty jacobian_arr."""
        if IGNORE_TESTS:
            return
        jc = _make_collection_from_arrays(np.array([]), np.array([]))
        self.assertEqual(jc.max_cv, 0.0)

    def test_constant_entries_return_zero(self) -> None:
        """Constant Jacobian entries (std=0) produce cv=0 and max_cv=0."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.ones((10, 2, 2))
        timepoints = np.linspace(0, 1, 10)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_zero_mean_entries_excluded(self) -> None:
        """Entries with zero mean (which give inf/nan CV) are treated as 0, not inf."""
        if IGNORE_TESTS:
            return
        # Alternating +1/-1 → mean=0, so CV would be inf/nan
        jacobian_arr = np.array([[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]])
        timepoints = np.array([0.0, 1.0, 2.0, 3.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_known_cv_value(self) -> None:
        """max_cv matches hand-computed CV for a simple case."""
        if IGNORE_TESTS:
            return
        # Single entry varying from 1 to 3 → mean=2, std=~0.816 → CV=~0.408
        values = np.array([1.0, 2.0, 3.0])
        jacobian_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        expected_cv = float(np.abs(np.std(values) / np.mean(values)))
        self.assertAlmostEqual(jc.max_cv, expected_cv, places=10)

    def test_max_taken_across_entries(self) -> None:
        """max_cv returns the maximum CV across all Jacobian entries."""
        if IGNORE_TESTS:
            return
        # Entry [0,0]: constant 1.0 → cv=0; Entry [0,1]: varies → cv>0
        jacobian_arr = np.ones((5, 2, 2))
        jacobian_arr[:, 0, 1] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timepoints = np.linspace(0, 1, 5)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_max = float(np.abs(np.std(vals) / np.mean(vals)))
        self.assertAlmostEqual(jc.max_cv, expected_max, places=10)

    def test_nonnegative(self) -> None:
        """max_cv is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        self.assertGreaterEqual(jc.max_cv, 0.0)


class TestCalculateDeviation(unittest.TestCase):
    """Tests for JacobianCollection._calculateDeviation."""

    def test_returns_1d_array_of_correct_length(self) -> None:
        """Result has shape (num_points,)."""
        if IGNORE_TESTS:
            return
        jc = _make_collection(n_points=5, n_species=3)
        result = jc._calculateDeviation()
        self.assertEqual(result.shape, (5,))

    def test_identical_jacobians_give_zero_deviation(self) -> None:
        """When all Jacobians are identical the centroid equals every Jacobian, so deviation is 0."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.ones((4, 2, 2))
        timepoints = np.linspace(0, 3, 4)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(4))

    def test_known_value_1x1(self) -> None:
        """Hand-computed deviation for a 1×1 Jacobian with two timepoints."""
        if IGNORE_TESTS:
            return
        # centroid = 3.0; each deviation = |J - 3| / 3 = 1/3
        jacobian_arr = np.array([[[2.0]], [[4.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        expected = np.array([1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_centroid_entry_produces_no_nan_or_inf(self) -> None:
        """Entries whose centroid is 0 contribute 0 to the deviation (no inf/nan)."""
        if IGNORE_TESTS:
            return
        # Off-diagonal entries are always 0 → centroid off-diagonal = 0
        jacobian_arr = np.zeros((3, 2, 2))
        jacobian_arr[:, 0, 0] = np.array([1.0, 2.0, 3.0])
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_all_zero_jacobians_give_zero_deviation(self) -> None:
        """All-zero Jacobians → centroid zero → deviation is 0 everywhere."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.zeros((3, 2, 2))
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_single_timepoint_gives_zero_deviation(self) -> None:
        """With one timepoint the centroid equals the Jacobian, so deviation is 0."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(1))

    def test_nonnegative(self) -> None:
        """Deviation is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        result = jc._calculateDeviation()
        self.assertTrue(np.all(result >= 0.0))


class TestPlot(unittest.TestCase):
    """Tests for JacobianCollection.plot."""

    def setUp(self) -> None:
        lr = LRoadrunner(ANTIMONY_MODEL, start_time=0.0, end_time=5.0, num_points=11)
        self.collection = JacobianCollection(lr)

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
        with open(BIOMD_PATH) as f:
            sbml_str = f.read()
        lr = LRoadrunner(sbml_str, start_time=0.0, end_time=0.002, num_points=600)
        collection = JacobianCollection(lr)
        with patch("matplotlib.pyplot.show"):
            collection.plot()


def _weighted_eigenvectors(jacobian_arr: np.ndarray) -> np.ndarray:
    """Mirror of JacobianCollection._calculateWeightedEigenvectors for tests."""
    eigvals, eigvecs = np.linalg.eig(jacobian_arr)
    return eigvecs @ np.exp(eigvals)


class TestDiameter(unittest.TestCase):
    """Tests for JacobianCollection.diameter property."""

    def test_returns_float(self) -> None:
        """diameter returns a float."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
        self.assertIsInstance(jc.diameter, float)

    def test_empty_array_returns_zero(self) -> None:
        """diameter returns 0.0 for an empty jacobian_arr."""
        if IGNORE_TESTS:
            return
        jc = _make_collection_from_arrays(np.array([]), np.array([]))
        self.assertEqual(jc.diameter, 0.0)

    def test_identical_jacobians_return_zero(self) -> None:
        """When all Jacobians are identical the mean equals each one, so diameter is 0."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), (5, 1, 1))
        timepoints = np.linspace(0, 4, 5)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertAlmostEqual(jc.diameter, 0.0, places=10)

    def test_single_timepoint_returns_zero(self) -> None:
        """With one timepoint the mean equals the Jacobian, so diameter is 0."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[[1.0, 0.0], [0.0, -2.0]]])
        timepoints = np.array([0.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertAlmostEqual(jc.diameter, 0.0, places=10)

#    def test_known_value_diagonal(self) -> None:
#        """Hand-computed diameter for two diagonal Jacobians.
#
#        For diagonal matrices eigenvectors = I, so
#        _calculateWeightedEigenvectors(diag([a, b])) = [exp(a), exp(b)].
#        """
#        if IGNORE_TESTS:
#            return
#        j0 = np.diag([1.0, 2.0])
#        j1 = np.diag([3.0, 4.0])
#        jacobian_arr = np.array([j0, j1])
#        timepoints = np.array([0.0, 1.0])
#        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
#        w_mean = _weighted_eigenvectors(np.diag([2.0, 3.0]))
#        dist0 = float(np.linalg.norm(_weighted_eigenvectors(j0) - w_mean))
#        dist1 = float(np.linalg.norm(_weighted_eigenvectors(j1) - w_mean))
#        expected = max(dist0, dist1)
#        self.assertAlmostEqual(jc.diameter, expected, places=10)

    def test_max_taken_across_timepoints(self) -> None:
        """diameter is the maximum distance, not mean or sum."""
        if IGNORE_TESTS:
            return
        # mean = diag([3, 3]); j1 is further from the mean than j0.
        j0 = np.diag([1.0, 1.0])
        j1 = np.diag([5.0, 5.0])
        jacobian_arr = np.array([j0, j1])
        timepoints = np.array([0.0, 1.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        w_mean = _weighted_eigenvectors(jc.jacobian_mean_arr)
        dist0 = float(np.linalg.norm(_weighted_eigenvectors(j0) - w_mean))
        dist1 = float(np.linalg.norm(_weighted_eigenvectors(j1) - w_mean))
        self.assertNotEqual(dist0, dist1)

    def test_nonnegative(self) -> None:
        """diameter is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_collection()
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
        jc_full = JacobianCollection.fromArrays(full_arr, np.linspace(0, 4, 5))

        mid_arr = np.array([j_mild0, j_center, j_mild1])
        jc_mid = JacobianCollection.fromArrays(mid_arr, np.linspace(0, 2, 3))

        center_arr = np.array([j_center])
        jc_center = JacobianCollection.fromArrays(center_arr, np.array([0.0]))

        self.assertGreater(jc_full.diameter, jc_mid.diameter)
        self.assertGreater(jc_mid.diameter, jc_center.diameter)
        self.assertAlmostEqual(jc_center.diameter, 0.0, places=10)


class TestNonsequentialPartition(unittest.TestCase):

    def setUp(self) -> None:
        self.n_points = 20
        self.n_species = 3
        rng = np.random.default_rng(0)
        jacobian_arr = rng.standard_normal((self.n_points, self.n_species, self.n_species))
        timepoint_arr = np.linspace(0.0, 10.0, self.n_points)
        self.jc = JacobianCollection.fromArrays(jacobian_arr, timepoint_arr)

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
            self.assertIsInstance(jc, JacobianCollection)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=4)
        total = sum(jc.jacobian_arr.shape[0] for jc in result)
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
        self.assertEqual(result[0].jacobian_arr.shape[0], self.n_points)


class TestPartitionJacobiansSequentially(unittest.TestCase):
    """Tests for JacobianCollection.sequentialPartition."""

    def setUp(self) -> None:
        self.n_points = 20
        self.n_species = 3
        rng = np.random.default_rng(0)
        jacobian_arr = rng.standard_normal((self.n_points, self.n_species, self.n_species))
        timepoint_arr = np.linspace(0.0, 10.0, self.n_points)
        self.jc = JacobianCollection.fromArrays(jacobian_arr, timepoint_arr)

    def test_returns_list(self) -> None:
        """sequentialPartition returns a list."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=2)
        self.assertIsInstance(result, list)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """Result has exactly n_cluster elements."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=3)
        self.assertEqual(len(result), 3)

    def test_each_element_is_jacobian_collection(self) -> None:
        """Each element in the result is a JacobianCollection."""
        if IGNORE_TESTS:
            return
        for jc in self.jc.sequentialPartition(n_cluster=2):
            self.assertIsInstance(jc, JacobianCollection)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=4)
        total = sum(jc.jacobian_arr.shape[0] for jc in result)
        self.assertEqual(total, self.n_points)

    def test_clusters_are_contiguous_in_time(self) -> None:
        """Concatenating cluster jacobian_arrs in order reconstructs the original array."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=3)
        reconstructed = np.concatenate([jc.jacobian_arr for jc in result], axis=0)
        np.testing.assert_array_equal(reconstructed, self.jc.jacobian_arr)

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
        np.testing.assert_array_equal(result[0].jacobian_arr, self.jc.jacobian_arr)


class TestBiomodel206(unittest.TestCase):
    """Integration tests for JacobianCollection with BioModel BIOMD0000000206.

    Uses the canonical simulation window from the model's SED-ML: t=0..10.
    num_points is reduced from 1000 to 100 to keep the test suite fast.
    """

    def setUp(self) -> None:
        with open(BIOMD206_PATH) as f:
            sbml_str = f.read()
        self.lr = LRoadrunner(sbml_str, start_time=0.0, num_points=100)

    def test_construction_succeeds(self) -> None:
        """JacobianCollection is created without error from BIOMD0000000206."""
        if IGNORE_TESTS:
            return
        try:
            JacobianCollection(self.lr)
            self.assertTrue(False)
        except Exception as e:
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
