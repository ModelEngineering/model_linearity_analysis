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

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMD_PATH = os.path.join(
    cn.BIOMODELS_DIR, "BIOMD0000000038", "BIOMD0000000038_url.xml"
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
        jc = _make_collection()
        self.assertEqual(jc.jacobian_arr.shape, (5, 3, 3))

    def test_stores_timepoints(self) -> None:
        """timepoints attribute is stored as-is."""
        jc = _make_collection()
        self.assertEqual(len(jc.timepoint_arr), 5)


class TestGetTimes(unittest.TestCase):
    """Tests for JacobianCollection.getTimes."""

    def test_returns_set(self) -> None:
        """getTimes returns a set."""
        jc = _make_collection()
        self.assertIsInstance(jc.getTimes(), set)

    def test_unique_timepoints(self) -> None:
        """getTimes returns the unique set of timepoints."""
        timepoints = np.array([0.0, 1.0, 1.0, 2.0])
        jacobian_arr = np.zeros((4, 2, 2))
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertEqual(jc.getTimes(), {0.0, 1.0, 2.0})

    def test_all_unique_timepoints(self) -> None:
        """getTimes returns all timepoints when none are duplicated."""
        jc = _make_collection(n_points=5)
        self.assertEqual(len(jc.getTimes()), 5)


class TestMaxCV(unittest.TestCase):
    """Tests for JacobianCollection.max_cv property."""

    def test_returns_float(self) -> None:
        """max_cv returns a float."""
        jc = _make_collection()
        self.assertIsInstance(jc.max_cv, float)

    def test_empty_array_returns_zero(self) -> None:
        """max_cv returns 0.0 for an empty jacobian_arr."""
        jc = _make_collection_from_arrays(np.array([]), np.array([]))
        self.assertEqual(jc.max_cv, 0.0)

    def test_constant_entries_return_zero(self) -> None:
        """Constant Jacobian entries (std=0) produce cv=0 and max_cv=0."""
        jacobian_arr = np.ones((10, 2, 2))
        timepoints = np.linspace(0, 1, 10)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_zero_mean_entries_excluded(self) -> None:
        """Entries with zero mean (which give inf/nan CV) are treated as 0, not inf."""
        # Alternating +1/-1 → mean=0, so CV would be inf/nan
        jacobian_arr = np.array([[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]])
        timepoints = np.array([0.0, 1.0, 2.0, 3.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_known_cv_value(self) -> None:
        """max_cv matches hand-computed CV for a simple case."""
        # Single entry varying from 1 to 3 → mean=2, std=~0.816 → CV=~0.408
        values = np.array([1.0, 2.0, 3.0])
        jacobian_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        expected_cv = float(np.abs(np.std(values) / np.mean(values)))
        self.assertAlmostEqual(jc.max_cv, expected_cv, places=10)

    def test_max_taken_across_entries(self) -> None:
        """max_cv returns the maximum CV across all Jacobian entries."""
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
        jc = _make_collection()
        self.assertGreaterEqual(jc.max_cv, 0.0)


class TestCalculateDeviation(unittest.TestCase):
    """Tests for JacobianCollection._calculateDeviation."""

    def test_returns_1d_array_of_correct_length(self) -> None:
        """Result has shape (num_points,)."""
        jc = _make_collection(n_points=5, n_species=3)
        result = jc._calculateDeviation()
        self.assertEqual(result.shape, (5,))

    def test_identical_jacobians_give_zero_deviation(self) -> None:
        """When all Jacobians are identical the centroid equals every Jacobian, so deviation is 0."""
        jacobian_arr = np.ones((4, 2, 2))
        timepoints = np.linspace(0, 3, 4)
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(4))

    def test_known_value_1x1(self) -> None:
        """Hand-computed deviation for a 1×1 Jacobian with two timepoints."""
        # centroid = 3.0; each deviation = |J - 3| / 3 = 1/3
        jacobian_arr = np.array([[[2.0]], [[4.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        expected = np.array([1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_centroid_entry_produces_no_nan_or_inf(self) -> None:
        """Entries whose centroid is 0 contribute 0 to the deviation (no inf/nan)."""
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
        jacobian_arr = np.zeros((3, 2, 2))
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_single_timepoint_gives_zero_deviation(self) -> None:
        """With one timepoint the centroid equals the Jacobian, so deviation is 0."""
        jacobian_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_collection_from_arrays(jacobian_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(1))

    def test_nonnegative(self) -> None:
        """Deviation is always non-negative."""
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
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()

    def test_creates_two_axes(self) -> None:
        """plot creates a figure with exactly two subplots."""
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes), 2)

    def test_first_axis_title(self) -> None:
        """First subplot title describes the Jacobian deviation."""
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(
            plt.gcf().axes[0].get_title(),
            "Normalized Distance of Jacobian to Centroid",
        )

    def test_second_axis_title(self) -> None:
        """Second subplot title describes the species timecourse."""
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(plt.gcf().axes[1].get_title(), "Species Timecourse")

    def test_first_axis_has_one_line(self) -> None:
        """First subplot contains exactly one line (the deviation curve)."""
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes[0].lines), 1)

    def test_second_axis_line_count_matches_species(self) -> None:
        """Second subplot has one line per floating species."""
        n_species = len(self.collection._l_roadrunner.roadrunner.getFloatingSpeciesIds())
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes[1].lines), n_species)

    def test_second_axis_legend_contains_species_ids(self) -> None:
        """Second subplot legend labels match the floating species IDs."""
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
        with open(BIOMD_PATH) as f:
            sbml_str = f.read()
        lr = LRoadrunner(sbml_str, start_time=0.0, end_time=0.002, num_points=600)
        collection = JacobianCollection(lr)
        with patch("matplotlib.pyplot.show"):
            collection.plot()


if __name__ == "__main__":
    unittest.main()
