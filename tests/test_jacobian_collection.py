"""Tests for JacobianCollection class."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jacobian_collection import JacobianCollection  # type: ignore


def _make_collection(n_points: int = 5, n_species: int = 3) -> JacobianCollection:
    """Return a JacobianCollection with deterministic values."""
    rng = np.random.default_rng(42)
    jacobian_arr = rng.standard_normal((n_points, n_species, n_species))
    timepoints = np.linspace(0, 10, n_points)
    return JacobianCollection(jacobian_arr, timepoints)


class TestJacobianCollectionInit(unittest.TestCase):
    """Tests for JacobianCollection.__init__."""

    def test_stores_jacobian_arr(self) -> None:
        """jacobian_arr attribute is stored as-is."""
        jc = _make_collection()
        self.assertEqual(jc.jacobian_arr.shape, (5, 3, 3))

    def test_stores_timepoints(self) -> None:
        """timepoints attribute is stored as-is."""
        jc = _make_collection()
        self.assertEqual(len(jc.timepoints), 5)


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
        jc = JacobianCollection(jacobian_arr, timepoints)
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
        jc = JacobianCollection(np.array([]), np.array([]))
        self.assertEqual(jc.max_cv, 0.0)

    def test_constant_entries_return_zero(self) -> None:
        """Constant Jacobian entries (std=0) produce cv=0 and max_cv=0."""
        jacobian_arr = np.ones((10, 2, 2))
        timepoints = np.linspace(0, 1, 10)
        jc = JacobianCollection(jacobian_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_zero_mean_entries_excluded(self) -> None:
        """Entries with zero mean (which give inf/nan CV) are treated as 0, not inf."""
        # Alternating +1/-1 → mean=0, so CV would be inf/nan
        jacobian_arr = np.array([[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]])
        timepoints = np.array([0.0, 1.0, 2.0, 3.0])
        jc = JacobianCollection(jacobian_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_known_cv_value(self) -> None:
        """max_cv matches hand-computed CV for a simple case."""
        # Single entry varying from 1 to 3 → mean=2, std=~0.816 → CV=~0.408
        values = np.array([1.0, 2.0, 3.0])
        jacobian_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = JacobianCollection(jacobian_arr, timepoints)
        expected_cv = float(np.abs(np.std(values) / np.mean(values)))
        self.assertAlmostEqual(jc.max_cv, expected_cv, places=10)

    def test_max_taken_across_entries(self) -> None:
        """max_cv returns the maximum CV across all Jacobian entries."""
        # Entry [0,0]: constant 1.0 → cv=0; Entry [0,1]: varies → cv>0
        jacobian_arr = np.ones((5, 2, 2))
        jacobian_arr[:, 0, 1] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timepoints = np.linspace(0, 1, 5)
        jc = JacobianCollection(jacobian_arr, timepoints)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_max = float(np.abs(np.std(vals) / np.mean(vals)))
        self.assertAlmostEqual(jc.max_cv, expected_max, places=10)

    def test_nonnegative(self) -> None:
        """max_cv is always non-negative."""
        jc = _make_collection()
        self.assertGreaterEqual(jc.max_cv, 0.0)


if __name__ == "__main__":
    unittest.main()
