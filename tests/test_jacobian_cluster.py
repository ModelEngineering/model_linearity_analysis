"""
Tests for JacobianCluster class.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jacobian_cluster import JacobianCluster  # type: ignore


def _make_jacobian(n: int = 2, seed: int = 0) -> np.ndarray:
    """Return a deterministic (n, n) Jacobian array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n))


class TestJacobianClusterInit(unittest.TestCase):
    """Tests for JacobianCluster.__init__."""

    def test_new_cluster_has_no_times(self) -> None:
        """A freshly constructed cluster contains no timepoints."""
        cluster = JacobianCluster()
        self.assertEqual(cluster.getTimes(), set())

    def test_new_cluster_has_empty_jacobians(self) -> None:
        """A freshly constructed cluster stores no Jacobians."""
        cluster = JacobianCluster()
        self.assertEqual(len(cluster._jacobians), 0)


class TestJacobianClusterAdd(unittest.TestCase):
    """Tests for JacobianCluster.add."""

    def test_add_single_entry(self) -> None:
        """Adding one entry results in one stored Jacobian and one timepoint."""
        cluster = JacobianCluster()
        cluster.add(_make_jacobian(), 1.0)
        self.assertEqual(len(cluster._jacobians), 1)
        self.assertEqual(len(cluster._times), 1)

    def test_add_multiple_entries(self) -> None:
        """Adding multiple entries accumulates all of them."""
        cluster = JacobianCluster()
        for t in [0.0, 1.0, 2.0]:
            cluster.add(_make_jacobian(seed=int(t)), t)
        self.assertEqual(len(cluster._jacobians), 3)
        self.assertEqual(len(cluster._times), 3)

    def test_add_stores_jacobian_values(self) -> None:
        """The Jacobian array stored is identical to the one passed in."""
        cluster = JacobianCluster()
        jacobian_arr = _make_jacobian()
        cluster.add(jacobian_arr, 5.0)
        np.testing.assert_array_equal(cluster._jacobians[0], jacobian_arr)

    def test_add_stores_timepoint_value(self) -> None:
        """The timepoint stored is identical to the one passed in."""
        cluster = JacobianCluster()
        cluster.add(_make_jacobian(), 3.14)
        self.assertEqual(cluster._times[0], 3.14)

    def test_add_preserves_order(self) -> None:
        """Jacobians and timepoints are stored in insertion order."""
        cluster = JacobianCluster()
        times = [0.5, 1.5, 2.5]
        for t in times:
            cluster.add(_make_jacobian(seed=int(t * 10)), t)
        self.assertEqual(cluster._times, times)

    def test_add_duplicate_timepoints(self) -> None:
        """Duplicate timepoints are stored independently (not deduplicated in storage)."""
        cluster = JacobianCluster()
        cluster.add(_make_jacobian(seed=0), 1.0)
        cluster.add(_make_jacobian(seed=1), 1.0)
        self.assertEqual(len(cluster._jacobians), 2)
        self.assertEqual(len(cluster._times), 2)


class TestJacobianClusterGetTimes(unittest.TestCase):
    """Tests for JacobianCluster.getTimes."""

    def test_returns_set(self) -> None:
        """getTimes returns a set."""
        cluster = JacobianCluster()
        self.assertIsInstance(cluster.getTimes(), set)

    def test_empty_cluster_returns_empty_set(self) -> None:
        """getTimes on an empty cluster returns an empty set."""
        cluster = JacobianCluster()
        self.assertEqual(cluster.getTimes(), set())

    def test_returns_all_timepoints(self) -> None:
        """getTimes contains every timepoint that was added."""
        cluster = JacobianCluster()
        times = [0.0, 1.0, 2.0, 5.0]
        for t in times:
            cluster.add(_make_jacobian(seed=int(t)), t)
        self.assertEqual(cluster.getTimes(), set(times))

    def test_deduplicates_timepoints(self) -> None:
        """getTimes returns unique timepoints even when duplicates were added."""
        cluster = JacobianCluster()
        cluster.add(_make_jacobian(seed=0), 1.0)
        cluster.add(_make_jacobian(seed=1), 1.0)
        cluster.add(_make_jacobian(seed=2), 2.0)
        self.assertEqual(cluster.getTimes(), {1.0, 2.0})

    def test_does_not_include_unadded_timepoints(self) -> None:
        """getTimes does not contain timepoints that were never added."""
        cluster = JacobianCluster()
        cluster.add(_make_jacobian(), 1.0)
        self.assertNotIn(2.0, cluster.getTimes())


if __name__ == "__main__":
    unittest.main()
