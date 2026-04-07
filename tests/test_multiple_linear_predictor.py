"""Tests for MultipleLinearPredictor class."""

import os
import sys
import unittest
import numpy as np  # type: ignore
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jacobian_collection import JacobianCollection  # type: ignore
from clustered_jacobian_collection import ClusteredJacobianCollection  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore
from multiple_linear_predictor import MultipleLinearPredictor  # type: ignore

IGNORE_TESTS = False

# Simple Antimony model: first-order decay with boundary-species forcing.
ANTIMONY_FORCED = """
$Xo -> S1; k1*Xo
S1 -> $X1; k2*S1
S1 = 0.0
k1 = 0.1; k2 = 0.2; Xo = 1.0; X1 = 0.0
"""

# Simple 2-species decay model.
ANTIMONY_DECAY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10.0; S2 = 0.0
"""


def _make_lroadrunner(antimony_str: str) -> LRoadrunner:
    """Return a real LRoadrunner for the given Antimony model."""
    return LRoadrunner(antimony_str, start_time=0.0, end_time=10.0, num_points=11)


class TestMultipleLinearPredictorInit(unittest.TestCase):
    """Tests for MultipleLinearPredictor.__init__."""

    def test_stores_clustered_jacobian_collection(self) -> None:
        """clustered_jacobian_collection attribute is stored."""
        if IGNORE_TESTS:
            return
        lr = MagicMock(spec=LRoadrunner)
        lr.makeJacobians.return_value = (
            np.full((3, 1, 1), -0.2), np.linspace(0.0, 10.0, 3)
        )
        jc = JacobianCollection(lr)
        cjc = ClusteredJacobianCollection([jc])
        predictor = MultipleLinearPredictor(cjc, lr)
        self.assertIs(predictor.clustered_jacobian_collection, cjc)

    def test_stores_l_roadrunner(self) -> None:
        """l_roadrunner attribute is stored."""
        if IGNORE_TESTS:
            return
        lr = MagicMock(spec=LRoadrunner)
        lr.makeJacobians.return_value = (
            np.full((3, 1, 1), -0.2), np.linspace(0.0, 10.0, 3)
        )
        jc = JacobianCollection(lr)
        cjc = ClusteredJacobianCollection([jc])
        predictor = MultipleLinearPredictor(cjc, lr)
        self.assertIs(predictor.l_roadrunner, lr)


class TestMultipleLinearPredictorPredict(unittest.TestCase):
    """Tests for MultipleLinearPredictor.predict."""

    def _make_predictor_from_model(self, antimony_str: str,
                                    n_clusters: int) -> MultipleLinearPredictor:
        """Return a MultipleLinearPredictor built from a real model."""
        lr = _make_lroadrunner(antimony_str)
        jc = JacobianCollection(lr)
        n_points = len(jc.timepoint_arr)
        chunk_size = n_points // n_clusters
        jcs = []
        for i in range(n_clusters):
            start = i * chunk_size
            end = start + chunk_size if i < n_clusters - 1 else n_points
            chunk_jac = jc.jacobian_arr[start:end]
            chunk_t = jc.timepoint_arr[start:end]
            jcs.append(JacobianCollection.fromArrays(chunk_jac, chunk_t, lr))
        cjc = ClusteredJacobianCollection(jcs)
        return MultipleLinearPredictor(cjc, lr)

    def test_predict_shape_single_cluster(self) -> None:
        """predict returns shape (1, n_species) for a single-cluster collection."""
        if IGNORE_TESTS:
            return
        predictor = self._make_predictor_from_model(ANTIMONY_FORCED, n_clusters=1)
        result = predictor.predict()
        self.assertEqual(result.shape[0], 1)
        self.assertGreater(result.shape[1], 0)

    def test_predict_shape_multiple_clusters(self) -> None:
        """predict returns shape (n_clusters, n_species)."""
        if IGNORE_TESTS:
            return
        n_clusters = 3
        predictor = self._make_predictor_from_model(ANTIMONY_DECAY, n_clusters=n_clusters)
        result = predictor.predict()
        self.assertEqual(result.shape[0], n_clusters)

    def test_predict_finite_values(self) -> None:
        """predict returns finite concentration values."""
        if IGNORE_TESTS:
            return
        predictor = self._make_predictor_from_model(ANTIMONY_FORCED, n_clusters=2)
        result = predictor.predict()
        self.assertTrue(np.all(np.isfinite(result)))

    def test_predict_non_negative_concentrations(self) -> None:
        """predict returns non-negative concentrations for a decay model starting at 0."""
        if IGNORE_TESTS:
            return
        predictor = self._make_predictor_from_model(ANTIMONY_FORCED, n_clusters=2)
        result = predictor.predict()
        self.assertTrue(np.all(result >= -1e-6))

    def test_predict_approaches_steady_state(self) -> None:
        """For ANTIMONY_FORCED, predictions should increase toward steady state 0.5."""
        if IGNORE_TESTS:
            return
        n_clusters = 5
        predictor = self._make_predictor_from_model(ANTIMONY_FORCED, n_clusters=n_clusters)
        result = predictor.predict()
        self.assertTrue(result[-1, 0] > result[0, 0])
        self.assertLess(result[-1, 0], 0.6)


if __name__ == "__main__":
    unittest.main()
