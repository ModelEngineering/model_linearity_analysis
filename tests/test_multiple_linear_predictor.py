"""Tests for MultipleLinearPredictor class."""

import os
import sys
import unittest
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jacobian_collection import JacobianCollection  # type: ignore
from clustered_jacobian_collection import ClusteredJacobianCollection  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore
from multiple_linear_predictor import MultipleLinearPredictor  # type: ignore

IGNORE_TESTS = False

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)

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


def _make_predictor_from_model(antimony_str: str,
                                n_clusters: int) -> MultipleLinearPredictor:
    """Return a MultipleLinearPredictor built from a real Antimony model."""
    lr = _make_lroadrunner(antimony_str)
    jc = JacobianCollection(lr)
    n_points = len(jc.timepoint_arr)
    chunk_size = n_points // n_clusters
    jcs = []
    for i in range(n_clusters):
        start = i * chunk_size
        end = start + chunk_size if i < n_clusters - 1 else n_points
        jcs.append(JacobianCollection.fromArrays(
            jc.jacobian_arr[start:end], jc.timepoint_arr[start:end], lr))
    cjc = ClusteredJacobianCollection(jcs)
    return MultipleLinearPredictor(cjc, lr)


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

    def test_predict_shape_single_cluster(self) -> None:
        """predict returns shape (1, n_species) for a single-cluster collection."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=1)
        result = predictor.predict()
        self.assertEqual(result.shape[0], 1)
        self.assertGreater(result.shape[1], 0)

    def test_predict_shape_multiple_clusters(self) -> None:
        """predict returns shape (n_clusters, n_species)."""
        if IGNORE_TESTS:
            return
        n_clusters = 3
        predictor = _make_predictor_from_model(ANTIMONY_DECAY, n_clusters=n_clusters)
        result = predictor.predict()
        self.assertEqual(result.shape[0], n_clusters)

    def test_predict_finite_values(self) -> None:
        """predict returns finite concentration values."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=2)
        result = predictor.predict()
        self.assertTrue(np.all(np.isfinite(result)))

    def test_predict_non_negative_concentrations(self) -> None:
        """predict returns non-negative concentrations for a decay model starting at 0."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=2)
        result = predictor.predict()
        self.assertTrue(np.all(result >= -1e-6))

    def test_predict_approaches_steady_state(self) -> None:
        """For ANTIMONY_FORCED, predictions should increase toward steady state 0.5."""
        if IGNORE_TESTS:
            return
        n_clusters = 5
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=n_clusters)
        result = predictor.predict()
        self.assertTrue(result[-1, 0] > result[0, 0])
        self.assertLess(result[-1, 0], 0.6)


class TestMultipleLinearPredictorPlot(unittest.TestCase):
    """Tests for MultipleLinearPredictor.plot."""

    def test_plot_returns_figure(self) -> None:
        """plot returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=1)
        fig = predictor.plot()
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_plot_creates_axes_when_none(self) -> None:
        """plot creates a new figure with one axes when ax is not supplied."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=1)
        fig = predictor.plot(ax=None)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_plot_uses_provided_axes(self) -> None:
        """plot draws on the supplied axes and returns its parent figure."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=1)
        external_fig, external_ax = plt.subplots()
        returned_fig = predictor.plot(ax=external_ax)
        self.assertIs(returned_fig, external_fig)
        plt.close(external_fig)

    def test_plot_no_vertical_lines_single_cluster(self) -> None:
        """With one cluster there are no vertical boundary lines."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_clusters=1)
        fig = predictor.plot()
        ax = fig.axes[0]
        vlines = [l for l in ax.lines
                  if len(l.get_xdata()) == 2
                  and l.get_xdata()[0] == l.get_xdata()[1]]
        self.assertEqual(len(vlines), 0)
        plt.close(fig)

    def test_plot_vertical_lines_for_multiple_clusters(self) -> None:
        """With n clusters there are n-1 vertical boundary lines."""
        if IGNORE_TESTS:
            return
        n_clusters = 3
        predictor = _make_predictor_from_model(ANTIMONY_DECAY, n_clusters=n_clusters)
        fig = predictor.plot()
        ax = fig.axes[0]
        vlines = [l for l in ax.lines
                  if len(l.get_xdata()) == 2
                  and l.get_xdata()[0] == l.get_xdata()[1]]
        self.assertEqual(len(vlines), n_clusters - 1)
        plt.close(fig)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestMultipleLinearPredictorPlotWithBioModels(unittest.TestCase):
    """Tests for MultipleLinearPredictor.plot using real BioModels data (BIOMD8)."""

    BIOMD8_DIR = os.path.join(BIOMODELS_DIR, "BIOMD0000000008")

    def _skip_if_no_endtime(self) -> None:
        """Skip if BIOMD8 is not in the precomputed end-time table."""
        if "BIOMD0000000008" not in LRoadrunner.endtime_dct:
            self.skipTest("BIOMD0000000008 not in endtime_dct")

    def _make_predictor(self, n_cluster: int) -> MultipleLinearPredictor:
        """Return a MultipleLinearPredictor for BIOMD8 with the given cluster count."""
        from linear_analyzer import LinearAnalyzer  # type: ignore
        cjc = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(
            self.BIOMD8_DIR, n_cluster=n_cluster, is_report=False,
            end_time=20)
        return MultipleLinearPredictor(cjc, cjc.l_roadrunner)

    def test_plot_biomd8_one_cluster(self) -> None:
        """plot returns a Figure for BIOMD8 with 1 cluster."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor = self._make_predictor(n_cluster=1)
        fig = predictor.plot()
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_plot_biomd8_three_clusters(self) -> None:
        """plot for BIOMD8 with 3 clusters has exactly 2 vertical boundary lines."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor = self._make_predictor(n_cluster=3)
        fig = predictor.plot()
        self.assertIsInstance(fig, mfigure.Figure)
        ax = fig.axes[0]
        vlines = [l for l in ax.lines
                if len(l.get_xdata()) == 2
                and l.get_xdata()[0] == l.get_xdata()[1]]
        self.assertEqual(len(vlines), 2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
