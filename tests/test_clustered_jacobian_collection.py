"""Tests for ClusteredJacobianCollection class."""

import os
import sys
import unittest
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jacobian_collection import JacobianCollection  # type: ignore
from clustered_jacobian_collection import ClusteredJacobianCollection  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore

IGNORE_TESTS = False

ANTIMONY_DECAY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10.0; S2 = 0.0
"""

ANTIMONY_FORCED = """
$Xo -> S1; k1*Xo
S1 -> $X1; k2*S1
S1 = 0.0
k1 = 0.1; k2 = 0.2; Xo = 1.0; X1 = 0.0
"""


def _make_cjc(antimony_str: str, n_clusters: int) -> ClusteredJacobianCollection:
    """Return a ClusteredJacobianCollection built from a real Antimony model."""
    lr = LRoadrunner(antimony_str, start_time=0.0, end_time=10.0, num_points=11)
    jc = JacobianCollection(lr)
    n_points = len(jc.timepoint_arr)
    chunk_size = n_points // n_clusters
    jcs = []
    for i in range(n_clusters):
        start = i * chunk_size
        end = start + chunk_size if i < n_clusters - 1 else n_points
        jcs.append(JacobianCollection.fromArrays(
            jc.jacobian_arr[start:end], jc.timepoint_arr[start:end], lr))
    return ClusteredJacobianCollection(jcs)


class TestClusteredJacobianCollectionHeatmaps(unittest.TestCase):
    """Tests for ClusteredJacobianCollection.heatmaps."""

    def test_heatmaps_returns_figure(self) -> None:
        """heatmaps returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        cjc = _make_cjc(ANTIMONY_FORCED, n_clusters=1)
        fig = cjc.heatmaps()
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_heatmaps_one_axes_per_cluster_plus_colorbar(self) -> None:
        """heatmaps produces n heatmap axes plus 1 shared colorbar axes."""
        if IGNORE_TESTS:
            return
        for n_clusters in [1, 2, 3]:
            cjc = _make_cjc(ANTIMONY_DECAY, n_clusters=n_clusters)
            fig = cjc.heatmaps()
            self.assertEqual(len(fig.axes), n_clusters + 1)
            plt.close(fig)

    def test_heatmaps_titles_contain_timepoints(self) -> None:
        """Each heatmap axes title contains the start and end timepoints of its cluster."""
        if IGNORE_TESTS:
            return
        cjc = _make_cjc(ANTIMONY_DECAY, n_clusters=2)
        fig = cjc.heatmaps()
        # First n axes are heatmaps; last is colorbar.
        heatmap_axes = fig.axes[:-1]
        for ax, jc in zip(heatmap_axes, cjc.jacobian_collections):
            t_start = f"{float(jc.timepoint_arr[0]):.3g}"
            t_end = f"{float(jc.timepoint_arr[-1]):.3g}"
            self.assertIn(t_start, ax.get_title())
            self.assertIn(t_end, ax.get_title())
        plt.close(fig)

    def test_heatmaps_colorbar_is_horizontal(self) -> None:
        """The shared colorbar is oriented horizontally."""
        if IGNORE_TESTS:
            return
        cjc = _make_cjc(ANTIMONY_DECAY, n_clusters=2)
        fig = cjc.heatmaps()
        cbar_ax = fig.axes[-1]
        # A horizontal colorbar axes is wider than it is tall.
        bbox = cbar_ax.get_position()
        self.assertGreater(bbox.width, bbox.height)
        plt.close(fig)

    def test_heatmaps_colorbar_does_not_overlap_heatmaps(self) -> None:
        """The colorbar axes does not overlap any heatmap axes."""
        if IGNORE_TESTS:
            return
        cjc = _make_cjc(ANTIMONY_DECAY, n_clusters=2)
        fig = cjc.heatmaps()
        cbar_ax = fig.axes[-1]
        cbar_bbox = cbar_ax.get_position()
        for ax in fig.axes[:-1]:
            hm_bbox = ax.get_position()
            overlaps = (
                cbar_bbox.x0 < hm_bbox.x1 and cbar_bbox.x1 > hm_bbox.x0
                and cbar_bbox.y0 < hm_bbox.y1 and cbar_bbox.y1 > hm_bbox.y0
            )
            self.assertFalse(overlaps)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
