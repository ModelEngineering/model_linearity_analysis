"""Tests for the Score class."""

import os
import sys
import unittest
from unittest.mock import patch
import matplotlib  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import (Score, DEFAULT_AGGREGATIONS,  # type: ignore
        makePercentileAggregation, DEFAULT_PERCENTILES,
        AGGREGATION_PERCENTILE_95, AGGREGATION_PERCENTILE_50, AGGREGATION_PERCENTILE_75)

IGNORE_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

TIMES = [0.0, 1.0, 2.0]
TRUE_DF = pd.DataFrame({'A': [1.0, 2.0, 4.0], 'B': [2.0, 4.0, 8.0]}, index=TIMES)
PRED_DF = 2.0 * TRUE_DF  # ARE = (2x - x) / x = 1.0 everywhere


def _make_score(description: str = "test") -> Score:
    """Returns a Score with one test result where ARE == 1.0 everywhere."""
    score = Score(description)
    score.addTestResult(TRUE_DF, PRED_DF)
    return score


class TestScoreInit(unittest.TestCase):
    """Tests for Score.__init__."""

    def test_description_stored(self) -> None:
        """Provided description is stored on the instance."""
        if IGNORE_TESTS:
            return
        score = Score("my description")
        self.assertEqual(score.description, "my description")

    def test_default_description_is_empty(self) -> None:
        """Default description is an empty string."""
        if IGNORE_TESTS:
            return
        score = Score()
        self.assertEqual(score.description, "")

    def test_are_dfs_initially_empty(self) -> None:
        """No test results are stored on construction."""
        if IGNORE_TESTS:
            return
        score = Score()
        self.assertEqual(len(score._are_dfs), 0)


class TestAddTestResult(unittest.TestCase):
    """Tests for Score.addTestResult."""

    def test_increments_are_dfs(self) -> None:
        """Calling addTestResult once stores one ARE DataFrame."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        self.assertEqual(len(score._are_dfs), 1)

    def test_are_values_correct(self) -> None:
        """ARE is computed as (prediction - true) / true."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        np.testing.assert_allclose(score._are_dfs[0].values, 1.0)

    def test_are_nan_when_true_zero(self) -> None:
        """ARE is NaN for entries where the true value is zero."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'A': [0.0, 1.0]}, index=[0.0, 1.0])
        pred_df = pd.DataFrame({'A': [1.0, 2.0]}, index=[0.0, 1.0])
        score = Score()
        score.addTestResult(true_df, pred_df)
        self.assertTrue(np.isnan(score._are_dfs[0].values[0, 0]))
        self.assertFalse(np.isnan(score._are_dfs[0].values[1, 0]))

    def test_multiple_results_accumulated(self) -> None:
        """Calling addTestResult twice stores two ARE DataFrames."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        score.addTestResult(TRUE_DF, PRED_DF)
        self.assertEqual(len(score._are_dfs), 2)

    def test_are_index_matches_true_timecourse(self) -> None:
        """ARE DataFrame index matches the timepoints of the true timecourse."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        np.testing.assert_array_equal(score._are_dfs[0].index, TIMES)

    def test_are_columns_match_species(self) -> None:
        """ARE DataFrame columns match the species names of the true timecourse."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        self.assertEqual(list(score._are_dfs[0].columns), ['A', 'B'])


class TestAggregateByTime(unittest.TestCase):
    """Tests for Score.aggregateByTime."""

    def setUp(self) -> None:
        self.score = _make_score()

    def test_returns_dataframe(self) -> None:
        """aggregateByTime returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime()
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_is_species_names(self) -> None:
        """Index of result contains the species names."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["mean"])
        self.assertEqual(list(result.index), ['A', 'B'])

    def test_columns_are_aggregation_names(self) -> None:
        """Columns of result match the requested aggregation names."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["mean", "min"])
        self.assertIn("mean", result.columns)
        self.assertIn("min", result.columns)

    def test_mean_correct_for_constant_are(self) -> None:
        """Mean aggregation returns 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["mean"])
        np.testing.assert_allclose(result["mean"].values, 1.0)

    def test_min_correct_for_constant_are(self) -> None:
        """Min aggregation returns 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["min"])
        np.testing.assert_allclose(result["min"].values, 1.0)

    def test_max_correct_for_constant_are(self) -> None:
        """Max aggregation returns 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["max"])
        np.testing.assert_allclose(result["max"].values, 1.0)

    def test_median_correct_for_constant_are(self) -> None:
        """Median aggregation returns 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["median"])
        np.testing.assert_allclose(result["median"].values, 1.0)

    def test_percentile_correct_for_constant_are(self) -> None:
        """p50 aggregation returns 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByTime(["p50"])
        np.testing.assert_allclose(result["p50"].values, 1.0)

    def test_varying_are_values(self) -> None:
        """Aggregations are correct when ARE varies across time."""
        if IGNORE_TESTS:
            return
        # ARE_A = [0, 1, 2] → mean=1, min=0, max=2
        true_df = pd.DataFrame({'A': [1.0, 1.0, 1.0]}, index=[0.0, 1.0, 2.0])
        pred_df = pd.DataFrame({'A': [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        score = Score()
        score.addTestResult(true_df, pred_df)
        result = score.aggregateByTime(["mean", "min", "max"])
        self.assertAlmostEqual(result.loc['A', 'mean'], 1.0)
        self.assertAlmostEqual(result.loc['A', 'min'], 0.0)
        self.assertAlmostEqual(result.loc['A', 'max'], 2.0)

    def test_empty_score_returns_empty_dataframe(self) -> None:
        """Returns an empty DataFrame when no test results have been added."""
        if IGNORE_TESTS:
            return
        score = Score()
        result = score.aggregateByTime()
        self.assertTrue(result.empty)

    def test_unknown_aggregation_raises(self) -> None:
        """Raises ValueError for an unrecognised aggregation name."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.score.aggregateByTime(["unknown"])

    def test_multiple_test_results_combined(self) -> None:
        """Aggregation is computed across all added test results."""
        if IGNORE_TESTS:
            return
        # First result: ARE_A = [0, 1, 2]; second result: ARE_A = [3, 4, 5]
        # Combined: mean = 2.5
        true_df = pd.DataFrame({'A': [1.0, 1.0, 1.0]}, index=[0.0, 1.0, 2.0])
        pred1_df = pd.DataFrame({'A': [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        pred2_df = pd.DataFrame({'A': [4.0, 5.0, 6.0]}, index=[0.0, 1.0, 2.0])
        score = Score()
        score.addTestResult(true_df, pred1_df)
        score.addTestResult(true_df, pred2_df)
        result = score.aggregateByTime(["mean"])
        self.assertAlmostEqual(result.loc['A', 'mean'], 2.5)


class TestAggregateBySpecies(unittest.TestCase):
    """Tests for Score.aggregateBySpecies."""

    def setUp(self) -> None:
        self.score = _make_score()

    def test_returns_dataframe(self) -> None:
        """aggregateBySpecies returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateBySpecies()
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_is_timepoints(self) -> None:
        """Index of result contains the timepoints."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateBySpecies(["mean"])
        np.testing.assert_array_equal(result.index, TIMES)

    def test_columns_are_aggregation_names(self) -> None:
        """Columns of result match the requested aggregation names."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateBySpecies(["mean", "max"])
        self.assertIn("mean", result.columns)
        self.assertIn("max", result.columns)

    def test_mean_correct_for_constant_are(self) -> None:
        """Mean aggregation returns 1.0 at every timepoint when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateBySpecies(["mean"])
        np.testing.assert_allclose(result["mean"].values, 1.0)

    def test_varying_are_across_species(self) -> None:
        """Aggregations are correct when ARE varies across species."""
        if IGNORE_TESTS:
            return
        # ARE_A=0.0, ARE_B=2.0 at t=0.0 → mean=1.0, min=0.0, max=2.0
        true_df = pd.DataFrame({'A': [1.0], 'B': [1.0]}, index=[0.0])
        pred_df = pd.DataFrame({'A': [1.0], 'B': [3.0]}, index=[0.0])
        score = Score()
        score.addTestResult(true_df, pred_df)
        result = score.aggregateBySpecies(["mean", "min", "max"])
        self.assertAlmostEqual(result.loc[0.0, 'mean'], 1.0)
        self.assertAlmostEqual(result.loc[0.0, 'min'], 0.0)
        self.assertAlmostEqual(result.loc[0.0, 'max'], 2.0)

    def test_empty_score_returns_empty_dataframe(self) -> None:
        """Returns an empty DataFrame when no test results have been added."""
        if IGNORE_TESTS:
            return
        score = Score()
        result = score.aggregateBySpecies()
        self.assertTrue(result.empty)

    def test_nan_are_excluded_from_aggregation(self) -> None:
        """NaN ARE entries (from zero true values) are ignored by aggregations."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'A': [0.0], 'B': [1.0]}, index=[0.0])
        pred_df = pd.DataFrame({'A': [1.0], 'B': [2.0]}, index=[0.0])
        score = Score()
        score.addTestResult(true_df, pred_df)
        # ARE_A=NaN, ARE_B=1.0 → nanmean at t=0 = 1.0
        result = score.aggregateBySpecies(["mean"])
        self.assertAlmostEqual(result.loc[0.0, 'mean'], 1.0)


class TestPlotTime(unittest.TestCase):
    """Tests for Score.plotTime."""

    def setUp(self) -> None:
        self.score = _make_score()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_figure(self) -> None:
        """plotTime returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotTime()
        self.assertIsInstance(fig, mfigure.Figure)

    def test_x_axis_label(self) -> None:
        """x-axis is labelled 'Time'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotTime()
        self.assertEqual(fig.axes[0].get_xlabel(), "Time")

    def test_y_axis_label(self) -> None:
        """y-axis is labelled 'ARE'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotTime()
        self.assertEqual(fig.axes[0].get_ylabel(), "ARE")

    def test_title_contains_description(self) -> None:
        """Plot title includes the Score description."""
        if IGNORE_TESTS:
            return
        score = Score("my test")
        score.addTestResult(TRUE_DF, PRED_DF)
        with patch("matplotlib.pyplot.show"):
            fig = score.plotTime()
        self.assertIn("my test", fig.axes[0].get_title())

    def test_one_line_per_aggregation(self) -> None:
        """One line is drawn for each aggregation function."""
        if IGNORE_TESTS:
            return
        aggregations = ["mean", "min", "max"]
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotTime(aggregations=aggregations)
        self.assertEqual(len(fig.axes[0].lines), len(aggregations))


class TestPlotSpecies(unittest.TestCase):
    """Tests for Score.plotSpecies."""

    def setUp(self) -> None:
        self.score = _make_score()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_figure(self) -> None:
        """plotSpecies returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotSpecies()
        self.assertIsInstance(fig, mfigure.Figure)

    def test_x_axis_label(self) -> None:
        """x-axis is labelled 'Species'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotSpecies()
        self.assertEqual(fig.axes[0].get_xlabel(), "Species")

    def test_y_axis_label(self) -> None:
        """y-axis is labelled 'ARE'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotSpecies()
        self.assertEqual(fig.axes[0].get_ylabel(), "ARE")

    def test_title_contains_description(self) -> None:
        """Plot title includes the Score description."""
        if IGNORE_TESTS:
            return
        score = Score("bar test")
        score.addTestResult(TRUE_DF, PRED_DF)
        with patch("matplotlib.pyplot.show"):
            fig = score.plotSpecies()
        self.assertIn("bar test", fig.axes[0].get_title())

    def test_bars_drawn(self) -> None:
        """Bar chart contains at least one patch per species per aggregation."""
        if IGNORE_TESTS:
            return
        aggregations = ["mean", "min"]
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotSpecies(aggregations=aggregations)
        n_patches = len(fig.axes[0].patches)
        # 2 species × 2 aggregations = 4 bars minimum
        self.assertGreaterEqual(n_patches, len(TRUE_DF.columns) * len(aggregations))


class TestMakePercentileAggregation(unittest.TestCase):
    """Tests for makePercentileAggregation helper."""

    def test_integer_percentile(self) -> None:
        """Whole-number percentile produces a compact label with no trailing zero."""
        if IGNORE_TESTS:
            return
        self.assertEqual(makePercentileAggregation(50.0), "p50")

    def test_fractional_percentile(self) -> None:
        """Fractional percentile preserves the decimal portion."""
        if IGNORE_TESTS:
            return
        self.assertEqual(makePercentileAggregation(99.5), "p99.5")

    def test_constants_use_correct_labels(self) -> None:
        """Module-level percentile constants match expected label strings."""
        if IGNORE_TESTS:
            return
        self.assertEqual(AGGREGATION_PERCENTILE_95, "p95")
        self.assertEqual(AGGREGATION_PERCENTILE_50, "p50")
        self.assertEqual(AGGREGATION_PERCENTILE_75, "p75")


class TestAggregateByPercentile(unittest.TestCase):
    """Tests for Score.aggregateByPercentile."""

    def setUp(self) -> None:
        self.score = _make_score()

    def test_returns_dataframe(self) -> None:
        """aggregateByPercentile returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByPercentile()
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_is_species_names(self) -> None:
        """Index contains species names."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByPercentile([50.0])
        self.assertEqual(list(result.index), ['A', 'B'])

    def test_columns_are_percentile_labels(self) -> None:
        """Columns match the percentile labels produced by makePercentileAggregation."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByPercentile([50.0, 75.0, 0.95])
        self.assertIn("p50", result.columns)
        self.assertIn("p75", result.columns)
        self.assertIn("p95", result.columns)

    def test_constant_are_returns_are_value(self) -> None:
        """All percentiles equal 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByPercentile([50.0, 75.0, 0.95])
        np.testing.assert_allclose(result.values, 1.0)

    def test_percentile_ordering(self) -> None:
        """Lower percentile returns a lower or equal value than a higher percentile."""
        if IGNORE_TESTS:
            return
        # ARE_A = [0, 1, 2, 3, 4] → p50 < p95
        true_df = pd.DataFrame({'A': [1.0] * 5}, index=range(5))
        pred_df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=range(5))
        score = Score()
        score.addTestResult(true_df, pred_df)
        result = score.aggregateByPercentile([50.0, 95.0])
        self.assertLessEqual(result.loc['A', 'p50'], result.loc['A', 'p95'])

    def test_p50_equals_median(self) -> None:
        """p50 aggregation matches the median aggregation."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'A': [1.0, 1.0, 1.0]}, index=[0.0, 1.0, 2.0])
        pred_df = pd.DataFrame({'A': [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        score = Score()
        score.addTestResult(true_df, pred_df)
        p50 = score.aggregateByPercentile([50.0]).loc['A', 'p50']
        median = score.aggregateByTime(["median"]).loc['A', 'median']
        self.assertAlmostEqual(p50, median)

    def test_empty_score_returns_empty_dataframe(self) -> None:
        """Returns an empty DataFrame when no test results have been added."""
        if IGNORE_TESTS:
            return
        score = Score()
        result = score.aggregateByPercentile()
        self.assertTrue(result.empty)

    def test_default_percentiles(self) -> None:
        """Default call produces columns for DEFAULT_PERCENTILES."""
        if IGNORE_TESTS:
            return
        result = self.score.aggregateByPercentile()
        expected_cols = [makePercentileAggregation(p) for p in DEFAULT_PERCENTILES]
        self.assertEqual(list(result.columns), expected_cols)


class TestPlotPercentile(unittest.TestCase):
    """Tests for Score.plotPercentile."""

    def setUp(self) -> None:
        self.score = _make_score()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_figure(self) -> None:
        """plotPercentile returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotPercentile()
        self.assertIsInstance(fig, mfigure.Figure)

    def test_x_axis_label(self) -> None:
        """x-axis is labelled 'Species'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotPercentile()
        self.assertEqual(fig.axes[0].get_xlabel(), "Species")

    def test_y_axis_label(self) -> None:
        """y-axis is labelled 'ARE'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotPercentile()
        self.assertEqual(fig.axes[0].get_ylabel(), "ARE")

    def test_title_contains_description(self) -> None:
        """Plot title includes the Score description."""
        if IGNORE_TESTS:
            return
        score = Score("pct test")
        score.addTestResult(TRUE_DF, PRED_DF)
        with patch("matplotlib.pyplot.show"):
            fig = score.plotPercentile()
        self.assertIn("pct test", fig.axes[0].get_title())

    def test_bars_drawn(self) -> None:
        """Bar chart contains at least one patch per species per percentile."""
        if IGNORE_TESTS:
            return
        percentiles = [25.0, 75.0]
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plotPercentile(percentiles=percentiles)
        n_patches = len(fig.axes[0].patches)
        self.assertGreaterEqual(n_patches, len(TRUE_DF.columns) * len(percentiles))


if __name__ == "__main__":
    unittest.main()
