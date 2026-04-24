"""Scores prediction timecourses against true timecourses using absolute relative error (ARE)."""

import io  # type: ignore
from typing import Dict, List, Optional  # type: ignore

import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

AGGREGATION_MEAN = "mean"
AGGREGATION_MIN = "min"
AGGREGATION_MAX = "max"
AGGREGATION_MEDIAN = "median"
AGGREGATION_PERCENTILE_50 = "p50"
AGGREGATION_PERCENTILE_75 = "p75"
AGGREGATION_PERCENTILE_95 = "p95"
DEFAULT_AGGREGATIONS = [AGGREGATION_MEAN, AGGREGATION_MIN, AGGREGATION_MAX,
        AGGREGATION_PERCENTILE_75, AGGREGATION_PERCENTILE_95]
DEFAULT_PERCENTILES = [50.0, 75.0, 0.95]
AGGREGATION_TYPE_TIME = "time"
AGGREGATION_TYPE_SPECIES = "species"
AGGREGATION_TYPE_ALL = "all"
AGGREGATION_TYPE = "aggregation_type"
SOURCE = "source"

# Columns in serializaiton dataframe
SERIALIZATION_COLUMNS = [AGGREGATION_TYPE, SOURCE]
SERIALIZATION_COLUMNS += DEFAULT_AGGREGATIONS

SERIALIZATION_PATH = "score_serialization.csv"

def _makePercentileAggregationStr(percentile: float) -> str:
    """Returns the aggregation string for the given percentile (e.g., 50.0 → 'p50')."""
    result =  f"p{percentile:g}"
    result = result.replace("p0.", "p")
    return result


class Score:
    """Scores prediction timecourses against true timecourses.

    ARE = (prediction - true) / true.
    """

    def __init__(self, description: str = "",
            serialization_path: str = SERIALIZATION_PATH) -> None:
        """
        Constructs an empty Score with descriptive information.

        Parameters
        ----------
        description : str
            Descriptive label for this score.
        serialization_path : Optional[str]
            Path to a CSV file for persistence. If provided, serialize() writes here
            and deserialize() reads from here.
        """
        self.description = description
        self._serialization_path = serialization_path
        # FIXME: Make this a dict
        self._are_dfs: List[pd.DataFrame] = []
        self._serialization_dct: Dict[str, List] = {c: [] for c in SERIALIZATION_COLUMNS}

    def addTestResult(self, true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame) -> None:
        """
        Adds a test result by computing ARE from true and prediction timecourses.
        ARE = (prediction - true) / true; NaN where true == 0.

        Parameters
        ----------
        true_timecourse_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_timecourse_df : pd.DataFrame
            Prediction timecourse with same structure as true_timecourse_df.
        """
        are_df = self._computeARE(true_timecourse_df, prediction_timecourse_df)
        self._are_dfs.append(are_df)
        self.serialize()

    def serialize(self) -> None:
        """Writes the Score state to a CSV at _serialization_path; no-op if path is None."""
        if self._serialization_path is None:
            return
        with open(self._serialization_path, 'w') as f:
            f.write(f"# {self.description}\n")
            if self._are_dfs:
                combined_df = pd.concat(
                        self._are_dfs, keys=range(len(self._are_dfs)),
                        names=['result_idx', 'time'])
                combined_df.to_csv(f)

    @classmethod
    def deserialize(cls, path: str) -> 'Score':
        """Reconstructs a Score from a CSV written by serialize().

        Parameters
        ----------
        path : str
            Path to the CSV file previously written by serialize().

        Returns
        -------
        Score
            Restored Score with _serialization_path set to path.
        """
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            remaining = f.read()
        description = first_line[2:] if first_line.startswith('# ') else ""
        score = cls(description=description, serialization_path=path)
        if remaining.strip():
            combined_df = pd.read_csv(io.StringIO(remaining), index_col=[0, 1])
            for idx in sorted(combined_df.index.get_level_values('result_idx').unique()):
                score._are_dfs.append(combined_df.loc[idx])
        return score

    def _computeARE(self, true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Computes ARE = (prediction - true) / true; NaN where true == 0."""
        with np.errstate(divide='ignore', invalid='ignore'):
            are_arr = np.where(
                    true_df.values == 0,
                    np.nan,
                    (prediction_df.values - true_df.values) / true_df.values)
        return pd.DataFrame(are_arr, index=true_df.index, columns=true_df.columns)

    def _getCombinedARE(self) -> pd.DataFrame:
        """Returns a single ARE DataFrame by concatenating all added test results along time."""
        if len(self._are_dfs) == 0:
            return pd.DataFrame()
        return pd.concat(self._are_dfs, axis=0)

    def _applyAggregation(self, arr: np.ndarray, aggregation: str) -> np.ndarray:
        """Applies the named aggregation to arr along axis 0."""
        if aggregation == AGGREGATION_MEAN:
            return np.nanmean(arr, axis=0)
        if aggregation == AGGREGATION_MIN:
            return np.nanmin(arr, axis=0)
        if aggregation == AGGREGATION_MAX:
            return np.nanmax(arr, axis=0)
        if aggregation == AGGREGATION_MEDIAN:
            return np.nanmedian(arr, axis=0)
        if aggregation.startswith("p"):
            percentile = float(aggregation[1:])
            return np.nanpercentile(arr, percentile, axis=0)
        raise ValueError(f"Unknown aggregation: {aggregation}")

    def aggregateByTime(self,
            aggregations: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregates ARE across time for each species.

        Parameters
        ----------
        aggregations : List[str]
            Aggregation functions. Supported: "mean", "min", "max", "median", "pN" (Nth percentile).

        Returns
        -------
        pd.DataFrame
            Index = species names, columns = aggregation names.
        """
        if aggregations is None:
            aggregations = DEFAULT_AGGREGATIONS
        combined_df = self._getCombinedARE()
        if combined_df.empty:
            return pd.DataFrame()
        result_dct: Dict[str, np.ndarray] = {}
        for aggregation in aggregations:
            result_dct[aggregation] = self._applyAggregation(
                    combined_df.values, aggregation)
        return pd.DataFrame(result_dct, index=combined_df.columns)

    def aggregateBySpecies(self,
            aggregations: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregates ARE across species at each timepoint.

        Parameters
        ----------
        aggregations : List[str]
            Aggregation functions. Supported: "mean", "min", "max", "median", "pN" (Nth percentile).

        Returns
        -------
        pd.DataFrame
            Index = timepoints, columns = aggregation names.
        """
        if aggregations is None:
            aggregations = DEFAULT_AGGREGATIONS
        combined_df = self._getCombinedARE()
        if combined_df.empty:
            return pd.DataFrame()
        result_dct: Dict[str, np.ndarray] = {}
        for aggregation in aggregations:
            result_dct[aggregation] = self._applyAggregation(
                    combined_df.values.T, aggregation)
        return pd.DataFrame(result_dct, index=combined_df.index)

    def plotTime(self, aggregations: Optional[List[str]] = None,
            ax: Optional[plt.Axes] = None) -> mfigure.Figure:  # type: ignore
        """
        Constructs a time plot of species aggregations over time.
        x-axis = time, y-axis = ARE aggregated across species, one line per aggregation.

        Parameters
        ----------
        aggregations : List[str]
            Aggregation functions to plot.
        ax : Optional[plt.Axes]
            Axes to draw on. A new figure is created if None.

        Returns
        -------
        mfigure.Figure
        """
        agg_df = self.aggregateBySpecies(aggregations)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        assert isinstance(fig, mfigure.Figure)
        for col in agg_df.columns:
            ax.plot(agg_df.index, agg_df[col], label=col)
        ax.set_xlabel("Time")
        ax.set_ylabel("ARE")
        ax.set_title(f"Species Aggregations over Time: {self.description}")
        ax.legend()
        fig.tight_layout()
        plt.show()
        return fig

    def aggregateByPercentile(self,
            percentiles: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Aggregates ARE by percentile across all timepoints and test results for each species.

        Parameters
        ----------
        percentiles : List[float]
            Percentile values in the range [0, 100].

        Returns
        -------
        pd.DataFrame
            Index = species names, columns = percentile labels (e.g., 'p50', 'p75', 'p95').
        """
        if percentiles is None:
            percentiles = DEFAULT_PERCENTILES
        aggregations = [_makePercentileAggregationStr(p) for p in percentiles]
        return self.aggregateByTime(aggregations)

    def plotPercentile(self, percentiles: Optional[List[float]] = None,
            ax: Optional[plt.Axes] = None) -> mfigure.Figure:  # type: ignore
        """
        Bar plot of ARE at each percentile for every species.
        x-axis = species, y-axis = ARE, grouped bars per percentile.

        Parameters
        ----------
        percentiles : List[float]
            Percentile values in the range [0, 100].
        ax : Optional[plt.Axes]
            Axes to draw on. A new figure is created if None.

        Returns
        -------
        mfigure.Figure
        """
        agg_df = self.aggregateByPercentile(percentiles)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        assert isinstance(fig, mfigure.Figure)
        agg_df.plot(kind='bar', ax=ax)
        ax.set_xlabel("Species")
        ax.set_ylabel("ARE")
        ax.set_title(f"ARE Percentiles by Species: {self.description}")
        ax.legend(title="Percentile")
        fig.tight_layout()
        plt.show()
        return fig

    def plotSpecies(self, aggregations: Optional[List[str]] = None,
            ax: Optional[plt.Axes] = None) -> mfigure.Figure:  # type: ignore
        """
        Constructs a bar plot of time aggregations over species.
        x-axis = species, y-axis = ARE aggregated across time, grouped bars per aggregation.

        Parameters
        ----------
        aggregations : List[str]
            Aggregation functions to plot.
        ax : Optional[plt.Axes]
            Axes to draw on. A new figure is created if None.

        Returns
        -------
        mfigure.Figure
        """
        agg_df = self.aggregateByTime(aggregations)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        assert isinstance(fig, mfigure.Figure)
        agg_df.plot(kind='bar', ax=ax)
        ax.set_xlabel("Species")
        ax.set_ylabel("ARE")
        ax.set_title(f"Time Aggregations over Species: {self.description}")
        ax.legend(title="Aggregation")
        fig.tight_layout()
        plt.show()
        return fig
