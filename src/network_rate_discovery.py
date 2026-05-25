"""
Chemical Network Rate Discovery via SINDy (Sparse Identification of Nonlinear Dynamics)
====================================================================================
Discovers a system of ODEs from time-series concentration data using PySINDy
that estimate the derivatives of each species as a sparse linear combination of polynomial features of the species concentrations.
Assumes rate laws are at most quadratic in the species concentrations (i.e.,
the library includes constant, linear, and pairwise-product terms).

Supports up to 10 chemical species.

Dependencies
------------
    pip install pysindy pandas numpy scipy matplotlib

Input
-----
A pandas DataFrame with:
  - One column named 'time'  (or passed separately as the `time_col` argument)
  - One column per species   (up to 10)

Usage
-----
    from chemical_network_sindy import NetworkDiscovery

    discovery = NetworkRateDiscovery(
        df,
        time_col="time",
        threshold=0.05,          # STLSQ sparsity threshold
        alpha=0.05,              # L2 regularisation
        differentiation="smooth" # "smooth" | "finite" | "spectral"
    )
    discovery.fit()
    discovery.print_equations()
    discovery.plot_results()
    summary = discovery.summary()
"""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pysindy as ps # type: ignore
from pysindy.feature_library import PolynomialLibrary # type: ignore
from scipy.integrate import solve_ivp # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)

# FIXME: Not able to force a value for a fixed input
# FIXME: Not fitting to the state variables; fitting to their derivatives.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SPECIES = 10
DifferentiationMethod = Literal["smooth", "finite", "spectral"]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NetworkRateDiscovery:
    """Discover a chemical reaction network from concentration time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series data.  Must contain a time column and one column per
        chemical species (concentrations must be non-negative).
    time_col : str
        Name of the column holding time values.  Default ``"time"``.
    threshold : float
        STLSQ sparsity threshold.  Terms whose coefficient magnitude falls
        below this value are pruned.  Tune this to trade sparsity for fit.
        Default ``0.05``.
    alpha : float
        L2 (ridge) regularisation coefficient for STLSQ.  Default ``0.05``.
    differentiation : str
        Numerical differentiation strategy:
        - ``"smooth"``   – SmoothedFiniteDifference (recommended for noisy data)
        - ``"finite"``   – standard finite differences
        - ``"spectral"`` – spectral derivative (requires uniform sampling)
        Default ``"smooth"``.
    poly_degree : int
        Maximum polynomial degree of the feature library.  Must be 1 or 2
        (linear or quadratic rate laws).  Default ``2``.
    include_bias : bool
        Whether to include a constant (zeroth-order / production) term in the
        library.  Default ``True``.
    species_names : list[str] | None
        Override species labels used in printed equations and plots.
        If ``None``, column names from *df* are used.
    bias_species : list[str] | None
        Names of species whose ODE is permitted to have a constant term.
        All other species have their constant coefficient forced to zero
        after fitting.  Names must match ``species_names`` (or the
        DataFrame column names when ``species_names`` is ``None``).
        When provided, ``include_bias`` is forced to ``True`` so that the
        constant feature exists in the library.  Default ``None`` (no
        per-species restriction).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        time_col: str = "time",
        threshold: float = 0.05,
        alpha: float = 0.05,
        differentiation: DifferentiationMethod = "smooth",
        poly_degree: int = 2,
        include_bias: bool = True,
        species_names: list[str] | None = None,
        bias_species: list[str] | None = None,
    ) -> None:
        if poly_degree not in (1, 2):
            raise ValueError("`poly_degree` must be 1 (linear) or 2 (quadratic).")

        self._validate_dataframe(df, time_col)

        self.df = df.copy()
        self.time_col = time_col
        self.threshold = threshold
        self.alpha = alpha
        self.differentiation = differentiation
        self.poly_degree = poly_degree
        self.include_bias = include_bias

        # Extract time and concentration arrays
        species_cols = [c for c in df.columns if c != time_col]
        if len(species_cols) > MAX_SPECIES:
            raise ValueError(
                f"DataFrame contains {len(species_cols)} species columns; "
                f"maximum supported is {MAX_SPECIES}."
            )

        self.species_cols = species_cols
        self.t: np.ndarray = df[time_col].to_numpy(dtype=float)
        self.X: np.ndarray = df[species_cols].to_numpy(dtype=float)

        if species_names is not None:
            if len(species_names) != len(species_cols):
                raise ValueError(
                    "`species_names` length must match the number of species columns."
                )
            self.species_names = species_names
        else:
            self.species_names = species_cols

        if bias_species is not None:
            invalid = set(bias_species) - set(self.species_names)
            if invalid:
                raise ValueError(
                    f"`bias_species` contains names not in species_names: {sorted(invalid)}"
                )
            self.include_bias = True
        self.bias_species: list[str] | None = bias_species

        self.model: ps.SINDy

        library = PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=True,
        )

        optimizer = ps.STLSQ(threshold=self.threshold, alpha=self.alpha)

        diff_method = self._build_differentiator()

        self.model = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
            differentiation_method=diff_method,
        )
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> "NetworkRateDiscovery":
        """Fit the SINDy model to the data.

        Returns
        -------
        self
        """

        dt = float(np.median(np.diff(self.t)))
        self.model.fit(self.X, t=dt, feature_names=self.species_names)
        if self.bias_species is not None:
            allowed = set(self.bias_species)
            for i, name in enumerate(self.species_names):
                if name not in allowed:
                    self.model.optimizer.coef_[i, 0] = 0.0
        self._is_fitted = True
        return self

    def print_equations(self) -> None:
        """Pretty-print the discovered ODE equations."""
        self._require_fitted()
        print("\n" + "=" * 60)
        print("  Discovered Chemical Rate Equations")
        print("=" * 60)
        self.model.print()
        print("=" * 60 + "\n")

    def predict(self) -> pd.DataFrame:
        """Integrate the discovered ODE and return predicted concentrations.

        Returns
        -------
        pd.DataFrame
            Predicted concentrations with time as the index and one column per
            species.  Raises ``RuntimeError`` if the ODE integrator fails.
        """
        self._require_fitted()
        X_sim = self._simulate()
        return pd.DataFrame(X_sim, index=self.t, columns=self.species_names)

    def r_squared(self, method: str = "derivative") -> dict[str, float]:
        """Compute R² for each species.

        Parameters
        ----------
        method : str
            ``"derivative"`` (default) – computes R² on the numerical time
            derivatives, which is fast and always works.
            ``"simulation"`` – integrates the ODE forward and compares
            trajectories; more informative but may fail for stiff systems or
            poorly-identified models.

        Returns
        -------
        dict mapping species name → R²
        """
        self._require_fitted()
        if method == "simulation":
            return self._r_squared_simulation()
        return self._r_squared_derivative()

    def _r_squared_derivative(self) -> dict[str, float]:
        """R² on predicted vs numerical derivatives."""
        dt = float(np.median(np.diff(self.t)))
        X_dot_pred = self.model.predict(self.X)          # predicted derivatives
        X_dot_num  = self.model.differentiation_method(self.X, self.t)  # numerical
        r2 = {}
        for i, name in enumerate(self.species_names):
            y_true = X_dot_num[:, i]
            y_pred = X_dot_pred[:, i]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return r2

    def _r_squared_simulation(self) -> dict[str, float]:
        """R² on simulated concentration trajectories."""
        try:
            X_sim = self._simulate()
            r2 = {}
            for i, name in enumerate(self.species_names):
                ss_res = np.sum((self.X[:, i] - X_sim[:, i]) ** 2)
                ss_tot = np.sum((self.X[:, i] - self.X[:, i].mean()) ** 2)
                r2[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            return r2
        except Exception as exc:
            warnings.warn(f"Simulation R² failed: {exc}. Use method='derivative'.")
            return {name: float("nan") for name in self.species_names}

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of non-zero coefficients for all species.

        Rows are candidate library terms; columns are species.

        Returns
        -------
        pd.DataFrame
        """
        self._require_fitted()
        feature_names = self.model.get_feature_names()
        coefs = self.model.coefficients()          # shape (n_species, n_features)
        df_coef = pd.DataFrame(
            coefs.T,
            index=feature_names,
            columns=[f"d{n}/dt" for n in self.species_names],
        )
        # Keep only rows where at least one coefficient is non-zero
        non_zero_mask = (df_coef.abs() > 0).any(axis=1)
        return df_coef[non_zero_mask]

    def plotResult(
        self,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot observed vs. model-simulated trajectories for each species.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.  Auto-sized if *None*.
        show : bool
            Call ``plt.show()`` at the end.  Set to ``False`` when embedding
            in a larger figure or saving manually.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_fitted()

        n = len(self.species_names)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 3.5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle(
            "Chemical Network Discovery – Observed vs Predicted",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        try:
            pred_df = self.predict()
            prediction_ok = True
        except Exception as exc:
            warnings.warn(f"Prediction failed for plotting: {exc}")
            pred_df = None
            prediction_ok = False

        r2_vals = self.r_squared(method="derivative")

        for idx, name in enumerate(self.species_names):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            color = f"C{idx}"
            ax.plot(self.t, self.X[:, idx], "-", lw=2, color=color, label=f"{name} (observed)")
            if prediction_ok and pred_df is not None:
                ax.plot(pred_df.index, pred_df[name], "--", lw=2, color=color, label=f"{name} (predicted)")
            r2 = r2_vals.get(name, float("nan"))
            title = f"{name}"
            if not np.isnan(r2):
                title += f"   R²={r2:.4f}"
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_coefficient_heatmap(
        self,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualise the coefficient matrix as a heatmap.

        Each row is a library feature; each column is a species.
        Non-zero entries (active terms) are highlighted.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_fitted()

        df_coef = self.summary()
        if df_coef.empty:
            print("No non-zero coefficients found; heatmap skipped.")
            return plt.figure()

        if figsize is None:
            figsize = (max(6, len(df_coef.columns) * 1.5), max(4, len(df_coef) * 0.5))

        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(df_coef.values, aspect="auto", cmap="RdBu_r")
        fig.colorbar(cax, ax=ax, label="Coefficient value")

        ax.set_xticks(range(len(df_coef.columns)))
        ax.set_xticklabels(df_coef.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df_coef.index)))
        ax.set_yticklabels(df_coef.index)
        ax.set_title("SINDy Coefficient Matrix (non-zero terms)", fontweight="bold")

        # Annotate cells
        for i in range(len(df_coef.index)):
            for j in range(len(df_coef.columns)):
                val = df_coef.iloc[i, j]
                if abs(val) > 1e-10:
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > df_coef.values.max() * 0.5 else "black",
                    )

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call `.fit()` before using this method.")

    def _build_differentiator(self):
        if self.differentiation == "smooth":
            return ps.SmoothedFiniteDifference()
        elif self.differentiation == "finite":
            return ps.FiniteDifference()
        elif self.differentiation == "spectral":
            return ps.SpectralDerivative()
        else:
            raise ValueError(
                f"Unknown differentiation method '{self.differentiation}'. "
                "Choose from: 'smooth', 'finite', 'spectral'."
            )

    def _simulate(self) -> np.ndarray:
        """Integrate the discovered ODE forward from the first observation."""
        x0 = self.X[0, :]

        def rhs(t, x):
            return self.model.predict(x.reshape(1, -1))[0]

        sol = solve_ivp(
            rhs,
            t_span=(self.t[0], self.t[-1]),
            y0=x0,
            t_eval=self.t,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y.T   # shape (n_timepoints, n_species)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, time_col: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame.")
        species_cols = [c for c in df.columns if c != time_col]
        if len(species_cols) == 0:
            raise ValueError("DataFrame must contain at least one species column.")
        if len(species_cols) > MAX_SPECIES:
            raise ValueError(
                f"DataFrame has {len(species_cols)} species columns; "
                f"maximum is {MAX_SPECIES}."
            )
        if df[time_col].is_monotonic_increasing is False:
            raise ValueError("Time column must be strictly increasing.")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def discover_network(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    threshold: float = 0.05,
    alpha: float = 0.05,
    differentiation: DifferentiationMethod = "smooth",
    poly_degree: int = 2,
    include_bias: bool = True,
    species_names: list[str] | None = None,
    plot: bool = True,
    heatmap: bool = True,
) -> NetworkRateDiscovery:
    """One-shot helper: construct, fit, print, and optionally plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (see :class:`NetworkRateDiscovery`).
    time_col : str
        Name of the time column.
    threshold : float
        STLSQ sparsity threshold.
    alpha : float
        Ridge regularisation.
    differentiation : str
        ``"smooth"`` | ``"finite"`` | ``"spectral"``.
    poly_degree : int
        1 (linear) or 2 (quadratic).
    include_bias : bool
        Include a constant term in the library.
    species_names : list[str] | None
        Human-readable species labels.
    plot : bool
        Show trajectory comparison plots.
    heatmap : bool
        Show coefficient heatmap.

    Returns
    -------
    NetworkRateDiscovery
        Fitted discovery object.

    Example
    -------
    >>> disc = discover_network(df, time_col="time", threshold=0.02)
    >>> disc.print_equations()
    >>> summary = disc.summary()
    """
    disc = NetworkRateDiscovery(
        df,
        time_col=time_col,
        threshold=threshold,
        alpha=alpha,
        differentiation=differentiation,
        poly_degree=poly_degree,
        include_bias=include_bias,
        species_names=species_names,
    )
    disc.fit()
    disc.print_equations()

    r2 = disc.r_squared(method="derivative")
    print("R² on time derivatives per species:")
    for name, val in r2.items():
        print(f"  {name}: {val:.6f}")
    print()

    try:
        r2_sim = disc.r_squared(method="simulation")
        print("R² on simulated trajectories per species:")
        for name, val in r2_sim.items():
            print(f"  {name}: {val:.6f}")
        print()
    except Exception:
        pass

    if plot:
        disc.plotResult()
    if heatmap:
        disc.plot_coefficient_heatmap()

    return disc


# ---------------------------------------------------------------------------
# Demo  (Brusselator – a classic chemical oscillator)
# ---------------------------------------------------------------------------


def _generate_brusselator(
    t_end: float = 20.0,
    n_points: int = 400,
    noise_std: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic Brusselator data for testing.

    The Brusselator is a simple two-species chemical oscillator:
        dX/dt = A + X²Y - (B+1)X
        dY/dt = BX - X²Y
    with A=1, B=3 → limit cycle.
    """
    rng = np.random.default_rng(seed)
    A, B = 1.0, 3.0

    def brusselator(t, z):
        X, Y = z
        return [A + X**2 * Y - (B + 1) * X, B * X - X**2 * Y]

    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(brusselator, [0, t_end], [0.5, 2.0], t_eval=t_eval, rtol=1e-8)

    X_data = sol.y[0] + rng.normal(0, noise_std, n_points)
    Y_data = sol.y[1] + rng.normal(0, noise_std, n_points)

    return pd.DataFrame({"time": t_eval, "X": X_data, "Y": Y_data})


if __name__ == "__main__":
    print("Brusselator demo\n" + "-" * 40)
    print("True equations:")
    print("  dX/dt =  1  +  X²Y  -  4X")
    print("  dY/dt = 3X  -  X²Y\n")

    df_demo = _generate_brusselator(noise_std=0.01)

    disc = discover_network(
        df_demo,
        time_col="time",
        threshold=0.1,
        alpha=0.01,
        differentiation="smooth",
        poly_degree=2,
        include_bias=True,
        plot=True,
        heatmap=True,
    )

    print("\nCoefficient summary:")
    print(disc.summary().to_string())
