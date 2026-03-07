# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The project uses a local venv at `mla/`. Always activate it before running anything:

```bash
source activate.sh
```

`activate.sh` also adds `src/` to `PYTHONPATH`, so imports like `from linear_analyzer import LinearAnalyzer` work in tests without a package install.

## Commands

```bash
# Run all tests
source activate.sh && python3 -m pytest tests/ -v

# Run a single test file
source activate.sh && python3 -m pytest tests/test_linear_analyzer.py -v

# Run a single test by name
source activate.sh && python3 -m pytest tests/test_linear_analyzer.py::TestCollectJacobians::test_shape -v

# Lint
source activate.sh && pylint src/

# Run with coverage
source activate.sh && python3 -m pytest tests/ --cov=src
```

## Architecture

All code lives in `src/`, all tests in `tests/`. There are two modules:

**[src/constants.py](src/constants.py)** — Project-wide paths: `PROJECT_DIR` (repo root) and `DATA_DIR` (`<repo>/data/`).

**[src/linear_analyzer.py](src/linear_analyzer.py)** — `LinearAnalyzer` class that:
1. Accepts an SBML XML string or Antimony string (auto-detected: SBML starts with `<?xml` or `<sbml`).
2. Uses [tellurium](https://tellurium.readthedocs.io/) / RoadRunner to simulate the model.
3. `collectJacobians()` — resets the model, runs a first simulation to obtain timepoints, then resets again and steps forward timepoint-by-timepoint calling `rr.getFullJacobian()` at each step. Returns `ndarray` of shape `(num_points, n_species, n_species)`. Only floating species appear in the Jacobian. Cached in `_jacobian_arr`.
4. `makeJacobianCVs()` — computes CV = |std/mean| across timepoints per Jacobian entry (zero-mean entries become NaN). Returns `ndarray` of shape `(n_species, n_species)`. Calls `collectJacobians()` automatically if not yet cached.
5. `plot()` — renders a seaborn heatmap of the CV matrix from `makeJacobianCVs()`. Returns `plt.Figure`.
6. `processBioModels(directory)` — class method that walks subdirectories, finds the first non-`manifest.xml` XML file in each, loads it as SBML, calls `collectJacobians()`, and returns `List[Tuple[str, LinearAnalyzer]]`. Failures are printed as warnings and skipped.
7. `processBioModelsCVs(directory, data_file)` — class method like `processBioModels` but calls `makeJacobianCVs()` instead. Returns `Dict[str, np.ndarray]` and writes results to a NaN-padded CSV at `data_file` (default: `data/model_linearity_analysis_data.csv`).

## BioModels Data

SBML models are stored in `/Users/jlheller/home/Technical/repos/temp-biomodels/final/`. Each subdirectory (e.g. `BIOMD0000000001/`) contains `<ID>_url.xml` (the SBML file) and `manifest.xml` (skip this).

## Coding Style

Per `docs/specification.md`:
- Method names: camelCase
- Variable names: lower_case_with_underscores with required type suffixes:
  - `_dct` for dicts, `_arr` for arrays, `_df` for DataFrames, `_ser` for Series
  - Lists end in `s` (no suffix), e.g. `results`, `jacobians`
- All functions have docstrings and type-annotated signatures
