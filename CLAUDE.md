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

All code lives in `src/`, all tests in `tests/`. There is one module:

**[src/linear_analyzer.py](src/linear_analyzer.py)** — `LinearAnalyzer` class that:
1. Accepts an SBML XML string or Antimony string (auto-detected: SBML starts with `<?xml` or `<sbml`)
2. Uses [tellurium](https://tellurium.readthedocs.io/) / RoadRunner to simulate the model
3. `collectJacobians()` — resets the model, runs a first simulation to obtain timepoints, then resets again and steps forward timepoint-by-timepoint calling `rr.getFullJacobian()` at each step. Returns `ndarray` of shape `(num_points, n_species, n_species)`. Only floating species appear in the Jacobian.
4. `plot()` — computes CV = |std/mean| across timepoints per Jacobian entry, renders a seaborn heatmap. Returns `plt.Figure`.
5. `processBioModels(directory)` — class method that walks subdirectories, finds the first non-`manifest.xml` XML file in each, loads it as SBML, collects Jacobians, and returns `List[Tuple[str, LinearAnalyzer]]`. Failures are printed as warnings and skipped.

## BioModels Data

SBML models are stored in `/Users/jlheller/home/Technical/repos/temp-biomodels/final/`. Each subdirectory (e.g. `BIOMD0000000001/`) contains `<ID>_url.xml` (the SBML file) and `manifest.xml` (skip this).

## Coding Style

Per `docs/specification.md`:
- Method names: camelCase
- Variable names: lower_case_with_underscores
- All functions have docstrings and type-annotated signatures
