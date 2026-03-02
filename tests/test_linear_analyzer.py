"""
Tests for LinearAnalyzer class.
"""

import os
import sys
import tempfile
import unittest

import matplotlib   # type: ignore
matplotlib.use("Agg")  # Non-interactive backend for testing; must precede pyplot import
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import tellurium as te  # type: ignore


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from linear_analyzer import LinearAnalyzer # type: ignore

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
FIRST_BIOMODEL_SBML = os.path.join(
    BIOMODELS_DIR, "BIOMD0000000001", "BIOMD0000000001_url.xml"
)


def _load_sbml(path: str) -> str:
    """Read an SBML file and return its contents as a string."""
    with open(path) as f:
        return f.read()


class TestLinearAnalyzerInit(unittest.TestCase):
    """Tests for LinearAnalyzer.__init__."""

    def test_init_antimony_defaults(self) -> None:
        """Initializing with an Antimony string uses default simulation parameters."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.start, 0)
        self.assertEqual(analyzer.end, 10)
        self.assertEqual(analyzer.num_point, 100)
        self.assertIsNotNone(analyzer._rr)

    def test_init_antimony_custom_params(self) -> None:
        """Custom start, end, and num_points are stored correctly."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, start=1, end=5, num_point=50)
        self.assertEqual(analyzer.start, 1)
        self.assertEqual(analyzer.end, 5)
        self.assertEqual(analyzer.num_point, 50)

    def test_init_sbml(self) -> None:
        """Initializing with an SBML string loads the model correctly."""
        sbml = _load_sbml(FIRST_BIOMODEL_SBML)
        analyzer = LinearAnalyzer(sbml)
        self.assertIsNotNone(analyzer._rr)
        self.assertGreater(len(analyzer._species_ids), 0)

    def test_init_species_ids_populated(self) -> None:
        """Floating species IDs are extracted from the model on construction."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertIn("S1", analyzer._species_ids)
        self.assertIn("S2", analyzer._species_ids)


class TestCollectJacobians(unittest.TestCase):
    """Tests for LinearAnalyzer.collectJacobians."""

    def test_returns_ndarray(self) -> None:
        """collectJacobians returns a numpy ndarray."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        jacobians = analyzer.collectJacobians()
        self.assertIsInstance(jacobians, np.ndarray)

    def test_shape(self) -> None:
        """Jacobian array has shape (num_points, n_species, n_species)."""
        num_points = 10
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=num_points)
        jacobians = analyzer.collectJacobians()
        n_species = len(analyzer._species_ids)
        self.assertEqual(jacobians.shape, (num_points, n_species, n_species))

    def test_jacobians_stored_on_instance(self) -> None:
        """After collection, _jacobians attribute is set on the instance."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        self.assertIsNone(analyzer._jacobian_arr)
        analyzer.collectJacobians()
        self.assertIsNotNone(analyzer._jacobian_arr)

    def test_jacobians_contain_finite_values(self) -> None:
        """Collected Jacobians contain at least some finite (non-NaN) values."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        jacobians = analyzer.collectJacobians()
        self.assertTrue(np.any(np.isfinite(jacobians)))

    def test_sbml_model_jacobians(self) -> None:
        """collectJacobians works correctly on an SBML model."""
        sbml = _load_sbml(FIRST_BIOMODEL_SBML)
        analyzer = LinearAnalyzer(sbml, num_point=5)
        jacobians = analyzer.collectJacobians()
        n_species = len(analyzer._species_ids)
        self.assertEqual(jacobians.shape, (5, n_species, n_species))


class TestPlot(unittest.TestCase):
    """Tests for LinearAnalyzer.plot."""

    def test_plot_returns_figure(self) -> None:
        """plot returns a matplotlib Figure object."""

        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        fig = analyzer.plot()
        self.assertIsInstance(fig, matplotlib.figure.Figure)  # type: ignore
        plt.close(fig)

    def test_plot_auto_collects_jacobians(self) -> None:
        """plot calls collectJacobians automatically if not yet collected."""
        import matplotlib.pyplot as plt

        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        self.assertIsNone(analyzer._jacobian_arr)
        fig = analyzer.plot()
        self.assertIsNotNone(analyzer._jacobian_arr)
        plt.close(fig)

    def test_plot_after_explicit_collect(self) -> None:
        """plot succeeds when called after an explicit collectJacobians call."""
        import matplotlib.pyplot as plt

        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        analyzer.collectJacobians()
        fig = analyzer.plot()
        self.assertIsInstance(fig, matplotlib.figure.Figure)  # type: ignore
        plt.close(fig)


class TestProcessBioModels(unittest.TestCase):
    """Tests for LinearAnalyzer.processBioModels."""

    def test_returns_list(self) -> None:
        """processBioModels returns a list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.processBioModels(directory=tmp_dir)
        self.assertIsInstance(results, list)

    def test_empty_directory_returns_empty_list(self) -> None:
        """An empty directory yields an empty results list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.processBioModels(directory=tmp_dir)
        self.assertEqual(results, [])

    def test_processes_sbml_model_in_subdirectory(self) -> None:
        """A valid SBML file in a subdirectory is loaded and processed."""

        rr = te.loada(ANTIMONY_MODEL)
        sbml_str = rr.getSBML()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            sbml_path = os.path.join(model_dir, "MODEL0001_url.xml")
            with open(sbml_path, "w") as f:
                f.write(sbml_str)

            results = LinearAnalyzer.processBioModels(directory=tmp_dir)

        self.assertEqual(len(results), 1)
        model_id, analyzer = results[0]
        self.assertEqual(model_id, "MODEL0001")
        self.assertIsInstance(analyzer, LinearAnalyzer)
        self.assertIsNotNone(analyzer._jacobian_arr)

    def test_result_tuples_contain_model_id_and_analyzer(self) -> None:
        """Each result tuple contains (str, LinearAnalyzer)."""
        import tellurium as te

        rr = te.loada(ANTIMONY_MODEL)
        sbml_str = rr.getSBML()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "TESTMODEL")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "TESTMODEL_url.xml"), "w") as f:
                f.write(sbml_str)

            results = LinearAnalyzer.processBioModels(directory=tmp_dir)

        self.assertEqual(len(results), 1)
        model_id, analyzer = results[0]
        self.assertIsInstance(model_id, str)
        self.assertIsInstance(analyzer, LinearAnalyzer)

    def test_skips_invalid_model(self) -> None:
        """A directory with an invalid XML file is skipped without raising."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_dir = os.path.join(tmp_dir, "BADMODEL")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "BADMODEL_url.xml"), "w") as f:
                f.write("<?xml version='1.0'?><not_sbml/>")

            results = LinearAnalyzer.processBioModels(directory=tmp_dir)

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()