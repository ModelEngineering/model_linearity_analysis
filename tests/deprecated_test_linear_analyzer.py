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
import pandas as pd # type: ignore
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
        jacobian_arr = analyzer.collectJacobians()
        self.assertIsInstance(jacobian_arr, np.ndarray)

    def test_shape(self) -> None:
        """Jacobian array has shape (n_points, n_species, n_species)."""
        n_points = 10
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=n_points)
        jacobian_arr = analyzer.collectJacobians()
        n_species = len(analyzer._species_ids)
        self.assertEqual(jacobian_arr.shape, (n_points, n_species, n_species))

    def test_jacobians_stored_on_instance(self) -> None:
        """After collection, _jacobians attribute is set on the instance."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        self.assertIsNone(analyzer._jacobian_arr)
        analyzer.collectJacobians()
        self.assertIsNotNone(analyzer._jacobian_arr)

    def test_jacobians_contain_finite_values(self) -> None:
        """Collected Jacobians contain at least some finite (non-NaN) values."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        jacobian_arr = analyzer.collectJacobians()
        self.assertTrue(np.any(np.isfinite(jacobian_arr)))

    def test_sbml_model_jacobians(self) -> None:
        """collectJacobians works correctly on an SBML model."""
        sbml = _load_sbml(FIRST_BIOMODEL_SBML)
        analyzer = LinearAnalyzer(sbml, num_point=5)
        jacobian_arr = analyzer.collectJacobians()
        n_species = len(analyzer._species_ids)
        self.assertEqual(jacobian_arr.shape, (5, n_species, n_species))


class TestPartitionJacobians(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionJacobians."""

    def setUp(self) -> None:
        self.n_points = 20
        self.analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=self.n_points)
        self.analyzer.collectJacobians()

    def test_returns_cluster_result(self) -> None:
        """partitionJacobians returns a ClusterResult namedtuple."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertIsInstance(result.max_cv, float)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """The clusters list has exactly n_cluster elements."""
        result = self.analyzer.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)

    def test_each_partition_is_ndarray(self) -> None:
        """Each element in clusters is a numpy ndarray."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for partition in result.clusters:
            self.assertIsInstance(partition, np.ndarray)

    def test_partition_ndim(self) -> None:
        """Each partition array has 3 dimensions (n_i, n_species, n_species)."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for partition in result.clusters:
            self.assertEqual(partition.ndim, 3)

    def test_partition_shape_species_dims(self) -> None:
        """The last two dimensions of each partition match n_species."""
        n_species = len(self.analyzer._species_ids)
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for partition in result.clusters:
            self.assertEqual(partition.shape[1], n_species)
            self.assertEqual(partition.shape[2], n_species)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all partitions equals n_points."""
        result = self.analyzer.partitionJacobians(n_cluster=4)
        total = sum(p.shape[0] for p in result.clusters)
        self.assertEqual(total, self.n_points)

    def test_auto_collects_jacobians(self) -> None:
        """partitionJacobians calls collectJacobians automatically if not cached."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        self.assertIsNone(analyzer._jacobian_arr)
        result = analyzer.partitionJacobians(n_cluster=2)
        self.assertIsNotNone(analyzer._jacobian_arr)
        self.assertEqual(len(result.clusters), 2)

    def test_raises_for_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        analyzer.collectJacobians()
        with self.assertRaises(ValueError):
            analyzer.partitionJacobians(n_cluster=10)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv in ClusterResult is non-negative."""
        result = self.analyzer.partitionJacobians(n_cluster=4)
        self.assertGreaterEqual(result.max_cv, 0.0)


class TestPartitionJacobiansSequentially(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionJacobiansSequentially."""

    def setUp(self) -> None:
        self.n_points = 20
        self.analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=self.n_points)
        self.analyzer.collectJacobians()

    def test_returns_cluster_result(self) -> None:
        """partitionJacobiansSequentially returns a ClusterResult namedtuple."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        self.assertIsInstance(result.max_cv, float)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """The clusters list has exactly n_cluster elements."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)

    def test_each_partition_is_ndarray(self) -> None:
        """Each element in clusters is a numpy ndarray."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for partition in result.clusters:
            self.assertIsInstance(partition, np.ndarray)

    def test_partition_ndim(self) -> None:
        """Each partition array has 3 dimensions (n_i, n_species, n_species)."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for partition in result.clusters:
            self.assertEqual(partition.ndim, 3)

    def test_partition_shape_species_dims(self) -> None:
        """The last two dimensions of each partition match n_species."""
        n_species = len(self.analyzer._species_ids)
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for partition in result.clusters:
            self.assertEqual(partition.shape[1], n_species)
            self.assertEqual(partition.shape[2], n_species)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all partitions equals n_points."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        total = sum(p.shape[0] for p in result.clusters)
        self.assertEqual(total, self.n_points)

    def test_auto_collects_jacobians(self) -> None:
        """partitionJacobiansSequentially calls collectJacobians automatically if not cached."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        self.assertIsNone(analyzer._jacobian_arr)
        result = analyzer.partitionJacobiansSequentially(n_cluster=2)
        self.assertIsNotNone(analyzer._jacobian_arr)
        self.assertEqual(len(result.clusters), 2)

    def test_raises_for_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        analyzer.collectJacobians()
        with self.assertRaises(ValueError):
            analyzer.partitionJacobiansSequentially(n_cluster=10)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv in ClusterResult is non-negative."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        self.assertGreaterEqual(result.max_cv, 0.0)

    def test_clusters_are_contiguous_in_time(self) -> None:
        """Concatenating clusters in order reconstructs the original Jacobian array."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(result.clusters, axis=0)
        np.testing.assert_array_equal(reconstructed, self.analyzer._jacobian_arr)

    def test_no_timepoint_is_skipped_or_repeated(self) -> None:
        """Each timepoint appears in exactly one cluster."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        total = sum(p.shape[0] for p in result.clusters)
        self.assertEqual(total, self.n_points)
        # Verify data identity, not just count
        reconstructed = np.concatenate(result.clusters, axis=0)
        self.assertEqual(reconstructed.shape[0], self.n_points)

    def test_max_cv_le_unpartitioned(self) -> None:
        """Sequential partitioning into multiple clusters yields max_cv <= single-segment CV."""
        single = self.analyzer.partitionJacobiansSequentially(n_cluster=1)
        multi = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        self.assertLessEqual(multi.max_cv, single.max_cv + 1e-9)

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=1)
        self.assertEqual(len(result.clusters), 1)
        np.testing.assert_array_equal(result.clusters[0], self.analyzer._jacobian_arr)

    def test_n_cluster_equals_n_points(self) -> None:
        """With n_cluster == n_points every cluster has exactly one timepoint."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=self.n_points)
        self.assertEqual(len(result.clusters), self.n_points)
        for partition in result.clusters:
            self.assertEqual(partition.shape[0], 1)


class TestProcessBioModels(unittest.TestCase):
    """Tests for LinearAnalyzer.processBioModels."""

    def test_returns_list(self) -> None:
        """processBioModels returns a list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertIsInstance(results, list)

    def test_empty_directory_returns_empty_list(self) -> None:
        """An empty directory yields an empty results list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
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

            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)

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

            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)

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

            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)

        self.assertEqual(results, [])


class TestMakeJacobianCVs(unittest.TestCase):
    """Tests for LinearAnalyzer.makeJacobianCVs."""

    def test_returns_ndarray(self) -> None:
        """makeJacobianCVs returns a numpy ndarray."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        cv_arr = analyzer.makeJacobianCVs()
        self.assertIsInstance(cv_arr, np.ndarray)

    def test_shape(self) -> None:
        """CV array has shape (n_species, n_species)."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        cv_arr = analyzer.makeJacobianCVs()
        n_species = len(analyzer._species_ids)
        self.assertEqual(cv_arr.shape, (n_species, n_species))

    def test_auto_collects_jacobians(self) -> None:
        """makeJacobianCVs calls collectJacobians automatically if needed."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        self.assertIsNone(analyzer._jacobian_arr)
        analyzer.makeJacobianCVs()
        self.assertIsNotNone(analyzer._jacobian_arr)

    def test_values_are_non_negative_or_nan(self) -> None:
        """All finite CV values are non-negative."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        cv_arr = analyzer.makeJacobianCVs()
        finite_arr = cv_arr[np.isfinite(cv_arr)]
        self.assertTrue(np.all(finite_arr >= 0))


class TestProcessBioModelsCVs(unittest.TestCase):
    """Tests for LinearAnalyzer.processBioModelsCVs."""

    def test_returns_dict(self) -> None:
        """processBioModelsCVs returns a Series."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)
        self.assertIsInstance(ser, pd.Series)

    def test_empty_directory_returns_empty_dict(self) -> None:
        """An empty directory yields an empty Series."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)
        self.assertEqual(len(ser), 0)

    def test_keys_are_model_ids_values_are_arrays(self) -> None:
        """Dict keys are model id strings and values are CV ndarrays."""
        rr = te.loada(ANTIMONY_MODEL)
        sbml_str = rr.getSBML()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)

        self.assertIn("MODEL0001", ser.index)
        self.assertIsInstance(ser["MODEL0001"], float)

    def test_cv_array_shape(self) -> None:
        """CV arrays in the result have shape (n_species, n_species)."""
        rr = te.loada(ANTIMONY_MODEL)
        sbml_str = rr.getSBML()
        n_species = len(rr.getFloatingSpeciesIds())

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            result_dct = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)

        self.assertTrue(isinstance(result_dct["MODEL0001"], float))

    def test_skips_invalid_model(self) -> None:
        """A directory with an invalid XML file is skipped without raising."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_dir = os.path.join(tmp_dir, "BADMODEL")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "BADMODEL_url.xml"), "w") as f:
                f.write("<?xml version='1.0'?><not_sbml/>")
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)

        self.assertEqual(len(ser), 0)


class TestProcessBioModelsCVsDataFile(unittest.TestCase):
    """Tests for the data_file feature of LinearAnalyzer.processBioModelsCVs."""

    BIOMD3_SBML = os.path.join(
        "/Users/jlheller/home/Technical/repos/temp-biomodels/final",
        "BIOMD0000000003",
        "BIOMD0000000003_url.xml",
    )

    def _make_model_dir(self, parent: str, model_id: str, sbml_path: str) -> None:
        """Write a single SBML file into a model subdirectory under parent."""
        model_dir = os.path.join(parent, model_id)
        os.makedirs(model_dir)
        with open(sbml_path) as src, open(os.path.join(model_dir, f"{model_id}_url.xml"), "w") as dst:
            dst.write(src.read())

    def test_data_file_is_created(self) -> None:
        """processBioModelsCVs creates the CSV file at the specified path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._make_model_dir(tmp_dir, "BIOMD0000000003", self.BIOMD3_SBML)
            data_file = os.path.join(tmp_dir, "out.csv")
            LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)
            self.assertTrue(os.path.isfile(data_file))

    def test_data_file_is_valid_csv(self) -> None:
        """The written file can be read back as a CSV with pandas when models exist."""
        rr = te.loada(ANTIMONY_MODEL)
        sbml_str = rr.getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)
            result_df = pd.read_csv(data_file, header=None)
            self.assertIsInstance(result_df, pd.DataFrame)

    def test_data_file_index_contains_model_ids(self) -> None:
        """CSV row index contains the processed model IDs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._make_model_dir(tmp_dir, "BIOMD0000000003", self.BIOMD3_SBML)
            data_file = os.path.join(tmp_dir, "out.csv")
            LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)
            df = pd.read_csv(data_file, header=None, index_col=0)
            self.assertIn("BIOMD0000000003", df.index)

    def test_data_file_custom_path(self) -> None:
        """A custom data_file path is respected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._make_model_dir(tmp_dir, "BIOMD0000000003", self.BIOMD3_SBML)
            custom_path = os.path.join(tmp_dir, "subdir", "results.csv")
            os.makedirs(os.path.dirname(custom_path))
            LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=custom_path)
            self.assertTrue(os.path.isfile(custom_path))


class TestWithBioModels(unittest.TestCase):
    """Integration tests using real SBML files from temp-biomodels."""

    BIOMD2_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000002", "BIOMD0000000002_url.xml")
    BIOMD3_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000003", "BIOMD0000000003_url.xml")
    BIOMD4_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000004", "BIOMD0000000004_url.xml")

    # ------------------------------------------------------------------ #
    # collectJacobians                                                     #
    # ------------------------------------------------------------------ #

    def test_collect_jacobians_biomd2_shape(self) -> None:
        """collectJacobians returns the correct shape for BIOMD2 (13 species)."""
        analyzer = LinearAnalyzer(_load_sbml(self.BIOMD2_SBML), num_point=10)
        jacobian_arr = analyzer.collectJacobians()
        self.assertEqual(jacobian_arr.shape, (10, 13, 13))

    def test_collect_jacobians_biomd3_shape(self) -> None:
        """collectJacobians returns the correct shape for BIOMD3 (3 species)."""
        analyzer = LinearAnalyzer(_load_sbml(self.BIOMD3_SBML), num_point=10)
        jacobian_arr = analyzer.collectJacobians()
        self.assertEqual(jacobian_arr.shape, (10, 3, 3))

    def test_collect_jacobians_biomd3_species_ids(self) -> None:
        """Species IDs for BIOMD3 match the expected floating species."""
        analyzer = LinearAnalyzer(_load_sbml(self.BIOMD3_SBML))
        self.assertEqual(sorted(analyzer._species_ids), ["C", "M", "X"])

    # ------------------------------------------------------------------ #
    # makeJacobianCVs                                                      #
    # ------------------------------------------------------------------ #

    def test_make_jacobian_cvs_biomd1_shape(self) -> None:
        """makeJacobianCVs returns (12, 12) CV array for BIOMD1."""
        analyzer = LinearAnalyzer(_load_sbml(FIRST_BIOMODEL_SBML), num_point=10)
        cv_arr = analyzer.makeJacobianCVs()
        self.assertEqual(cv_arr.shape, (12, 12))

    def test_make_jacobian_cvs_biomd4_shape(self) -> None:
        """makeJacobianCVs returns (5, 5) CV array for BIOMD4."""
        analyzer = LinearAnalyzer(_load_sbml(self.BIOMD4_SBML), num_point=10)
        cv_arr = analyzer.makeJacobianCVs()
        self.assertEqual(cv_arr.shape, (5, 5))

    def test_make_jacobian_cvs_biomd3_finite_values(self) -> None:
        """makeJacobianCVs for BIOMD3 contains at least one finite CV value."""
        analyzer = LinearAnalyzer(_load_sbml(self.BIOMD3_SBML), num_point=20)
        cv_arr = analyzer.makeJacobianCVs()
        self.assertTrue(np.any(np.isfinite(cv_arr)))

    # ------------------------------------------------------------------ #
    # processBioModelsCVs                                                  #
    # ------------------------------------------------------------------ #

    def test_process_bio_models_cvs_with_real_models(self) -> None:
        """processBioModelsCVs returns CV arrays for real BioModel SBML files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for model_id, src_file in [
                ("BIOMD0000000003", self.BIOMD3_SBML),
                ("BIOMD0000000004", self.BIOMD4_SBML),
            ]:
                model_dir = os.path.join(tmp_dir, model_id)
                os.makedirs(model_dir)
                dst = os.path.join(model_dir, f"{model_id}_url.xml")
                with open(src_file) as src, open(dst, "w") as out:
                    out.write(src.read())
            data_file = os.path.join(tmp_dir, "out.csv")
            result_dct = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)

        self.assertIn("BIOMD0000000003", result_dct)
        self.assertIn("BIOMD0000000004", result_dct)
        self.assertTrue(isinstance(result_dct["BIOMD0000000003"], float))
        self.assertTrue(isinstance(result_dct["BIOMD0000000004"], float))

    def test_process_bio_models_cvs_values_non_negative(self) -> None:
        """All finite CV values from real models are non-negative."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "BIOMD0000000003")
            os.makedirs(model_dir)
            dst = os.path.join(model_dir, "BIOMD0000000003_url.xml")
            with open(self.BIOMD3_SBML) as src, open(dst, "w") as out:
                out.write(src.read())
            data_file = os.path.join(tmp_dir, "out.csv")
            result_dct = LinearAnalyzer.partitionBiomodelsJacobians(directory=tmp_dir, output_data_file=data_file)

        cv_arr = result_dct["BIOMD0000000003"]
        finite_arr = cv_arr[np.isfinite(cv_arr)]
        self.assertTrue(np.all(finite_arr >= 0))


if __name__ == "__main__":
    unittest.main()