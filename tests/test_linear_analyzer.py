"""
Tests for LinearAnalyzer class.
"""

import os
import sys
import tempfile
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from jacobian_collection import JacobianCollection  # type: ignore
from linear_analyzer import LinearAnalyzer, ClusterResult  # type: ignore

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
BIOMD1_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000001", "BIOMD0000000001_url.xml")
BIOMD3_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000003", "BIOMD0000000003_url.xml")
BIOMD4_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000004", "BIOMD0000000004_url.xml")
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)


def _load_sbml(path: str) -> str:
    """Read an SBML file and return its contents as a string."""
    with open(path) as f:
        return f.read()


class TestLinearAnalyzerInit(unittest.TestCase):
    """Tests for LinearAnalyzer.__init__."""

    def test_defaults_stored(self) -> None:
        """Default simulation parameters are stored on the instance."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.start, 0)
        self.assertEqual(analyzer.end, 10)
        self.assertEqual(analyzer.num_point, 100)

    def test_custom_params_stored(self) -> None:
        """Custom start, end, and num_point are stored correctly."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, start=1, end=5, num_point=50)
        self.assertEqual(analyzer.start, 1)
        self.assertEqual(analyzer.end, 5)
        self.assertEqual(analyzer.num_point, 50)

    def test_model_stored(self) -> None:
        """The model string is stored on the instance."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.model, ANTIMONY_MODEL)

    def test_jacobian_collection_created(self) -> None:
        """A JacobianCollection is created during __init__."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        self.assertIsInstance(analyzer._jacobian_collection, JacobianCollection)

    def test_jacobian_collection_shape(self) -> None:
        """The JacobianCollection has the expected shape for the ANTIMONY_MODEL."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.shape, (10, 2, 2))  # 2 floating species: S1, S2

    def test_init_sbml(self) -> None:
        """Initializing with an SBML string (from Antimony) loads correctly."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        analyzer = LinearAnalyzer(sbml_str, num_point=5)
        self.assertIsInstance(analyzer._jacobian_collection, JacobianCollection)
        self.assertEqual(analyzer._jacobian_collection.jacobian_arr.ndim, 3)


class TestPartitionJacobians(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionJacobians."""

    def setUp(self) -> None:
        self.n_points = 20
        self.analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=self.n_points)

    def test_returns_cluster_result(self) -> None:
        """partitionJacobians returns a ClusterResult namedtuple."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertIsInstance(result, ClusterResult)
        self.assertTrue(hasattr(result, "clusters"))
        self.assertTrue(hasattr(result, "max_cv"))

    def test_cluster_count_equals_n_cluster(self) -> None:
        """The clusters list has exactly n_cluster elements."""
        result = self.analyzer.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)

    def test_each_cluster_is_ndarray(self) -> None:
        """Each element in clusters is a numpy ndarray."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for cluster in result.clusters:
            self.assertIsInstance(cluster, np.ndarray)

    def test_cluster_ndim(self) -> None:
        """Each cluster array has 3 dimensions (n_i, n_species, n_species)."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for cluster in result.clusters:
            self.assertEqual(cluster.ndim, 3)

    def test_cluster_species_dims(self) -> None:
        """The last two dimensions of each cluster match n_species (2 for ANTIMONY_MODEL)."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for cluster in result.clusters:
            self.assertEqual(cluster.shape[1], 2)
            self.assertEqual(cluster.shape[2], 2)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        result = self.analyzer.partitionJacobians(n_cluster=4)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, self.n_points)

    def test_max_cv_is_float(self) -> None:
        """max_cv in ClusterResult is a float."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertIsInstance(result.max_cv, float)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv is non-negative."""
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertGreaterEqual(result.max_cv, 0.0)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        with self.assertRaises(ValueError):
            analyzer.partitionJacobians(n_cluster=10)

    def test_n_cluster_one(self) -> None:
        """With n_cluster=1, returns a single cluster containing all timepoints."""
        result = self.analyzer.partitionJacobians(n_cluster=1)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].shape[0], self.n_points)

    def test_n_cluster_equals_n_points(self) -> None:
        """With n_cluster == n_points, each cluster has at least one timepoint."""
        result = self.analyzer.partitionJacobians(n_cluster=self.n_points)
        self.assertEqual(len(result.clusters), self.n_points)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, self.n_points)


class TestPartitionJacobiansSequentially(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionJacobiansSequentially."""

    def setUp(self) -> None:
        self.n_points = 20
        self.analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=self.n_points)

    def test_returns_cluster_result(self) -> None:
        """partitionJacobiansSequentially returns a ClusterResult namedtuple."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        self.assertIsInstance(result, ClusterResult)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """The clusters list has exactly n_cluster elements."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)

    def test_each_cluster_is_ndarray(self) -> None:
        """Each element in clusters is a numpy ndarray."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for cluster in result.clusters:
            self.assertIsInstance(cluster, np.ndarray)

    def test_cluster_ndim(self) -> None:
        """Each cluster array has 3 dimensions."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for cluster in result.clusters:
            self.assertEqual(cluster.ndim, 3)

    def test_cluster_species_dims(self) -> None:
        """The last two dimensions of each cluster match n_species."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for cluster in result.clusters:
            self.assertEqual(cluster.shape[1], 2)
            self.assertEqual(cluster.shape[2], 2)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, self.n_points)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv is non-negative."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        self.assertGreaterEqual(result.max_cv, 0.0)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        with self.assertRaises(ValueError):
            analyzer.partitionJacobiansSequentially(n_cluster=10)

    def test_clusters_are_contiguous_in_time(self) -> None:
        """Concatenating clusters in order reconstructs the original Jacobian array."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(result.clusters, axis=0)
        np.testing.assert_array_equal(
            reconstructed, self.analyzer._jacobian_collection.jacobian_arr
        )

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=1)
        self.assertEqual(len(result.clusters), 1)
        np.testing.assert_array_equal(
            result.clusters[0], self.analyzer._jacobian_collection.jacobian_arr
        )

    def test_n_cluster_equals_n_points(self) -> None:
        """With n_cluster == n_points, every cluster has exactly one timepoint."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=self.n_points)
        self.assertEqual(len(result.clusters), self.n_points)
        for cluster in result.clusters:
            self.assertEqual(cluster.shape[0], 1)

    def test_max_cv_le_unpartitioned(self) -> None:
        """Sequential partitioning into multiple clusters yields max_cv <= single-segment CV."""
        single = self.analyzer.partitionJacobiansSequentially(n_cluster=1)
        multi = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        self.assertLessEqual(multi.max_cv, single.max_cv + 1e-9)

    def test_no_timepoints_skipped_or_repeated(self) -> None:
        """Each timepoint appears in exactly one cluster."""
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, self.n_points)
        reconstructed = np.concatenate(result.clusters, axis=0)
        self.assertEqual(reconstructed.shape[0], self.n_points)


class TestMakeBioModelAnalyzers(unittest.TestCase):
    """Tests for LinearAnalyzer.makeBioModelAnalyzers."""

    def test_returns_list(self) -> None:
        """makeBioModelAnalyzers returns a list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertIsInstance(results, list)

    def test_empty_directory_returns_empty_list(self) -> None:
        """An empty directory yields an empty list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertEqual(results, [])

    def test_processes_sbml_in_subdirectory(self) -> None:
        """A valid SBML file in a subdirectory produces one (model_id, LinearAnalyzer) tuple."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertEqual(len(results), 1)
        model_id, analyzer = results[0]
        self.assertEqual(model_id, "MODEL0001")
        self.assertIsInstance(analyzer, LinearAnalyzer)

    def test_result_tuple_types(self) -> None:
        """Each result tuple is (str, LinearAnalyzer)."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "TESTMODEL")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "TESTMODEL_url.xml"), "w") as f:
                f.write(sbml_str)
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
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

    def test_skips_manifest_xml(self) -> None:
        """manifest.xml files are not loaded as SBML models."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            # Write only a manifest.xml — should be ignored
            with open(os.path.join(model_dir, "manifest.xml"), "w") as f:
                f.write(sbml_str)
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertEqual(results, [])

    def test_analyzer_has_jacobian_collection(self) -> None:
        """Each returned LinearAnalyzer has a populated JacobianCollection."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        _, analyzer = results[0]
        self.assertIsInstance(analyzer._jacobian_collection, JacobianCollection)
        self.assertGreater(analyzer._jacobian_collection.jacobian_arr.size, 0)


class TestPartitionBiomodelsJacobians(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionBiomodelsJacobians."""

    def test_returns_series(self) -> None:
        """partitionBiomodelsJacobians returns a pd.Series."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertIsInstance(ser, pd.Series)

    def test_empty_directory_returns_empty_series(self) -> None:
        """An empty directory yields an empty Series."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertEqual(len(ser), 0)

    def test_index_contains_model_id(self) -> None:
        """Series index contains the processed model ID."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertIn("MODEL0001", ser.index)

    def test_values_are_floats(self) -> None:
        """Series values are floats (max_cv per model)."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertIsInstance(ser["MODEL0001"], float)

    def test_csv_is_created(self) -> None:
        """The output CSV file is created."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
            self.assertTrue(os.path.isfile(data_file))

    def test_csv_is_valid_and_readable(self) -> None:
        """The written CSV can be read back by pandas and contains the model ID."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
            df = pd.read_csv(data_file, header=None, index_col=0)
        self.assertIn("MODEL0001", df.index)

    def test_skips_invalid_model(self) -> None:
        """A directory with an invalid XML file is skipped without raising."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_dir = os.path.join(tmp_dir, "BADMODEL")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "BADMODEL_url.xml"), "w") as f:
                f.write("<?xml version='1.0'?><not_sbml/>")
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertEqual(len(ser), 0)

    def test_excluded_models_are_skipped(self) -> None:
        """Models in excluded_models are not processed."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir,
                output_data_file=data_file,
                excluded_models=["MODEL0001"],
            )
        self.assertNotIn("MODEL0001", ser.index)

    def test_sequential_partition_flag(self) -> None:
        """is_sequential_partition=True runs without error and returns a Series."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir,
                output_data_file=data_file,
                is_sequential_partition=True,
            )
        self.assertIsInstance(ser, pd.Series)
        self.assertIn("MODEL0001", ser.index)

    def test_already_processed_model_is_skipped(self) -> None:
        """A model already in the CSV is not reprocessed."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "MODEL0001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "MODEL0001_url.xml"), "w") as f:
                f.write(sbml_str)
            data_file = os.path.join(tmp_dir, "out.csv")
            # First run — processes the model
            LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
            # Second run — should skip the already-processed model
            ser2 = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        # Second run returns empty new results (model was already processed)
        self.assertEqual(len(ser2), 0)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestWithBioModels(unittest.TestCase):
    """Integration tests using real SBML files from temp-biomodels."""

    def test_init_biomd3(self) -> None:
        """LinearAnalyzer initializes correctly for BIOMD3 (3 floating species)."""
        analyzer = LinearAnalyzer(_load_sbml(BIOMD3_SBML), num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.shape, (10, 3, 3))

    def test_init_biomd1_timepoints(self) -> None:
        """JacobianCollection timepoints length equals num_point."""
        analyzer = LinearAnalyzer(_load_sbml(BIOMD1_SBML), num_point=10)
        self.assertEqual(len(analyzer._jacobian_collection.timepoint_arr), 10)

    def test_partition_jacobians_biomd3(self) -> None:
        """partitionJacobians works on a real SBML model."""
        analyzer = LinearAnalyzer(_load_sbml(BIOMD3_SBML), num_point=20)
        result = analyzer.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, 20)

    def test_sequential_partition_biomd3(self) -> None:
        """partitionJacobiansSequentially produces contiguous segments on a real SBML model."""
        analyzer = LinearAnalyzer(_load_sbml(BIOMD3_SBML), num_point=20)
        result = analyzer.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(result.clusters, axis=0)
        np.testing.assert_array_equal(
            reconstructed, analyzer._jacobian_collection.jacobian_arr
        )

    def test_partition_biomodels_with_real_models(self) -> None:
        """partitionBiomodelsJacobians returns valid max_cv values for real models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for model_id, src_file in [
                ("BIOMD0000000003", BIOMD3_SBML),
                ("BIOMD0000000004", BIOMD4_SBML),
            ]:
                model_dir = os.path.join(tmp_dir, model_id)
                os.makedirs(model_dir)
                dst = os.path.join(model_dir, f"{model_id}_url.xml")
                with open(src_file) as src, open(dst, "w") as out:
                    out.write(src.read())
            data_file = os.path.join(tmp_dir, "out.csv")
            ser = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertIn("BIOMD0000000003", ser.index)
        self.assertIn("BIOMD0000000004", ser.index)
        self.assertGreaterEqual(ser["BIOMD0000000003"], 0.0)
        self.assertGreaterEqual(ser["BIOMD0000000004"], 0.0)


if __name__ == "__main__":
    unittest.main()
