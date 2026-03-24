"""
Tests for LinearAnalyzer class.
"""

import os
import shutil
import sys
import tempfile
import unittest
from typing import ClassVar

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from jacobian_collection import JacobianCollection  # type: ignore
from linear_analyzer import LinearAnalyzer, ClusterResult  # type: ignore
from clustered_jacobian_collection import ClusteredJacobianCollection  # type: ignore
from jacobian_collection import JacobianCollection as JC  # type: ignore

IGNORE_TESTS = False
ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
BIOMD1_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000001", "BIOMD0000000001_url.xml")
BIOMD300_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000300", "BIOMD0000000300_url.xml")
BIOMD4_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000004", "BIOMD0000000004_url.xml")
BIOMD206_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000206", "BIOMD0000000206_url.xml")
BIOMD241_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000241", "BIOMD0000000241_url.xml")
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)


def _load_sbml(path: str) -> str:
    """Read an SBML file and return its contents as a string."""
    with open(path) as f:
        return f.read()


class TestLinearAnalyzerInit(unittest.TestCase):
    """Tests for LinearAnalyzer.__init__."""

    def test_defaults_stored(self) -> None:
        """Default simulation parameters are stored on the instance."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.start, 0)
        self.assertIsNone(analyzer.end)  # None triggers auto-detect of steady-state end time
        self.assertEqual(analyzer.num_point, 100)

    def test_custom_params_stored(self) -> None:
        """Custom start, end, and num_point are stored correctly."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, start=1, end=5, num_point=50)
        self.assertEqual(analyzer.start, 1)
        self.assertEqual(analyzer.end, 5)
        self.assertEqual(analyzer.num_point, 50)

    def test_model_stored(self) -> None:
        """The model string is stored on the instance."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.model, ANTIMONY_MODEL)

    def test_jacobian_collection_created(self) -> None:
        """A JacobianCollection is created during __init__."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        self.assertIsInstance(analyzer._jacobian_collection, JacobianCollection)

    def test_jacobian_collection_shape(self) -> None:
        """The JacobianCollection has the expected shape for the ANTIMONY_MODEL."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.shape, (10, 2, 2))  # 2 floating species: S1, S2

    def test_init_sbml(self) -> None:
        """Initializing with an SBML string (from Antimony) loads correctly."""
        if IGNORE_TESTS:
            return
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
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertIsInstance(result, ClusterResult)
        self.assertTrue(hasattr(result, "clusters"))
        self.assertTrue(hasattr(result, "max_cv"))

    def test_cluster_count_equals_n_cluster(self) -> None:
        """The clusters list has exactly n_cluster elements."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)

    def test_each_cluster_is_ndarray(self) -> None:
        """Each element in clusters is a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for cluster in result.clusters:
            self.assertIsInstance(cluster, np.ndarray)

    def test_cluster_ndim(self) -> None:
        """Each cluster array has 3 dimensions (n_i, n_species, n_species)."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for cluster in result.clusters:
            self.assertEqual(cluster.ndim, 3)

    def test_cluster_species_dims(self) -> None:
        """The last two dimensions of each cluster match n_species (2 for ANTIMONY_MODEL)."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=2)
        for cluster in result.clusters:
            self.assertEqual(cluster.shape[1], 2)
            self.assertEqual(cluster.shape[2], 2)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=4)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, self.n_points)

    def test_max_cv_is_float(self) -> None:
        """max_cv in ClusterResult is a float."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertIsInstance(result.max_cv, float)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv is non-negative."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=2)
        self.assertGreaterEqual(result.max_cv, 0.0)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        with self.assertRaises(ValueError):
            analyzer.partitionJacobians(n_cluster=10)

    def test_n_cluster_one(self) -> None:
        """With n_cluster=1, returns a single cluster containing all timepoints."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=1)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].shape[0], self.n_points)

    def test_n_cluster_equals_n_points(self) -> None:
        """With n_cluster == n_points, each cluster has at least one timepoint."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobians(n_cluster=self.n_points)
        self.assertEqual(len(result.clusters), self.n_points)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, self.n_points)


class TestPartitionJacobiansSequentially(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionJacobiansSequentially."""

    def setUp(self) -> None:
        self.n_points = 20
        self.analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=self.n_points)

    def test_returns_clustered_jacobian_collection(self) -> None:
        """partitionJacobiansSequentially returns a ClusteredJacobianCollection."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        self.assertIsInstance(result, ClusteredJacobianCollection)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """The jacobian_collections list has exactly n_cluster elements."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        self.assertEqual(len(result.jacobian_collections), 3)

    def test_each_cluster_is_jacobian_collection(self) -> None:
        """Each element in jacobian_collections is a JacobianCollection."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for jc in result.jacobian_collections:
            self.assertIsInstance(jc, JC)

    def test_cluster_ndim(self) -> None:
        """Each cluster's jacobian_arr has 3 dimensions."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for jc in result.jacobian_collections:
            self.assertEqual(jc.jacobian_arr.ndim, 3)

    def test_cluster_species_dims(self) -> None:
        """The last two dimensions of each cluster's jacobian_arr match n_species."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        for jc in result.jacobian_collections:
            self.assertEqual(jc.jacobian_arr.shape[1], 2)
            self.assertEqual(jc.jacobian_arr.shape[2], 2)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        total = sum(jc.jacobian_arr.shape[0] for jc in result.jacobian_collections)
        self.assertEqual(total, self.n_points)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv is non-negative."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=2)
        self.assertGreaterEqual(result.max_cv, 0.0)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=5)
        with self.assertRaises(ValueError):
            analyzer.partitionJacobiansSequentially(n_cluster=10)

    def test_clusters_are_contiguous_in_time(self) -> None:
        """Concatenating cluster jacobian_arrs in order reconstructs the original array."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(
            [jc.jacobian_arr for jc in result.jacobian_collections], axis=0
        )
        np.testing.assert_array_equal(
            reconstructed, self.analyzer._jacobian_collection.jacobian_arr
        )

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=1)
        self.assertEqual(len(result.jacobian_collections), 1)
        np.testing.assert_array_equal(
            result.jacobian_collections[0].jacobian_arr,
            self.analyzer._jacobian_collection.jacobian_arr,
        )

    def test_n_cluster_equals_n_points(self) -> None:
        """With n_cluster == n_points, every cluster has exactly one timepoint."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=self.n_points)
        self.assertEqual(len(result.jacobian_collections), self.n_points)
        for jc in result.jacobian_collections:
            self.assertEqual(jc.jacobian_arr.shape[0], 1)

    def test_max_cv_le_unpartitioned(self) -> None:
        """Sequential partitioning into multiple clusters yields max_cv <= single-segment CV."""
        if IGNORE_TESTS:
            return
        single = self.analyzer.partitionJacobiansSequentially(n_cluster=1)
        multi = self.analyzer.partitionJacobiansSequentially(n_cluster=4)
        self.assertLessEqual(multi.max_cv, single.max_cv + 1e-9)

    def test_no_timepoints_skipped_or_repeated(self) -> None:
        """Each timepoint appears in exactly one cluster."""
        if IGNORE_TESTS:
            return
        result = self.analyzer.partitionJacobiansSequentially(n_cluster=3)
        total = sum(jc.jacobian_arr.shape[0] for jc in result.jacobian_collections)
        self.assertEqual(total, self.n_points)
        reconstructed = np.concatenate(
            [jc.jacobian_arr for jc in result.jacobian_collections], axis=0
        )
        self.assertEqual(reconstructed.shape[0], self.n_points)


class TestMakeBioModelAnalyzers(unittest.TestCase):
    """Tests for LinearAnalyzer.makeBioModelAnalyzers."""

    def test_returns_list(self) -> None:
        """makeBioModelAnalyzers returns a list."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertIsInstance(results, list)

    def test_empty_directory_returns_empty_list(self) -> None:
        """An empty directory yields an empty list."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertEqual(results, [])

    def test_processes_sbml_in_subdirectory(self) -> None:
        """A valid SBML file in a subdirectory produces one (model_id, LinearAnalyzer) tuple."""
        if IGNORE_TESTS:
            return
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
        if IGNORE_TESTS:
            return
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
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_dir = os.path.join(tmp_dir, "BADMODEL")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "BADMODEL_url.xml"), "w") as f:
                f.write("<?xml version='1.0'?><not_sbml/>")
            results = LinearAnalyzer.makeBioModelAnalyzers(directory=tmp_dir)
        self.assertEqual(results, [])

    def test_skips_manifest_xml(self) -> None:
        """manifest.xml files are not loaded as SBML models."""
        if IGNORE_TESTS:
            return
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
        if IGNORE_TESTS:
            return
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


FIRST_FIVE_MODEL_IDS = [
    "BIOMD0000000026",
    "BIOMD0000000027",
    "BIOMD0000000028",
    "BIOMD0000000029",
    "BIOMD0000000030",
]


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestPartitionBiomodelsJacobians(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionBiomodelsJacobians.

    Uses the first 5 models from the BioModels final directory.  Some models
    may fail to load; tests that require at least one success will assert on
    ``len(df) > 0`` rather than a hard-coded model ID.
    """
    def setUp(self) -> None:
        """Copy the first 5 BioModels into a shared temporary directory."""
        self._tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir_obj.name
        for model_id in FIRST_FIVE_MODEL_IDS:
            src = os.path.join(BIOMODELS_DIR, model_id)
            dst = os.path.join(self.tmp_dir, model_id)
            shutil.copytree(src, dst)

    def tearDown(self) -> None:
        self._tmp_dir_obj.cleanup()

    def _data_file(self) -> str:
        """Return a per-test CSV path inside the shared tmp directory."""
        return os.path.join(self.tmp_dir, f"{self._testMethodName}.csv")

    def test_returns_dataframe(self) -> None:
        """partitionBiomodelsJacobians returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file()
        )
        self.assertIsInstance(df, pd.DataFrame)

    def test_empty_directory_returns_empty_dataframe(self) -> None:
        """An empty directory yields an empty DataFrame."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as empty_dir:
            data_file = os.path.join(empty_dir, "out.csv")
            df = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=empty_dir, output_data_file=data_file
            )
        self.assertEqual(len(df), 0)

    def test_index_contains_model_ids(self) -> None:
        """DataFrame index contains at least one of the first-five model IDs."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file()
        )
        self.assertTrue(
            any(mid in df.index for mid in FIRST_FIVE_MODEL_IDS),
            f"Expected at least one of {FIRST_FIVE_MODEL_IDS} in index {list(df.index)}",
        )

    def test_has_max_cv_and_end_time_columns(self) -> None:
        """DataFrame has max_cv and end_time columns."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file()
        )
        self.assertIn("max_cv", df.columns)
        self.assertIn("end_time", df.columns)

    def test_values_are_floats(self) -> None:
        """max_cv values for successfully loaded models are floats."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file()
        )
        self.assertGreater(len(df), 0, "No models succeeded; cannot check value types")
        first_model = df.index[0]
        self.assertIsInstance(df.loc[first_model, "max_cv"], (int, float))

    def test_csv_is_created(self) -> None:
        """The output CSV file is created."""
        if IGNORE_TESTS:
            return
        data_file = self._data_file()
        LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file
        )
        self.assertTrue(os.path.isfile(data_file))

    def test_csv_is_valid_and_readable(self) -> None:
        """The written CSV can be read back by pandas and contains model IDs."""
        if IGNORE_TESTS:
            return
        data_file = self._data_file()
        LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file
        )
        df_on_disk = pd.read_csv(data_file, header=None, index_col=0)
        self.assertTrue(
            any(mid in df_on_disk.index for mid in FIRST_FIVE_MODEL_IDS),
            "CSV does not contain any expected model ID",
        )

    def test_skips_invalid_model(self) -> None:
        """A directory with an invalid XML file is skipped without raising."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_dir = os.path.join(tmp_dir, "BADMODEL")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "BADMODEL_url.xml"), "w") as f:
                f.write("<?xml version='1.0'?><not_sbml/>")
            data_file = os.path.join(tmp_dir, "out.csv")
            df = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertEqual(len(df), 0)

    def test_excluded_models_are_skipped(self) -> None:
        """Models in excluded_models are not processed."""
        if IGNORE_TESTS:
            return
        excluded = FIRST_FIVE_MODEL_IDS[:3]
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir,
            output_data_file=self._data_file(),
            excluded_models=excluded,
        )
        for model_id in excluded:
            self.assertNotIn(model_id, df.index)

    def test_sequential_partition_flag(self) -> None:
        """is_sequential_partition=True runs without error and returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir,
            output_data_file=self._data_file(),
            is_sequential_partition=True,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(
            any(mid in df.index for mid in FIRST_FIVE_MODEL_IDS),
            "No first-five model IDs found in sequential-partition result",
        )

    def test_already_processed_model_is_skipped(self) -> None:
        """A model already in the CSV is not reprocessed (row count stays the same)."""
        if IGNORE_TESTS:
            return
        data_file = self._data_file()
        df1 = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file
        )
        df2 = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file
        )
        self.assertEqual(len(df1), len(df2))


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestWithBioModels(unittest.TestCase):
    """Integration tests using real SBML files from temp-biomodels."""

    def test_init_biomd3(self) -> None:
        """LinearAnalyzer initializes correctly for BIOMD300 (3 floating species)."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.shape, (10, 3, 3))

    def test_init_biomd1_timepoints(self) -> None:
        """JacobianCollection timepoints length equals num_point."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=10)
        self.assertEqual(len(analyzer._jacobian_collection.timepoint_arr), 10)

    def test_partition_jacobians_biomd3(self) -> None:
        """partitionJacobians works on a real SBML model."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=20)
        result = analyzer.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)
        total = sum(c.shape[0] for c in result.clusters)
        self.assertEqual(total, 20)

    def test_sequential_partition_biomd3(self) -> None:
        """partitionJacobiansSequentially produces contiguous segments on a real SBML model."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=20)
        result = analyzer.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(
            [jc.jacobian_arr for jc in result.jacobian_collections], axis=0
        )
        np.testing.assert_array_equal(
            reconstructed, analyzer._jacobian_collection.jacobian_arr
        )

    def test_partition_biomodels_with_real_models(self) -> None:
        """partitionBiomodelsJacobians returns valid max_cv values for real models."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            for model_id, src_file in [
                ("BIOMD0000000300", BIOMD300_SBML),
                ("BIOMD0000000004", BIOMD4_SBML),
            ]:
                model_dir = os.path.join(tmp_dir, model_id)
                os.makedirs(model_dir)
                dst = os.path.join(model_dir, f"{model_id}_url.xml")
                with open(src_file) as src, open(dst, "w") as out:
                    out.write(src.read())
            data_file = os.path.join(tmp_dir, "out.csv")
            df = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file
            )
        self.assertIn("BIOMD0000000300", df.index)
        self.assertIn("BIOMD0000000004", df.index)
        self.assertGreaterEqual(df.loc["BIOMD0000000300", "max_cv"], 0.0)  # type: ignore[arg-type]

    def test_init_biomd241(self) -> None:
        """LinearAnalyzer initializes correctly for BIOMD241."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD241_SBML), num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.ndim, 3)
        self.assertEqual(arr.shape[0], 10)

    def test_partition_jacobians_biomd241(self) -> None:
        """partitionJacobians works on BIOMD241."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD241_SBML), num_point=20)
        result = analyzer.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result.clusters), 3)
        self.assertEqual(sum(c.shape[0] for c in result.clusters), 20)

    def test_sequential_partition_biomd241(self) -> None:
        """partitionJacobiansSequentially produces contiguous segments on BIOMD241."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD241_SBML), num_point=20)
        result = analyzer.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(
            [jc.jacobian_arr for jc in result.jacobian_collections], axis=0
        )
        np.testing.assert_array_equal(
            reconstructed, analyzer._jacobian_collection.jacobian_arr
        )


if __name__ == "__main__":
    unittest.main()
