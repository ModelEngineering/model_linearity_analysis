'''Integration tests for scripts/make_biomodels_endtime.py.

Tests use the real BioModels directory at cn.BIOMODELS_DIR without mocks.
CSV output is written to a temp directory so the production data file is never touched.
'''
import src.constants as cn
from scripts.make_biomodels_endtime import findFilesWithExtension, main # type: ignore

import os
import pandas as pd  # type: ignore
import sys
import shutil
import tempfile
import unittest

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

# A small, well-known model used as the single unprocessed model in main() tests.
TEST_MODEL = "BIOMD0000000001"


def _all_models_except(excluded: str) -> list:
    '''Return all model directory names in cn.BIOMODELS_DIR except *excluded*.'''
    return [m for m in os.listdir(cn.BIOMODELS_DIR) if m != excluded]


def _make_prepopulated_csv(path: str, model_names: list) -> None:
    '''Write a CSV at *path* containing fake rows for each name in *model_names*.'''
    df = pd.DataFrame({
        cn.COL_MODEL: model_names,
        cn.COL_ENDTIME_SOURCE: [cn.ENDTIME_SOURCE_SEDML] * len(model_names),
        cn.COL_ENDTIME: [10.0] * len(model_names),
    })
    df.to_csv(path, index=False)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestFindFilesWithExtension(unittest.TestCase):
    '''Tests for findFilesWithExtension using real BioModel directories.'''

    def test_returns_list(self) -> None:
        '''Returns a list for a known model directory.'''
        if IGNORE_TESTS:
            return
        result = findFilesWithExtension(TEST_MODEL, ".xml")
        self.assertIsInstance(result, list)

    def test_finds_xml_files(self) -> None:
        '''Returns a non-empty list when XML files exist.'''
        if IGNORE_TESTS:
            return
        result = findFilesWithExtension(TEST_MODEL, ".xml")
        self.assertGreater(len(result), 0)

    def test_returned_items_end_with_extension(self) -> None:
        '''Every returned path ends with the requested extension.'''
        if IGNORE_TESTS:
            return
        result = findFilesWithExtension(TEST_MODEL, ".xml")
        for path in result:
            self.assertTrue(path.endswith(".xml"), msg=f"Unexpected path: {path}")

    def test_returned_items_are_strings(self) -> None:
        '''Every returned path is a string.'''
        if IGNORE_TESTS:
            return
        result = findFilesWithExtension(TEST_MODEL, ".xml")
        for path in result:
            self.assertIsInstance(path, str)

    def test_nonexistent_extension_returns_empty(self) -> None:
        '''Returns an empty list when no files match the extension.'''
        if IGNORE_TESTS:
            return
        result = findFilesWithExtension(TEST_MODEL, ".zzz_nonexistent")
        self.assertEqual(result, [])

    def test_manifest_xml_excluded_when_filtering_sedml(self) -> None:
        '''Filtering for .sedml does not accidentally return .xml files.'''
        if IGNORE_TESTS:
            return
        result = findFilesWithExtension(TEST_MODEL, ".sedml")
        for path in result:
            self.assertTrue(path.endswith(".sedml"), msg=f"Unexpected path: {path}")


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestMain(unittest.TestCase):
    '''Integration tests for main() using real BioModels data and a temp output path.'''

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "biomodels_endtime.csv")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _prepopulate_all_except_test_model(self) -> None:
        '''Write a CSV pre-populated with every model except TEST_MODEL.'''
        _make_prepopulated_csv(self.output_path, _all_models_except(TEST_MODEL))

    def test_returns_dataframe(self) -> None:
        '''main() returns a pandas DataFrame.'''
        if IGNORE_TESTS:
            return
        self._prepopulate_all_except_test_model()
        result = main(self.output_path)
        self.assertIsInstance(result, pd.DataFrame)

    def test_csv_file_is_written(self) -> None:
        '''main() writes a CSV file at the given output path.'''
        if IGNORE_TESTS:
            return
        self._prepopulate_all_except_test_model()
        main(self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_csv_has_required_columns(self) -> None:
        '''Written CSV contains the required columns.'''
        if IGNORE_TESTS:
            return
        self._prepopulate_all_except_test_model()
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        for col in [cn.COL_MODEL, cn.COL_ENDTIME_SOURCE, cn.COL_ENDTIME]:
            self.assertIn(col, written_df.columns)

    def test_test_model_appears_in_csv(self) -> None:
        '''The newly processed TEST_MODEL appears in the written CSV.'''
        if IGNORE_TESTS:
            return
        self._prepopulate_all_except_test_model()
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        self.assertIn(TEST_MODEL, written_df[cn.COL_MODEL].tolist())

    def test_end_time_is_positive(self) -> None:
        '''The end_time recorded for TEST_MODEL is strictly positive.'''
        if IGNORE_TESTS:
            return
        self._prepopulate_all_except_test_model()
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        row = written_df[written_df[cn.COL_MODEL] == TEST_MODEL].iloc[0]
        self.assertGreater(row[cn.COL_ENDTIME], 0.0)

    def test_end_time_is_finite(self) -> None:
        '''The end_time recorded for TEST_MODEL is finite.'''
        if IGNORE_TESTS:
            return
        self._prepopulate_all_except_test_model()
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        row = written_df[written_df[cn.COL_MODEL] == TEST_MODEL].iloc[0]
        import math
        self.assertTrue(math.isfinite(row[cn.COL_ENDTIME]))

    def test_end_time_source_is_valid(self) -> None:
        '''The end_time_source for TEST_MODEL is one of the recognised source constants.'''
        if IGNORE_TESTS:
            return
        valid_sources = {
            cn.ENDTIME_SOURCE_SEDML,
            cn.ENDTIME_SOURCE_STEADYSTATE,
            cn.ENDTIME_SOURCE_RECIROCAL_MIN_EIGENVALUE,
            cn.ENDTIME_SOURCE_MAX_MEDIAN_CV,
        }
        self._prepopulate_all_except_test_model()
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        row = written_df[written_df[cn.COL_MODEL] == TEST_MODEL].iloc[0]
        self.assertIn(row[cn.COL_ENDTIME_SOURCE], valid_sources)

    def test_previously_processed_models_preserved_in_csv(self) -> None:
        '''Pre-existing rows from the input CSV are preserved in the written CSV.'''
        if IGNORE_TESTS:
            return
        sentinel_model = "BIOMD_SENTINEL"
        existing_df = pd.DataFrame({
            cn.COL_MODEL: [sentinel_model],
            cn.COL_ENDTIME_SOURCE: [cn.ENDTIME_SOURCE_SEDML],
            cn.COL_ENDTIME: [5.0],
        })
        # Pre-populate with sentinel plus all real models except TEST_MODEL.
        all_except = _all_models_except(TEST_MODEL)
        _make_prepopulated_csv(self.output_path, [sentinel_model] + all_except)
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        self.assertIn(sentinel_model, written_df[cn.COL_MODEL].tolist())

    def test_already_processed_model_not_duplicated(self) -> None:
        '''A model already in the CSV appears exactly once after main() runs.'''
        if IGNORE_TESTS:
            return
        all_models = os.listdir(cn.BIOMODELS_DIR)
        _make_prepopulated_csv(self.output_path, all_models)
        main(self.output_path)
        written_df = pd.read_csv(self.output_path)
        counts = written_df[cn.COL_MODEL].value_counts()
        for count in counts:
            self.assertEqual(count, 1)

    def test_no_csv_written_when_all_models_processed(self) -> None:
        '''main() does not overwrite the output CSV when every model is already processed.'''
        if IGNORE_TESTS:
            return
        all_models = os.listdir(cn.BIOMODELS_DIR)
        _make_prepopulated_csv(self.output_path, all_models)
        mtime_before = os.path.getmtime(self.output_path)
        main(self.output_path)
        mtime_after = os.path.getmtime(self.output_path)
        self.assertEqual(mtime_before, mtime_after)


if __name__ == "__main__":
    unittest.main()
