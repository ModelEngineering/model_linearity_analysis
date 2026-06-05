"""Tests for Timecourse."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from timecourse import Timecourse  # type: ignore

IGNORE_TESTS = False

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_POINT = 11
NUM_SPECIES = 2
MODEL_NAME = "test_model"


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL, model_name=MODEL_NAME)


def _makeTimecourse() -> Timecourse:
    """Return a Timecourse pre-populated with synthetic data (no simulation)."""
    rng = np.random.default_rng(42)
    timepoint_arr = np.linspace(0.0, 10.0, NUM_POINT)
    timecourse_df = pd.DataFrame(
            rng.standard_normal((NUM_POINT, NUM_SPECIES)),
            index=timepoint_arr,
            columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    jacobian_collection_arr = rng.standard_normal((NUM_POINT, NUM_SPECIES, NUM_SPECIES))
    return Timecourse(
            model=_makeModel(),
            timecourse_df=timecourse_df,
            jacobian_collection_arr=jacobian_collection_arr,
    )


class TestTimecourseInit(unittest.TestCase):
    """Tests for Timecourse.__init__."""

    def test_stores_model(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertIsInstance(tc.model, Model)

    def test_stores_start_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), start_time=1.5)
        self.assertEqual(tc.start_time, 1.5)

    def test_stores_num_points(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), num_point=50)
        self.assertEqual(tc.num_point, 50)

    def test_timecourse_df_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertFalse(tc._timecourse_df.empty)

    def test_jacobian_collection_arr_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertGreater(tc._jacobian_collection_arr.size, 0)


class TestTimecourseSerialize(unittest.TestCase):
    """Tests for Timecourse.serialize."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_string_path(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertIsInstance(path, str)

    def test_file_exists_after_serialize(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertTrue(os.path.isfile(path))

    def test_filename_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertIn(MODEL_NAME, os.path.basename(path))

    def test_raises_without_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=Model(ANTIMONY_MODEL))
        with self.assertRaises(ValueError):
            tc.serialize()


class TestTimecourseRoundtrip(unittest.TestCase):
    """Roundtrip tests for Timecourse.serialize / Timecourse.deserialize."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _serializeAndDeserialize(self) -> tuple:
        """Return (original, restored) Timecourse pair via a temp directory."""
        original = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = original.serialize()
        restored = Timecourse.deserialize(path)
        return original, restored

    def test_timecourse_df_roundtrip(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        pd.testing.assert_frame_equal(
                original._timecourse_df, restored._timecourse_df)

    def test_jacobian_collection_arr_roundtrip(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        np.testing.assert_array_equal(
                original._jacobian_collection_arr,
                restored._jacobian_collection_arr)

    def test_start_time_roundtrip(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        self.assertEqual(original.start_time, restored.start_time)

    def test_end_time_roundtrip(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        self.assertEqual(original.end_time, restored.end_time)

    def test_num_points_roundtrip(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        self.assertEqual(original.num_point, restored.num_point)

    def test_model_name_roundtrip(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        self.assertEqual(original.model.model_name, restored.model.model_name)


if __name__ == "__main__":
    unittest.main()
