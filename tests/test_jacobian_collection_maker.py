"""
Tests for JacobianCollectionMaker class.
"""

import os
import sys
import tellurium as te # type: ignore
import unittest
from unittest.mock import MagicMock

import numpy as np  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from jacobian_collection import JacobianCollection # type: ignore
from jacobian_collection_maker import JacobianCollectionMaker  # type: ignore

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


class TestJacobianCollectionMakerInit(unittest.TestCase):
    """Tests for JacobianCollectionMaker.__init__."""

    def test_defaults_from_constants(self) -> None:
        """Default timing parameters match constants."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL)
        self.assertEqual(maker._start_time, cn.START_TIME)
        self.assertEqual(maker._end_time, cn.END_TIME)
        self.assertEqual(maker._num_points, cn.NUM_POINTS)

    def test_custom_params_stored(self) -> None:
        """Custom start, end, and num_points are stored correctly."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, start_time=1.0, end_time=5.0, num_points=20)
        self.assertEqual(maker._start_time, 1.0)
        self.assertEqual(maker._end_time, 5.0)
        self.assertEqual(maker._num_points, 20)

    def test_roadrunner_deleted_when_not_kept(self) -> None:
        """RoadRunner instance is deleted after __init__ when is_keep_roadrunner=False."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, is_keep_roadrunner=False)
        self.assertFalse(hasattr(maker, "_roadrunner"))

    def test_roadrunner_kept_when_requested(self) -> None:
        """RoadRunner instance is retained when is_keep_roadrunner=True."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, is_keep_roadrunner=True)
        self.assertTrue(hasattr(maker, "_roadrunner"))
        self.assertIsNotNone(maker._roadrunner)

    def test_roadrunner_specification_stored_for_string(self) -> None:
        """Model string is stored in _roadrunner_specification for later restoration."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL)
        self.assertEqual(maker._roadrunner_specification, ANTIMONY_MODEL)

    def test_roadrunner_specification_none_for_rr_instance(self) -> None:
        """_roadrunner_specification is None when initialized from a RoadRunner instance."""
        rr = te.loada(ANTIMONY_MODEL)
        maker = JacobianCollectionMaker(rr, is_keep_roadrunner=True)
        self.assertIsNone(maker._roadrunner_specification)

    def test_invalid_specification_raises(self) -> None:
        """Passing an unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            JacobianCollectionMaker(12345)


class TestJacobianCollectionMakerLoadModel(unittest.TestCase):
    """Tests for JacobianCollectionMaker._loadModel."""

    def setUp(self) -> None:
        self.maker = JacobianCollectionMaker(ANTIMONY_MODEL, is_keep_roadrunner=True)

    def test_load_antimony(self) -> None:
        """Antimony strings load successfully."""
        rr = self.maker._loadModel(ANTIMONY_MODEL)
        self.assertTrue(hasattr(rr, "getFloatingSpeciesIds"))

    def test_load_sbml(self) -> None:
        """SBML strings (containing <?xml) load successfully."""
        import tellurium as te
        rr_antimony = te.loada(ANTIMONY_MODEL)
        sbml_str = rr_antimony.getSBML()
        rr = self.maker._loadModel(sbml_str)
        self.assertTrue(hasattr(rr, "getFloatingSpeciesIds"))


class TestJacobianCollectionMakerRestoreRoadRunner(unittest.TestCase):
    """Tests for JacobianCollectionMaker.restoreRoadRunner."""

    def test_restore_from_string(self) -> None:
        """RoadRunner can be restored after deletion when created from a string."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, is_keep_roadrunner=False)
        self.assertFalse(hasattr(maker, "_roadrunner"))
        maker.restoreRoadRunner()
        self.assertTrue(hasattr(maker, "_roadrunner"))
        self.assertIsNotNone(maker._roadrunner)

    def test_restore_is_noop_when_already_present(self) -> None:
        """restoreRoadRunner is a no-op when _roadrunner already exists."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, is_keep_roadrunner=True)
        original_rr = maker._roadrunner
        maker.restoreRoadRunner()
        self.assertIs(maker._roadrunner, original_rr)

    def test_restore_raises_when_created_from_rr_instance(self) -> None:
        """restoreRoadRunner raises ValueError when initialized with a RoadRunner (no string saved)."""
        import tellurium as te
        rr = te.loada(ANTIMONY_MODEL)
        maker = JacobianCollectionMaker(rr, is_keep_roadrunner=False)
        with self.assertRaises(ValueError):
            maker.restoreRoadRunner()


class TestJacobianCollectionMakerMakeCollection(unittest.TestCase):
    """Tests for JacobianCollectionMaker.makeCollection."""

    def setUp(self) -> None:
        self.maker = JacobianCollectionMaker(
            ANTIMONY_MODEL,
            start_time=0.0,
            end_time=5.0,
            num_points=11,
            is_keep_roadrunner=True,
        )

    def test_returns_jacobian_collection(self) -> None:
        """makeCollection returns a JacobianCollection instance."""
        collection = self.maker.makeCollection()
        self.assertIsInstance(collection, JacobianCollection)

    def test_jacobian_shape(self) -> None:
        """Returned jacobian_arr has shape (num_points, n_species, n_species)."""
        collection = self.maker.makeCollection()
        n_points = collection.jacobian_arr.shape[0]
        n_species = collection.jacobian_arr.shape[1]
        self.assertEqual(n_points, 11)
        self.assertEqual(collection.jacobian_arr.ndim, 3)
        self.assertEqual(collection.jacobian_arr.shape[1], collection.jacobian_arr.shape[2])
        # S1 and S2 are floating species
        self.assertEqual(n_species, 2)

    def test_timepoints_length(self) -> None:
        """Timepoints array length matches num_points."""
        collection = self.maker.makeCollection()
        self.assertEqual(len(collection.timepoints), 11)

    def test_timepoints_values(self) -> None:
        """First and last timepoints are approximately start and end times."""
        collection = self.maker.makeCollection()
        self.assertAlmostEqual(collection.timepoints[0], 0.0, places=5)
        self.assertAlmostEqual(collection.timepoints[-1], 5.0, places=5)

    def test_jacobian_values_are_finite(self) -> None:
        """All Jacobian entries are finite (no NaN or Inf)."""
        collection = self.maker.makeCollection()
        self.assertTrue(np.all(np.isfinite(collection.jacobian_arr)))

    def test_make_collection_from_rr_instance(self) -> None:
        """makeCollection works when initialized with a RoadRunner instance directly."""
        import tellurium as te
        rr = te.loada(ANTIMONY_MODEL)
        maker = JacobianCollectionMaker(rr, is_keep_roadrunner=True)
        collection = maker.makeCollection()
        self.assertIsInstance(collection, JacobianCollection)
        self.assertEqual(collection.jacobian_arr.ndim, 3)


if __name__ == "__main__":
    unittest.main()
