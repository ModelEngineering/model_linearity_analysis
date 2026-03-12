"""
Tests for RoadRunnerMaker class.
"""

import os
import sys
import unittest

import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from roadrunner_maker import RoadRunnerMaker, NO_SPECIFICATION  # type: ignore

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


class TestRoadRunnerMakerInit(unittest.TestCase):
    """Tests for RoadRunnerMaker.__init__."""

    def test_defaults_from_constants(self) -> None:
        """Default timing parameters match constants."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        self.assertEqual(maker._start_time, cn.START_TIME)
        self.assertEqual(maker._end_time, cn.END_TIME)
        self.assertEqual(maker._num_points, cn.NUM_POINTS)

    def test_custom_timing_params_stored(self) -> None:
        """Custom start_time, end_time, and num_points are stored correctly."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL, start_time=1.0, end_time=5.0, num_points=20)
        self.assertEqual(maker._start_time, 1.0)
        self.assertEqual(maker._end_time, 5.0)
        self.assertEqual(maker._num_points, 20)

    def test_roadrunner_kept_by_default(self) -> None:
        """RoadRunner instance is retained when is_keep_roadrunner=True (default)."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        self.assertTrue(hasattr(maker, "_roadrunner"))

    def test_roadrunner_deleted_when_not_kept(self) -> None:
        """RoadRunner instance is deleted after __init__ when is_keep_roadrunner=False."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL, is_keep_roadrunner=False)
        self.assertFalse(hasattr(maker, "_roadrunner"))

    def test_specification_stored_for_string(self) -> None:
        """Model string is stored in _roadrunner_specification."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        self.assertEqual(maker._roadrunner_specification, ANTIMONY_MODEL)

    def test_specification_is_sentinel_for_rr_instance(self) -> None:
        """_roadrunner_specification is NO_SPECIFICATION when initialized from a RoadRunner instance."""
        rr = te.loada(ANTIMONY_MODEL)
        maker = RoadRunnerMaker(rr)
        self.assertEqual(maker._roadrunner_specification, NO_SPECIFICATION)

    def test_invalid_specification_raises(self) -> None:
        """Passing an unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            RoadRunnerMaker(12345)


class TestRoadRunnerMakerRoadrunnerProperty(unittest.TestCase):
    """Tests for RoadRunnerMaker.roadrunner property."""

    def test_returns_roadrunner_when_present(self) -> None:
        """roadrunner property returns the RoadRunner instance when it exists."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        rr = maker.roadrunner
        self.assertTrue(hasattr(rr, "getFloatingSpeciesIds"))

    def test_returns_same_instance(self) -> None:
        """roadrunner property returns the same object on repeated access."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        self.assertIs(maker.roadrunner, maker.roadrunner)

    def test_roadrunner_from_rr_instance(self) -> None:
        """roadrunner property works when initialized with a RoadRunner instance."""
        rr = te.loada(ANTIMONY_MODEL)
        maker = RoadRunnerMaker(rr)
        self.assertIs(maker.roadrunner, rr)


class TestRoadRunnerMakerDeleteRoadRunner(unittest.TestCase):
    """Tests for RoadRunnerMaker.deleteRoadRunner."""

    def test_deletes_roadrunner(self) -> None:
        """deleteRoadRunner removes the _roadrunner attribute."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        maker.deleteRoadRunner()
        self.assertFalse(hasattr(maker, "_roadrunner"))

    def test_delete_is_idempotent(self) -> None:
        """Calling deleteRoadRunner twice does not raise."""
        maker = RoadRunnerMaker(ANTIMONY_MODEL)
        maker.deleteRoadRunner()
        maker.deleteRoadRunner()  # should not raise


class TestRoadRunnerMakerLoadModel(unittest.TestCase):
    """Tests for RoadRunnerMaker._loadModel."""

    def setUp(self) -> None:
        self.maker = RoadRunnerMaker(ANTIMONY_MODEL)

    def test_load_antimony(self) -> None:
        """Antimony strings load successfully."""
        rr = self.maker._loadModel(ANTIMONY_MODEL)
        self.assertTrue(hasattr(rr, "getFloatingSpeciesIds"))

    def test_load_sbml(self) -> None:
        """SBML strings (containing <?xml) load successfully."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        rr = self.maker._loadModel(sbml_str)
        self.assertTrue(hasattr(rr, "getFloatingSpeciesIds"))

    def test_load_antimony_has_correct_species(self) -> None:
        """Loaded Antimony model contains the expected floating species."""
        rr = self.maker._loadModel(ANTIMONY_MODEL)
        species = rr.getFloatingSpeciesIds()
        self.assertIn("S1", species)
        self.assertIn("S2", species)


if __name__ == "__main__":
    unittest.main()
