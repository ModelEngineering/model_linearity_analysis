"""Tests for Roadrunner class."""

import os
import sys
import unittest

import numpy as np  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from l_roadrunner import LRoadrunner  # type: ignore

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

# Production-degradation model: SS = k_in/k_out = 10; time constant = 1/k_out = 10 s.
PRODUCTION_MODEL = """
-> S1; k_in
S1 -> ; k_out * S1
k_in = 1.0; k_out = 0.1; S1 = 0
"""


class TestLRoadrunnerInit(unittest.TestCase):
    """Tests for LRoadrunner.__init__."""

    def test_defaults_from_constants(self) -> None:
        """Default timing parameters match constants."""
        rr = LRoadrunner(ANTIMONY_MODEL)
        self.assertEqual(rr.start_time, cn.START_TIME)
        self.assertEqual(rr.num_points, cn.NUM_POINTS)

    def test_custom_timing_params_stored(self) -> None:
        """Custom start_time, end_time, and num_points are stored correctly."""
        rr = LRoadrunner(ANTIMONY_MODEL, start_time=1.0, end_time=5.0, num_points=20)
        self.assertEqual(rr.start_time, 1.0)
        self.assertEqual(rr._end_time, 5.0)
        self.assertEqual(rr.num_points, 20)

    def test_end_time_none_by_default(self) -> None:
        """_end_time is None when not explicitly provided."""
        rr = LRoadrunner(ANTIMONY_MODEL)
        self.assertIsNone(rr._end_time)

    def test_roadrunner_instance_stored(self) -> None:
        """Internal RoadRunner instance is created and stored."""
        rr = LRoadrunner(ANTIMONY_MODEL)
        self.assertTrue(hasattr(rr, "_roadrunner"))
        self.assertTrue(hasattr(rr._roadrunner, "getFloatingSpeciesIds"))

    def test_invalid_specification_raises(self) -> None:
        """Passing an unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            LRoadrunner(12345)

    def test_load_from_rr_instance(self) -> None:
        """LRoadrunner can be initialized from an existing RoadRunner instance."""
        rr_raw = te.loada(ANTIMONY_MODEL)
        rr = LRoadrunner(rr_raw)
        self.assertIs(rr._roadrunner, rr_raw)


class TestLRoadrunnerProperty(unittest.TestCase):
    """Tests for LRoadrunner.roadrunner property."""

    def test_returns_valid_rr_instance(self) -> None:
        """roadrunner property returns an object with getFloatingSpeciesIds."""
        rr = LRoadrunner(ANTIMONY_MODEL)
        self.assertTrue(hasattr(rr.roadrunner, "getFloatingSpeciesIds"))

    def test_resets_model_state(self) -> None:
        """roadrunner property resets state so species concentrations return to initial values."""
        rr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        rr_raw = rr.roadrunner
        rr_raw.simulate(0.0, 50.0, 10)
        post_sim = np.array(rr_raw.getFloatingSpeciesConcentrations())
        # Accessing property again should reset
        rr_raw = rr.roadrunner
        after_reset = np.array(rr_raw.getFloatingSpeciesConcentrations())
        initial = np.array(te.loada(ANTIMONY_MODEL).getFloatingSpeciesConcentrations())
        np.testing.assert_array_almost_equal(after_reset, initial)

    def test_returns_same_underlying_instance(self) -> None:
        """roadrunner property returns the same RoadRunner object each call."""
        rr = LRoadrunner(ANTIMONY_MODEL)
        self.assertIs(rr.roadrunner, rr.roadrunner)


class TestLoadModel(unittest.TestCase):
    """Tests for LRoadrunner._loadModel."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(ANTIMONY_MODEL)

    def test_load_antimony(self) -> None:
        """Antimony strings load successfully."""
        rr_raw = self.rr._loadModel(ANTIMONY_MODEL)
        self.assertTrue(hasattr(rr_raw, "getFloatingSpeciesIds"))

    def test_load_sbml(self) -> None:
        """SBML strings (containing <?xml) load successfully."""
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        rr_raw = self.rr._loadModel(sbml_str)
        self.assertTrue(hasattr(rr_raw, "getFloatingSpeciesIds"))

    def test_antimony_has_correct_species(self) -> None:
        """Loaded Antimony model contains the expected floating species."""
        rr_raw = self.rr._loadModel(ANTIMONY_MODEL)
        species = rr_raw.getFloatingSpeciesIds()
        self.assertIn("S1", species)
        self.assertIn("S2", species)


class TestEndTime(unittest.TestCase):
    """Tests for LRoadrunner.end_time property."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(PRODUCTION_MODEL)

    def test_returns_float(self) -> None:
        """end_time returns a float."""
        self.assertIsInstance(self.rr.end_time, float)

    def test_result_is_positive(self) -> None:
        """end_time returns a positive value."""
        self.assertGreater(self.rr.end_time, 0.0)

    def test_explicit_end_time_returned_unchanged(self) -> None:
        """end_time returns the explicitly provided value without computing."""
        rr = LRoadrunner(PRODUCTION_MODEL, end_time=42.0)
        self.assertEqual(rr.end_time, 42.0)

    def test_simulation_reaches_steady_state(self) -> None:
        """Simulating to end_time puts each species within 1% of its steady-state value."""
        threshold = 0.01
        end_time = self.rr.end_time
        rr_raw = self.rr.roadrunner
        rr_raw.steadyState()
        ss_arr = np.array(rr_raw.getFloatingSpeciesConcentrations())
        ss_arr_safe = np.array([max(v, 1e-8) for v in ss_arr])
        rr_raw.reset()
        rr_raw.simulate(0.0, end_time, 2)
        final_arr = np.array(rr_raw.getFloatingSpeciesConcentrations())
        divergence = np.max(np.abs(final_arr / ss_arr_safe - 1))
        self.assertLess(divergence, threshold)

    def test_result_is_cached(self) -> None:
        """end_time returns the same value on repeated access (caches result)."""
        first = self.rr.end_time
        second = self.rr.end_time
        self.assertEqual(first, second)
        self.assertIsNotNone(self.rr._end_time)

    def test_no_steady_state_raises(self) -> None:
        """end_time raises ValueError for a model that cannot reach steady state."""
        # Linear chain that decays completely: steady state is 0 for all species,
        # but the clamping to 1e-8 means the algorithm will always succeed.
        # Use a model with an invalid steady state solver instead.
        rr = LRoadrunner(PRODUCTION_MODEL)
        # Patch to simulate an unreachable steady state by corrupting the model
        # via making steadyState raise. We test the error path indirectly by
        # confirming the property still raises ValueError when propagated.
        import unittest.mock as mock
        with mock.patch.object(rr._roadrunner, "steadyState", side_effect=RuntimeError("no SS")):
            with self.assertRaises(ValueError):
                _ = rr.end_time


class TestGetSteadyState(unittest.TestCase):
    """Tests for LRoadrunner.getSteadyState."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(PRODUCTION_MODEL)

    def test_returns_ndarray(self) -> None:
        """getSteadyState returns a numpy ndarray."""
        result = self.rr.getSteadyState()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_matches_species_count(self) -> None:
        """getSteadyState returns a 1-D array with one entry per floating species."""
        result = self.rr.getSteadyState()
        n_species = len(self.rr.roadrunner.getFloatingSpeciesIds())
        self.assertEqual(result.shape, (n_species,))

    def test_known_steady_state_value(self) -> None:
        """getSteadyState returns the analytically known value for the production model (SS = 10)."""
        result = self.rr.getSteadyState()
        self.assertAlmostEqual(float(result[0]), 10.0, places=4)


class TestSimulate(unittest.TestCase):
    """Tests for LRoadrunner.simulate."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_points=50)

    def test_returns_ndarray(self) -> None:
        """simulate returns a numpy ndarray."""
        result = self.rr.simulate()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_num_points_by_species(self) -> None:
        """simulate returns shape (num_points, n_species)."""
        result = self.rr.simulate()
        n_species = len(self.rr.roadrunner.getFloatingSpeciesIds())
        self.assertEqual(result.shape, (50, n_species))

    def test_values_are_finite(self) -> None:
        """All simulated concentrations are finite."""
        result = self.rr.simulate()
        self.assertTrue(np.all(np.isfinite(result)))

    def test_values_are_non_negative(self) -> None:
        """All simulated concentrations are non-negative."""
        result = self.rr.simulate()
        self.assertTrue(np.all(result >= 0.0))


class TestMakeJacobians(unittest.TestCase):
    """Tests for LRoadrunner.makeJacobians."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_points=10)

    def test_returns_tuple_of_two(self) -> None:
        """makeJacobians returns a tuple of two elements."""
        result = self.rr.makeJacobians()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_jacobians_shape(self) -> None:
        """Jacobians array has shape (num_points, n_species, n_species)."""
        jacobians, _ = self.rr.makeJacobians()
        n_species = len(self.rr.roadrunner.getFloatingSpeciesIds())
        self.assertEqual(jacobians.shape, (self.rr.num_points, n_species, n_species))

    def test_times_shape(self) -> None:
        """Times array has shape (num_points,)."""
        _, times = self.rr.makeJacobians()
        self.assertEqual(times.shape, (self.rr.num_points,))

    def test_times_are_monotonically_increasing(self) -> None:
        """Timepoints are strictly increasing."""
        _, times = self.rr.makeJacobians()
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_jacobians_are_finite(self) -> None:
        """All Jacobian entries are finite."""
        jacobians, _ = self.rr.makeJacobians()
        self.assertTrue(np.all(np.isfinite(jacobians)))

    def test_raises_for_no_floating_species(self) -> None:
        """makeJacobians raises ValueError when the model has no floating species."""
        # Boundary-species-only model: S1 is a boundary species (fixed), no floating species.
        boundary_model = """
$S1 -> $S2; k1*S1
k1 = 0.1; S1 = 10; S2 = 0
"""
        rr = LRoadrunner(boundary_model, end_time=10.0)
        with self.assertRaises(ValueError):
            rr.makeJacobians()


if __name__ == "__main__":
    unittest.main()
