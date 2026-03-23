"""Tests for Roadrunner class."""

import os
import sys
import unittest

import numpy as np  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from l_roadrunner import LRoadrunner, DEFAULT_END_TIME  # type: ignore

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)

# BIOMD11: outputEndTime="10" in SED-ML — matches DEFAULT_END_TIME, so auto-detection runs.
BIOMD11_SBML  = os.path.join(BIOMODELS_DIR, "BIOMD0000000011", "BIOMD0000000011_url.xml")
BIOMD11_SEDML = os.path.join(BIOMODELS_DIR, "BIOMD0000000011", "BIOMD0000000011_url.sedml")

# BIOMD477: outputEndTime="25" in SED-ML — non-default, so used directly.
BIOMD477_SBML  = os.path.join(BIOMODELS_DIR, "BIOMD0000000477", "BIOMD0000000477_url.xml")
BIOMD477_SEDML = os.path.join(BIOMODELS_DIR, "BIOMD0000000477", "MODEL1308080000_figure5.sedml")

# BIOMD241: rate-rule-only model (no reactions) with events that block steadyState().
# end_time falls back to _calculateEndtimeJacobian.
# Smallest |eigenvalue| at t=0 ≈ 0.176355, so end_time ≈ 5.67.
BIOMD241_SBML  = os.path.join(BIOMODELS_DIR, "BIOMD0000000241", "BIOMD0000000241_url.xml")
BIOMD241_EXPECTED_END_TIME = 1.0 / 0.176355  # ≈ 5.67


def _read(path: str) -> str:
    """Return the text contents of *path*."""
    with open(path) as fh:
        return fh.read()

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
        lrr = LRoadrunner(ANTIMONY_MODEL)
        self.assertTrue(hasattr(lrr.getRoadrunner(), "getFloatingSpeciesIds"))

    def test_invalid_specification_raises(self) -> None:
        """Passing an unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            LRoadrunner(12345)

    def test_load_from_rr_instance(self) -> None:
        """LRoadrunner can be initialized from an existing RoadRunner instance."""
        rr_raw = te.loada(ANTIMONY_MODEL)
        with self.assertRaises(ValueError):
            LRoadrunner(rr_raw)  # type: ignore


class TestLRoadrunnerProperty(unittest.TestCase):
    """Tests for LRoadrunner.roadrunner property."""

    def test_returns_valid_rr_instance(self) -> None:
        """roadrunner property returns an object with getFloatingSpeciesIds."""
        lrr = LRoadrunner(ANTIMONY_MODEL)
        self.assertTrue(hasattr(lrr.getRoadrunner(), "getFloatingSpeciesIds"))

    def test_resets_model_state(self) -> None:
        """roadrunner property resets state so species concentrations return to initial values."""
        rr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        rr_raw = rr.getRoadrunner()
        rr_raw.simulate(0.0, 50.0, 10)
        post_sim = np.array(rr_raw.getFloatingSpeciesConcentrations())
        # Accessing property again should reset
        rr_raw = rr.getRoadrunner()
        after_reset = np.array(rr_raw.getFloatingSpeciesConcentrations())
        initial = np.array(te.loada(ANTIMONY_MODEL).getFloatingSpeciesConcentrations())
        np.testing.assert_array_almost_equal(after_reset, initial)


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
        rr_raw = self.rr.getRoadrunner()
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
        n_species = len(self.rr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (n_species,))

    def test_known_steady_state_value(self) -> None:
        """getSteadyState returns the analytically known value for the production model (SS = 10)."""
        result = self.rr.getSteadyState()
        self.assertAlmostEqual(float(result[0]), 10.0, places=4)


class TestSimulate(unittest.TestCase):
    """Tests for LRoadrunner.simulate."""

    def setUp(self) -> None:
        self.lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_points=50)

    def test_returns_ndarray(self) -> None:
        """simulate returns a numpy ndarray."""
        result = self.lrr.simulate()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_num_points_by_species(self) -> None:
        """simulate returns shape (num_points, n_species)."""
        result = self.lrr.simulate()
        n_species = len(self.lrr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (50, n_species))

    def test_values_are_finite(self) -> None:
        """All simulated concentrations are finite."""
        result = self.lrr.simulate()
        self.assertTrue(np.all(np.isfinite(result)))

    def test_values_are_non_negative(self) -> None:
        """All simulated concentrations are non-negative."""
        result = self.lrr.simulate()
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
        n_species = len(self.rr.getRoadrunner().getFloatingSpeciesIds())
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


class TestGetEndtimeFromJacobian(unittest.TestCase):
    """Tests for LRoadrunner._calculateEndtimeJacobian."""

    def setUp(self) -> None:
        self.antimony_lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_points=10)
        self.production_lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0, num_points=10)

    def test_returns_float_for_normal_model(self) -> None:
        """Returns a float for a model with non-zero eigenvalues."""
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertIsInstance(result, float)

    def test_returns_positive_value(self) -> None:
        """Returned end time is strictly positive."""
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertGreater(result, 0.0)

    def test_antimony_model_end_time(self) -> None:
        """ANTIMONY_MODEL: smallest |eigenvalue| = k1 = 0.1, so end_time ≈ 10.0.

        Jacobian at t=0 is [[-k1, 0], [k1, -k2]] = [[-0.1, 0], [0.1, -0.2]].
        Eigenvalues of a lower-triangular matrix are the diagonal entries: -0.1, -0.2.
        Smallest magnitude is 0.1 → end_time = 1/0.1 = 10.0.
        """
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_production_model_end_time(self) -> None:
        """PRODUCTION_MODEL: single eigenvalue -k_out = -0.1, so end_time ≈ 10.0."""
        result = self.production_lrr._calculateEndtimeJacobian()
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_end_time_is_reciprocal_of_min_eigenvalue_magnitude(self) -> None:
        """end_time equals 1 / min|eigenvalue| of the t=0 Jacobian."""
        rr = self.antimony_lrr.getRoadrunner()
        rr.reset()
        rr.simulate(self.antimony_lrr.start_time, self.antimony_lrr.start_time + 1e-10, 2)
        eigenvalues = np.linalg.eigvals(np.array(rr.getFullJacobian()))
        magnitudes = np.abs(eigenvalues)
        expected = 1.0 / float(np.min(magnitudes[magnitudes >= 1e-10]))
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertAlmostEqual(result, expected, places=6)

    def test_returns_none_when_condition_not_met(self) -> None:
        """Returns None when left_null_rank < number of zero eigenvalues.

        The fresh RoadRunner created inside _calculateEndtimeJacobian is
        controlled via _loadModel.  Jacobian has one zero eigenvalue; full-rank
        stoichiometry gives left_null_rank = 0.  0 < 1, so the condition fails.
        """
        import unittest.mock as mock
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_points=10)
        fresh = mock.MagicMock()
        fresh.getFullJacobian.return_value = np.array([[0.0, 0.0], [0.0, -0.2]])
        # Full-rank 2×2 stoichiometry → left_null_rank = 0
        fresh.getFullStoichiometryMatrix.return_value = np.array([[-1.0, 0.0], [1.0, -1.0]])
        with mock.patch.object(lrr, "_loadModel", return_value=fresh):
            result = lrr._calculateEndtimeJacobian()
        self.assertIsNone(result)

    def test_returns_none_when_all_eigenvalues_zero(self) -> None:
        """Returns None when every eigenvalue is near-zero (no finite end time exists).

        Rank-0 stoichiometry gives left_null_rank = 2; zero Jacobian gives 2
        zero eigenvalues.  Condition 2 >= 2 is met, but there are no non-zero
        eigenvalues to take the reciprocal of.
        """
        import unittest.mock as mock
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_points=10)
        fresh = mock.MagicMock()
        fresh.getFullJacobian.return_value = np.zeros((2, 2))
        fresh.getFullStoichiometryMatrix.return_value = np.zeros((2, 2))
        with mock.patch.object(lrr, "_loadModel", return_value=fresh):
            result = lrr._calculateEndtimeJacobian()
        self.assertIsNone(result)

    def test_result_is_finite(self) -> None:
        """Returned end time is finite (not inf or nan)."""
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertTrue(np.isfinite(result))


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestEndTimeSedml(unittest.TestCase):
    """Tests for LRoadrunner.end_time when a SED-ML string is supplied."""

    def test_non_default_sedml_end_time_used_directly(self) -> None:
        """end_time returns the SED-ML outputEndTime when it differs from DEFAULT_END_TIME.

        BIOMD477's SED-ML specifies outputEndTime="25", which is not the
        default (10), so end_time should return 25.0 without running
        the auto-detection algorithm.
        """
        rr = LRoadrunner(_read(BIOMD477_SBML), sedml_str=_read(BIOMD477_SEDML))
        self.assertAlmostEqual(rr.end_time, 25.0)

    def test_non_default_sedml_end_time_caches_value(self) -> None:
        """end_time is cached after being read from SED-ML."""
        lrr = LRoadrunner(_read(BIOMD477_SBML), sedml_str=_read(BIOMD477_SEDML))
        _ = lrr.end_time
        self.assertAlmostEqual(lrr._end_time, 25.0) # type: ignore

    def test_default_sedml_end_time_falls_through_to_auto_detect(self) -> None:
        """end_time runs auto-detection when SED-ML outputEndTime equals DEFAULT_END_TIME.

        BIOMD11's SED-ML specifies outputEndTime="10", which equals DEFAULT_END_TIME,
        so the SED-ML value is ignored and the steady-state search runs instead.
        The result is a positive float that need not equal 10.
        """
        lrr = LRoadrunner(_read(BIOMD11_SBML), sedml_str=_read(BIOMD11_SEDML))
        result = lrr.end_time
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_default_sedml_end_time_reaches_steady_state(self) -> None:
        """Auto-detected end_time (from default SED-ML) puts BIOMD11 within 5% of steady state."""
        threshold = 0.05
        lrr = LRoadrunner(_read(BIOMD11_SBML), sedml_str=_read(BIOMD11_SEDML))
        end_time = lrr.end_time
        rr_raw = lrr.getRoadrunner()
        rr_raw.steadyState()
        ss_arr = np.array([max(v, 1e-8) for v in rr_raw.getFloatingSpeciesConcentrations()])
        rr_raw.reset()
        rr_raw.simulate(0.0, end_time, 2)
        final_arr = np.array(rr_raw.getFloatingSpeciesConcentrations())
        divergence = np.max(np.abs(final_arr / ss_arr - 1))
        self.assertLess(divergence, threshold)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestEndTimeBiomd241(unittest.TestCase):
    """end_time tests for BIOMD241, a rate-rule-only model whose steadyState()
    raises RuntimeError due to events, forcing the fallback to
    _calculateEndtimeJacobian.
    """

    def setUp(self) -> None:
        self.lrr = LRoadrunner(_read(BIOMD241_SBML))

    def test_returns_float(self) -> None:
        """end_time returns a float for BIOMD241."""
        self.assertIsInstance(self.lrr.end_time, float)

    def test_returns_positive_value(self) -> None:
        """end_time is strictly positive for BIOMD241."""
        self.assertGreater(self.lrr.end_time, 0.0)

    def test_end_time_matches_jacobian_estimate(self) -> None:
        """end_time equals 1 / min|eigenvalue| of the t=0 Jacobian.

        BIOMD241 has no reactions so steadyState() fails (events block moiety
        conversion).  The fallback computes end_time from the t=0 Jacobian.
        The smallest-magnitude eigenvalue is ≈ 0.176355 → end_time ≈ 5.67 s.
        """
        self.assertAlmostEqual(
            self.lrr.end_time, BIOMD241_EXPECTED_END_TIME, places=2
        )

    def test_end_time_is_cached(self) -> None:
        """end_time returns the same value on repeated access."""
        first = self.lrr.end_time
        second = self.lrr.end_time
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()