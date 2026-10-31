import unittest
from finite_state_crc_recertification import (
    LinearFiniteCRC, count_vectors, two_type_reference, summarize_named
)


class FiniteStateCRCRecertificationTests(unittest.TestCase):
    def setUp(self):
        self.result = two_type_reference()
        self.engine = self.result["engine"]
        self.ctrl = self.result["controllers"]

    def test_multiset_count_vectors_do_not_enumerate_permutations(self):
        self.assertEqual(count_vectors(2, 2), [(0, 2), (1, 1), (2, 0)])
        self.assertEqual(len(count_vectors(50, 3)), 1326)
        self.assertEqual(len(count_vectors(51, 3)), 1378)

    def test_baseline_standard_and_envelope_crc_match_paper(self):
        standard = summarize_named(self.ctrl["standard_crc"])
        envelope = summarize_named(self.ctrl["envelope_crc"])
        self.assertAlmostEqual(standard["HH"], .75, places=10)
        self.assertAlmostEqual(standard["HL"], 2/3, places=10)
        self.assertAlmostEqual(standard["LL"], .5, places=10)
        self.assertAlmostEqual(envelope["HH"], .5, places=10)
        self.assertAlmostEqual(envelope["HL"], .4, places=10)
        self.assertAlmostEqual(envelope["LL"], .25, places=10)

    def test_one_and_two_equal_share_rounds_match_paper(self):
        one = summarize_named(self.ctrl["equal_share_round_1"])
        two = summarize_named(self.ctrl["equal_share_round_2"])
        self.assertAlmostEqual(one["HH"], .5, places=9)
        self.assertAlmostEqual(one["HL"], 23/60, places=8)
        self.assertAlmostEqual(one["LL"], .2, places=8)
        self.assertAlmostEqual(two["HH"], .5, places=9)
        self.assertAlmostEqual(two["HL"], 17/45, places=8)
        self.assertAlmostEqual(two["LL"], 31/180, places=8)

    def test_every_equal_share_round_is_valid_and_pointwise_less_conservative(self):
        history = self.result["equal_share_history"]
        for before, after in zip(history[:-1], history[1:]):
            self.assertTrue(self.engine.validity_report(after)["valid"])
            for D in self.engine.states:
                self.assertLessEqual(after[D], before[D] + 1e-10)

    def test_limit_matches_closed_form(self):
        got = summarize_named(self.ctrl["equal_share_limit"])
        self.assertAlmostEqual(got["HH"], .5, places=8)
        self.assertAlmostEqual(got["HL"], .375, places=7)
        self.assertAlmostEqual(got["LL"], .125, places=7)
        self.assertTrue(self.result["coordinatewise_admissible"]["limit"])

    def test_coordinate_sweep_finds_other_admissible_controller(self):
        got = summarize_named(self.ctrl["coordinate_sweep_alternate"])
        self.assertAlmostEqual(got["HH"], .5, places=8)
        self.assertAlmostEqual(got["HL"], .4, places=8)
        self.assertAlmostEqual(got["LL"], .1, places=8)
        self.assertTrue(self.result["coordinatewise_admissible"]["alternate"])

    def test_pointwise_minimum_of_incomparable_admissible_controllers_is_invalid(self):
        report = self.result["validity"]["pointwise_minimum"]
        self.assertFalse(report["valid"])
        self.assertAlmostEqual(report["augmented_totals"]["1,2"][0], 1.525, places=7)

    def test_multi_risk_controller_is_jointly_valid(self):
        engine = LinearFiniteCRC([[1., .5], [.5, 1.]], [.5, .4], n_calibration=3)
        baseline = engine.envelope_crc()
        self.assertTrue(engine.validity_report(baseline)["valid"])
        one = engine.recertify_equal_share(baseline)
        self.assertTrue(engine.validity_report(one)["valid"])
        swept = engine.coordinate_sweep(baseline)
        self.assertTrue(engine.validity_report(swept)["valid"])
        self.assertTrue(engine.is_coordinatewise_admissible(swept))

    def test_joint_controller_uses_one_shared_action(self):
        engine = LinearFiniteCRC([[1., .5], [.5, 1.]], [.5, .4], n_calibration=3)
        baseline = engine.envelope_crc()
        for action in baseline.values():
            self.assertIsInstance(action, float)
            self.assertGreaterEqual(action, 0.)
            self.assertLessEqual(action, 1.)

    def test_invalid_starting_controller_fails_closed(self):
        invalid = {(0, 2): 0., (1, 1): 0., (2, 0): 0.}
        self.assertFalse(self.engine.validity_report(invalid)["valid"])
        with self.assertRaises(ValueError):
            self.engine.recertify_equal_share(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
