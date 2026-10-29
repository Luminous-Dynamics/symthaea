import unittest
import numpy as np

from crosscheck_huber_catoni_reference import clip_interval, compare_bounds, make_streams


class ReferenceCrosscheckUtilityTests(unittest.TestCase):
    def test_reference_interval_is_intersected_with_clean_support(self):
        self.assertEqual(clip_interval(float("-inf"), float("inf")), (0.0, 1.0))
        self.assertEqual(clip_interval(-0.25, 0.7), (0.0, 0.7))
        self.assertEqual(clip_interval(0.2, 1.25), (0.2, 1.0))

    def test_invalid_support_is_rejected(self):
        with self.assertRaises(ValueError):
            clip_interval(0.0, 1.0, (1.0, 1.0))

    def test_comparison_reports_endpointwise_error(self):
        result = compare_bounds((0.1, 0.8), (0.1001, 0.799))
        self.assertAlmostEqual(result["lower_abs_error"], 0.0001)
        self.assertAlmostEqual(result["upper_abs_error"], 0.001)
        self.assertAlmostEqual(result["max_abs_error"], 0.001)

    def test_nonfinite_mismatch_is_not_hidden(self):
        result = compare_bounds((0.0, 1.0), (float("-inf"), 1.0))
        self.assertIsNone(result["lower_abs_error"])
        self.assertIsNone(result["max_abs_error"])
        self.assertTrue(result["nonfinite_mismatch"])

    def test_streams_are_seeded_and_support_bounded(self):
        first = make_streams(32, 7)
        second = make_streams(32, 7)
        self.assertEqual(set(first), {"bernoulli_037", "alternating_01", "constant_040"})
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertTrue(np.all((first[name] >= 0.0) & (first[name] <= 1.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
