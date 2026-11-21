import unittest

from sign_test_power import (
    critical_rejection_count,
    exact_sign_test_power,
    exact_two_sided_sign_p,
    minimum_n_for_power,
)


class SignTestPowerTests(unittest.TestCase):
    def test_exact_p_known_answers(self):
        self.assertAlmostEqual(exact_two_sided_sign_p(23, 23), 2 / (2**23), places=16)
        self.assertAlmostEqual(exact_two_sided_sign_p(5, 5), 0.0625, places=15)
        self.assertEqual(exact_two_sided_sign_p(4, 2), 1.0)

    def test_n23_power_known_answers(self):
        self.assertAlmostEqual(exact_sign_test_power(23, 0.60), 0.124982, places=6)
        self.assertAlmostEqual(exact_sign_test_power(23, 0.70), 0.439965, places=6)
        self.assertAlmostEqual(exact_sign_test_power(23, 0.75), 0.653728, places=6)
        self.assertAlmostEqual(exact_sign_test_power(23, 0.80), 0.840167, places=6)

    def test_minimum_n_for_80_percent_power(self):
        self.assertEqual(minimum_n_for_power(0.60), 199)
        self.assertEqual(minimum_n_for_power(0.65), 90)
        self.assertEqual(minimum_n_for_power(0.70), 49)
        self.assertEqual(minimum_n_for_power(0.75), 30)
        self.assertEqual(minimum_n_for_power(0.80), 20)

    def test_critical_region(self):
        self.assertEqual(critical_rejection_count(23), 6)
        self.assertEqual(critical_rejection_count(30), 9)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            exact_two_sided_sign_p(0, 0)
        with self.assertRaises(ValueError):
            exact_sign_test_power(20, 1.2)
        with self.assertRaises(ValueError):
            minimum_n_for_power(0.7, target_power=1.0)


if __name__ == "__main__":
    unittest.main()
