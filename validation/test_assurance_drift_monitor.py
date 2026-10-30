import unittest
import numpy as np

from assurance_drift_monitor import (
    binomial_upper_tail, log_mixture_e_value, monitor_e_process,
    monitor_bonferroni_rolling, preflight_scope, run_benchmark,
)


class AssuranceDriftMonitorTests(unittest.TestCase):
    def test_exact_binomial_tails(self):
        self.assertAlmostEqual(binomial_upper_tail(1, 1, 0.2), 0.2)
        self.assertAlmostEqual(binomial_upper_tail(0, 5, 0.2), 1.0)
        self.assertAlmostEqual(binomial_upper_tail(5, 5, 0.2), 0.2**5)

    def test_e_process_mixture_is_one_at_zero_observations(self):
        self.assertAlmostEqual(log_mixture_e_value([], 0.02, (0.03, 0.1)), 0.0)

    def test_composite_null_one_step_e_process_is_supermartingale(self):
        p0 = 0.02
        alternatives = (0.03, 0.05, 0.1, 0.2)
        for true_rate in (0.0, 0.01, 0.02):
            expected = 0.0
            for q in alternatives:
                expected += true_rate * (q / p0) + (1.0 - true_rate) * ((1.0-q)/(1.0-p0))
            expected /= len(alternatives)
            self.assertLessEqual(expected, 1.0 + 1e-12)

    def test_e_value_grows_under_sufficiently_many_violations(self):
        obs = np.ones(80, dtype=np.int8)
        loge = log_mixture_e_value(obs, 0.02, (0.03, 0.05, 0.1, 0.2))
        self.assertGreater(loge, np.log(100))
        alarm = monitor_e_process(obs, null_rate=0.02, alternatives=(0.03, 0.05, 0.1, 0.2), alpha=0.01)
        self.assertTrue(alarm["alarm"])

    def test_e_process_does_not_alarm_on_empty_stream(self):
        result = monitor_e_process([], null_rate=0.02, alternatives=(0.03, 0.05), alpha=0.01)
        self.assertFalse(result["alarm"])
        self.assertIsNone(result["time"])

    def test_scope_epoch_and_expiry_fail_closed(self):
        kwargs = dict(evidence_scope="issuer/risk/A", requested_scope="issuer/risk/A",
                      evidence_epoch="epoch-2", requested_epoch="epoch-2", expired=False)
        self.assertEqual(preflight_scope(**kwargs)["status"], "continue_monitoring")
        self.assertEqual(preflight_scope(**{**kwargs, "requested_epoch": "epoch-3"})["status"], "refuse_unresolved")
        self.assertEqual(preflight_scope(**{**kwargs, "expired": True})["reason"], "evidence_expired")
        self.assertEqual(preflight_scope(**{**kwargs, "requested_scope": "other"})["status"], "refuse_unresolved")

    def test_bonferroni_monitor_detects_many_recent_violations(self):
        obs = np.array([0]*100 + [1]*20 + [0]*30, dtype=np.int8)
        result = monitor_bonferroni_rolling(obs, null_rate=0.02, alpha=0.01, window=20, step=5)
        self.assertTrue(result["alarm"])
        self.assertLessEqual(result["p_value"], result["per_look_alpha"])

    def test_invalid_null_alternative_and_alpha_rejected(self):
        with self.assertRaises(ValueError):
            log_mixture_e_value([0], 0.02, (0.01, 0.1))
        with self.assertRaises(ValueError):
            monitor_e_process([0], null_rate=0.02, alternatives=(0.05,), alpha=1.0)

    def test_small_benchmark_is_reproducible(self):
        a = run_benchmark(repeats=30, horizon=200, change_after=50, seed=4)
        b = run_benchmark(repeats=30, horizon=200, change_after=50, seed=4)
        self.assertEqual(a, b)
        self.assertIn("abrupt_future_0.100", a["results"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
