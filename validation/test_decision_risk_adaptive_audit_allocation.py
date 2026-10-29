import unittest
import numpy as np
import decision_risk_adaptive_audit_allocation as b

class AdaptiveAuditAllocationTests(unittest.TestCase):
    def test_beta_mixture_cs_contains_empirical_mean(self):
        lo, hi = b.beta_mixture_cs(500, 1000, b.DELTA / 8)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)
        self.assertLess(hi - lo, 0.2)

    def test_adaptive_and_uniform_use_same_fail_closed_classification(self):
        rng = np.random.default_rng(11)
        truth = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.95])
        streams = np.vstack([rng.binomial(1, p, size=b.MAX_PER_CELL) for p in truth]).astype(np.int8)
        exact = b.run_strategy(streams, truth.copy(), strategy='uniform')
        adaptive = b.run_strategy(streams, truth.copy(), strategy='adaptive')
        self.assertIn(exact['status'], ('admit', 'reject', 'refuse_unresolved'))
        self.assertIn(adaptive['status'], ('admit', 'reject', 'refuse_unresolved'))
        self.assertGreaterEqual(exact['total_observations'], len(truth) * b.PILOT)
        self.assertGreaterEqual(adaptive['total_observations'], len(truth) * b.PILOT)

    def test_admission_never_happens_for_clear_error_in_small_control(self):
        rng = np.random.default_rng(42)
        truth = np.array([0.015, 0.035, 0.075, 0.14, 0.27, 0.50, 0.73, 0.94])
        model = truth.copy(); model[5] += b.EPSILON_ALLOWANCE + 0.02
        streams = np.vstack([rng.binomial(1, p, size=b.MAX_PER_CELL) for p in truth]).astype(np.int8)
        for method in ('uniform','adaptive'):
            out = b.run_strategy(streams, model, strategy=method)
            self.assertNotEqual(out['status'], 'admit')

    def test_run_protocol_and_unsafe_scenarios_are_fail_closed(self):
        out = b.run(repeats=50)
        self.assertEqual(out['protocol']['cells'], 8)
        for method in ('uniform','adaptive'):
            for scenario in ('near_over_budget_high_variance_cell','clearly_over_budget_high_variance_cell'):
                self.assertEqual(out['results'][scenario][method]['false_admission_count'], 0)

if __name__ == '__main__': unittest.main(verbosity=2)
