import unittest
import numpy as np
import decision_risk_frontier_deficit_allocation as b

class FrontierDeficitTests(unittest.TestCase):
    def test_bernoulli_kl_properties(self):
        self.assertAlmostEqual(b.bernoulli_kl(.5, .5), 0.0)
        self.assertGreater(b.bernoulli_kl(.8, .5), 0.0)
        self.assertAlmostEqual(b.bernoulli_kl(0.0, .5), np.log(2.0))

    def test_scheduler_proxy_never_controls_outcome(self):
        rng=np.random.default_rng(21)
        truth=np.array([.015,.035,.075,.14,.27,.50,.73,.94])
        streams=np.vstack([rng.binomial(1,p,size=b.MAX_PER_CELL) for p in truth]).astype(np.int8)
        for method in ('uniform','max_error_upper','widest_active_ci','marginal_ci_reduction','frontier_deficit'):
            result=b.run_strategy(streams,truth.copy(),method)
            self.assertIn(result['status'],('admit','reject','refuse_unresolved'))
            self.assertGreaterEqual(result['total_observations'],len(truth)*b.PILOT)
            self.assertLessEqual(max(result['per_cell_observations']),b.MAX_PER_CELL)

    def test_invalid_strategy_fails_closed(self):
        with self.assertRaises(ValueError):
            b.run_strategy(np.zeros((2,100),dtype=np.int8),np.zeros(2),'oracle')

    def test_fifty_repeat_protocol_has_no_unsafe_admission_on_fixed_seed(self):
        out=b.run(repeats=50)
        self.assertEqual(out['protocol']['cells'],8)
        for scenario in ('near_over_budget_high_variance_cell','clearly_over_budget_high_variance_cell'):
            self.assertEqual(out['results'][scenario]['frontier_deficit']['false_admissions'],0)

if __name__ == '__main__': unittest.main(verbosity=2)
