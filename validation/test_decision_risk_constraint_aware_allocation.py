import unittest
import numpy as np
import decision_risk_constraint_aware_allocation as b

class ConstraintAwareAllocationTests(unittest.TestCase):
    def test_allocation_methods_share_valid_stopping_outcomes(self):
        rng=np.random.default_rng(5)
        truth=np.array([.02,.05,.1,.2,.4,.5,.7,.95])
        streams=np.vstack([rng.binomial(1,p,size=b.MAX_PER_CELL) for p in truth]).astype(np.int8)
        for method in ('uniform','max_error_upper','widest_active_ci','marginal_ci_reduction'):
            result=b.run_strategy(streams,truth.copy(),method)
            self.assertIn(result['status'],('admit','reject','refuse_unresolved'))
            self.assertGreaterEqual(result['total_observations'],len(truth)*b.PILOT)
            self.assertLessEqual(max(result['per_cell_observations']),b.MAX_PER_CELL)

    def test_invalid_strategy_rejected(self):
        with self.assertRaises(ValueError): b.run_strategy(np.zeros((2,100),dtype=np.int8),np.zeros(2), 'oracle')

    def test_fixed_seed_safe_and_unsafe_scenarios(self):
        out=b.run(repeats=50)
        for method in ('uniform','max_error_upper','widest_active_ci','marginal_ci_reduction'):
            for scenario in ('near_over_budget_high_variance_cell','clearly_over_budget_high_variance_cell'):
                self.assertEqual(out['results'][scenario][method]['false_admissions'],0)
        self.assertEqual(out['protocol']['cells'],8)

if __name__=='__main__': unittest.main(verbosity=2)
