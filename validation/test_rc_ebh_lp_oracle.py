import unittest
import numpy as np
from rc_ebh_lp_oracle import (
    compare_rc_to_full_lp, rc_ebh_tails, full_all_subsets_lp,
    verify_all_subset_constraints
)


class RCEBHFullLPOracleTests(unittest.TestCase):
    def test_random_small_families_match_full_all_subset_lp(self):
        rng=np.random.default_rng(20261031)
        tested=0
        for k in range(1,6):
            for _ in range(40):
                e=np.exp(rng.normal(0,2.0,k))
                alpha=float(rng.choice([.01,.025,.05,.1,.2,.5]))
                result=compare_rc_to_full_lp(e,alpha)
                self.assertTrue(result["rc_feasible"],result)
                self.assertTrue(result["objective_match"],result)
                self.assertLessEqual(result["objective_gap"],1e-8)
                tested+=1
        self.assertEqual(tested,200)

    def test_boundary_and_tied_vectors_match(self):
        vectors=[
            [0.0],[1.0],[0.0,0.0],[1.0,1.0],[100.0,1.0],
            [1.0,1.0,1.0,1.0,1.0],[1e-8,1e-4,1,1e4,1e8],
            [0.0,0.5,1.0,2.0,10.0],[1000,100,10,1,.1]
        ]
        for e in vectors:
            for alpha in (.01,.05,.1,.5):
                got=compare_rc_to_full_lp(np.asarray(e),alpha)
                self.assertTrue(got["rc_feasible"],got)
                self.assertTrue(got["objective_match"],got)

    def test_rc_rule_is_feasible_for_every_intersection(self):
        rng=np.random.default_rng(7)
        for k in range(1,6):
            for _ in range(30):
                e=np.exp(rng.normal(0,2,k));q=rc_ebh_tails(e,.1)
                self.assertTrue(verify_all_subset_constraints(e,.1,q)["valid"])

    def test_full_lp_distribution_is_valid(self):
        rng=np.random.default_rng(8)
        for k in range(1,6):
            e=np.exp(rng.normal(0,1.5,k))
            lp=full_all_subsets_lp(e,.05)
            p=lp["probabilities"]
            self.assertAlmostEqual(float(p.sum()),1.,places=8)
            self.assertTrue(np.all(p>=-1e-8))
            self.assertTrue(np.all(np.diff(lp["tails"])<=1e-8))

    def test_expected_rejection_optimum_not_just_feasibility(self):
        e=np.array([9.2,3.1,1.5,1.01,.2])
        for alpha in (.025,.05,.1,.2):
            got=compare_rc_to_full_lp(e,alpha)
            self.assertTrue(got["objective_match"],got)
            self.assertAlmostEqual(got["rc_objective"],got["lp_objective"],places=8)


if __name__=="__main__":
    unittest.main(verbosity=2)
