import unittest
from fractions import Fraction as F
import berger_boos as b

class BergerBoosTests(unittest.TestCase):
    def test_probability_kernels_normalize_exactly(self):
        for theta in b.THETA_GRID:
            self.assertEqual(sum(b.support_probs(12,theta),F(0)),1)

    def test_confidence_sets_cover_each_declared_nuisance_value(self):
        for theta in b.THETA_GRID:
            coverage=b.exact_coverage(20,theta,b.THETA_GRID,F(1,100))
            self.assertGreaterEqual(coverage,F(99,100))

    def test_berger_boos_false_rejection_bounded_for_each_grid_value(self):
        for theta in b.THETA_GRID:
            size=b.exact_rejection_probability(20,20,theta,F(1,20),F(1,100))
            self.assertLessEqual(size,F(1,20))

    def test_empty_confidence_set_fails_conservatively(self):
        self.assertEqual(b.berger_boos_pvalue(4,0,4,0,grid=tuple()),F(1))

    def test_adjustment_dominates_all_pvalues_in_confidence_set(self):
        n1,n2,x,y=12,12,5,3
        c=b.confidence_set(n1,x,b.THETA_GRID,F(1,100))
        if c:
            adjusted=b.berger_boos_pvalue(n1,x,n2,y,b.THETA_GRID,F(1,100))
            self.assertGreaterEqual(adjusted,max(b.lower_tail_pvalue(n2,y,t) for t in c))

    def test_exact_report_checks_coverage_and_size(self):
        out=b.verify(12,12)
        self.assertTrue(all(r["coverage_at_least_1_minus_beta"] for r in out["null_checks"]))
        self.assertTrue(all(r["size_at_most_alpha"] for r in out["null_checks"]))
        self.assertGreaterEqual(F(out["alternative_power"]),F(0))

if __name__=="__main__":
    unittest.main(verbosity=2)
