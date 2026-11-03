"""Finite-sample audit sensitivity tests."""
import unittest
from fractions import Fraction as Q
import audit_power as a

class TestAuditPower(unittest.TestCase):
    def test_binomial_normalization(self):
        for p in (Q(1,100),Q(1,20),Q(1,10),Q(1,5),Q(19,20)):
            for n in (1,2,5,16):
                self.assertEqual(sum(a.binom(n,k,p) for k in range(n+1)),Q(1))
    def test_profiles_normalize(self):
        for sens,fpr in a.AUDITS.values():
            for prior in a.PRIORS.values():
                for truth in ("history","leak"):
                    self.assertEqual(sum(a.profile(8,sens,fpr,truth,prior).values()),Q(1))
    def test_minimum_sample_sensitivities(self):
        self.assertEqual(a.min_n(*a.AUDITS["high"],Q(1,2)),2)
        self.assertEqual(a.min_n(*a.AUDITS["moderate"],Q(1,2)),5)
        self.assertEqual(a.min_n(*a.AUDITS["weak"],Q(1,2)),9)
        self.assertEqual(a.min_n(*a.AUDITS["poor"],Q(1,2)),23)
    def test_prior_odds_change_weak_audit_sample(self):
        sens,fpr=a.AUDITS["weak"]
        self.assertEqual(a.min_n(sens,fpr,Q(1,10)),11)
        self.assertEqual(a.min_n(sens,fpr,Q(1,2)),9)
        self.assertEqual(a.min_n(sens,fpr,Q(9,10)),11)

if __name__=="__main__": unittest.main(verbosity=2)
