"""Tests for the standard-library HIST-CAUSAL-001 v0.2 reference."""
import unittest
from fractions import Fraction as Q
import reference as r

class ExactIdentifiability(unittest.TestCase):
    def test_all_kernels_normalize(self):
        for m in r.MODELS:
            for c in r.CONTEXTS:
                self.assertEqual(sum(r.dist(m,c).values(),Q(0)),Q(1))

    def test_observation_does_not_identify_common_cause_vs_transfer(self):
        self.assertEqual(r.dist("common_cause","observational"),
                         r.dist("open_transfer","observational"))

    def test_randomization_separates_common_cause(self):
        self.assertNotEqual(r.dist("common_cause","do_open_1"),
                            r.dist("open_transfer","do_open_1"))

    def test_no_leak_audit_means_history_and_leak_are_equivalent(self):
        suite=("observational","do_open_0","do_open_1","do_sealed_0","do_sealed_1")
        self.assertTrue(r.equivalent("sealed_history","sealed_leak",suite))

    def test_independent_audit_separates_history_and_leak_likelihoods(self):
        self.assertNotEqual(r.dist("sealed_history","sealed_audit_1"),
                            r.dist("sealed_leak","sealed_audit_1"))

    def test_full_suite_separates_all_declared_pairs(self):
        self.assertEqual(r.score(r.CONTEXTS)["separated_pairs"],15)

    def test_minimum_cost_probe_suite(self):
        plan=r.min_cost_suite()
        self.assertEqual(plan["cost_exact"],"9/2")
        self.assertEqual(r.score(plan["contexts"])["separated_pairs"],15)

if __name__ == "__main__":
    unittest.main(verbosity=2)
