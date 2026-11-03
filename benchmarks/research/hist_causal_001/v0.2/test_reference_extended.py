import unittest
from fractions import Fraction as Q
import reference as h

class ExtendedIdentifiability(unittest.TestCase):
    def test_all_model_context_distributions_normalize(self):
        for m in h.MODELS:
            for c in h.CONTEXTS:
                self.assertEqual(sum(h.dist(m,c).values(),Q(0)),Q(1))
    def test_observational_common_cause_transfer_alias(self):
        self.assertEqual(h.dist('common_cause','observational'),h.dist('open_transfer','observational'))
    def test_randomization_separates_common_cause(self):
        self.assertNotEqual(h.dist('common_cause','do_open_1'),h.dist('open_transfer','do_open_1'))
    def test_sealed_history_and_leak_equivalent_without_audit(self):
        suite=('observational','do_open_0','do_open_1','do_sealed_0','do_sealed_1')
        self.assertTrue(h.equivalent('sealed_history','sealed_leak',suite))
    def test_audit_separates_sealed_history_and_leak(self):
        self.assertNotEqual(h.dist('sealed_history','sealed_audit_1'),h.dist('sealed_leak','sealed_audit_1'))
    def test_local_memory_only_separated_by_within_person_probe(self):
        self.assertEqual(h.dist('null','observational'),h.dist('local_memory','observational'))
        self.assertNotEqual(h.dist('null','same_person_memory_1'),h.dist('local_memory','same_person_memory_1'))
    def test_future_exposure_is_invariant_for_all_models(self):
        expected=h.dist('null','future_exposure_0')
        for m in h.MODELS:
            self.assertEqual(h.dist(m,'future_exposure_0'),expected)
    def test_full_probe_menu_separates_all_pairs(self):
        self.assertEqual(h.score(h.CONTEXTS)['separated_pairs'],15)
    def test_weighted_optimum_identifies_all_models(self):
        result=h.min_cost_suite()
        self.assertEqual(result['cost_exact'],'9/2')
        self.assertEqual(h.score(result['contexts'])['separated_pairs'],15)
    def test_score_is_order_invariant(self):
        self.assertEqual(h.score(h.CONTEXTS)['separated_pairs'],h.score(tuple(reversed(h.CONTEXTS)))['separated_pairs'])

if __name__=='__main__': unittest.main(verbosity=2)
