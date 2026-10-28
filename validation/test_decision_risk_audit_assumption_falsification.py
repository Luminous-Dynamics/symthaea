import unittest
import numpy as np
import decision_risk_audit_assumption_falsification as b

class AuditAssumptionFalsificationTests(unittest.TestCase):
    def test_iid_control_is_numerically_admitted_under_low_noise(self):
        out = b.run()
        good = out['results']['iid_low_noise_positive_control']
        self.assertTrue(good['admitted_by_numeric_bound'])
        self.assertTrue(good['preflight']['admitted'])
        self.assertLess(good['certificate_regret_bound_bits'], b.REGRET_BUDGET_BITS)

    def test_shared_systematic_bias_falsifies_unverified_iid_bound(self):
        out = b.run()['results']['systematic_bias_adversarial_control']
        self.assertTrue(out['admitted_by_numeric_bound'])
        self.assertTrue(out['unsafe_admission'])
        self.assertGreater(out['true_planning_error_bits_scoring_only'], out['epsilon_allowance_bits'])
        self.assertFalse(out['fail_closed_preflight']['admitted'])
        self.assertEqual(out['fail_closed_preflight']['reason'], 'AUDIT_ASSUMPTIONS_UNVERIFIED')

    def test_missing_cell_can_look_safe_but_support_guard_rejects(self):
        out = b.run()['results']['missing_support_control']
        self.assertTrue(out['naive_incomplete_support_admitted'])
        self.assertEqual(out['observed_cell_count'], 3)
        self.assertFalse(out['fail_closed_preflight']['admitted'])
        self.assertEqual(out['fail_closed_preflight']['reason'], 'AUDIT_SUPPORT_INCOMPLETE')

    def test_manifest_mismatch_invalid_digest_and_missing_receipt_fail_closed(self):
        obs = {f'p{i}::w{j}': [1.0, 1.0] for i in range(2) for j in range(2)}
        common = dict(policy_ids=('p0','p1'), world_ids=('w0','w1'), cell_observations=obs,
                      model_manifest_sha256='a'*64, audit_target_manifest_sha256='a'*64,
                      audit_protocol_sha256='b'*64, assumption_receipt_verified=True)
        self.assertTrue(b.fail_closed_preflight(**common)['admitted'])
        bad = dict(common); bad['audit_target_manifest_sha256'] = 'c'*64
        self.assertEqual(b.fail_closed_preflight(**bad)['reason'], 'AUDIT_TARGET_MANIFEST_MISMATCH')
        bad = dict(common); bad['audit_protocol_sha256'] = 'not-a-digest'
        self.assertEqual(b.fail_closed_preflight(**bad)['reason'], 'INVALID_OR_MISSING_DIGEST')
        bad = dict(common); bad['assumption_receipt_verified'] = False
        self.assertEqual(b.fail_closed_preflight(**bad)['reason'], 'AUDIT_ASSUMPTIONS_UNVERIFIED')

    def test_envelope_function_rejects_missing_extra_invalid_and_out_of_range_cells(self):
        model = np.ones((2,2)); policy_ids=('p0','p1'); world_ids=('w0','w1')
        obs = {f'p{i}::w{j}': [1.0,1.0] for i in range(2) for j in range(2)}
        self.assertGreater(b.empirical_bernstein_epsilon_upper(model, obs, policy_ids, world_ids)['epsilon_upper_bits'], 0.0)
        with self.assertRaises(b.AuditEvidenceError):
            b.empirical_bernstein_epsilon_upper(model, {k:v for k,v in obs.items() if k!='p0::w0'}, policy_ids, world_ids)
        with self.assertRaises(b.AuditEvidenceError):
            b.empirical_bernstein_epsilon_upper(model, {**obs,'p9::w9':[1.0,1.0]}, policy_ids, world_ids)
        bad = dict(obs); bad['p0::w0'] = [1.0, float('nan')]
        with self.assertRaises(b.AuditEvidenceError):
            b.empirical_bernstein_epsilon_upper(model, bad, policy_ids, world_ids)
        bad = dict(obs); bad['p0::w0'] = [1.0, 4.1]
        with self.assertRaises(b.AuditEvidenceError):
            b.empirical_bernstein_epsilon_upper(model, bad, policy_ids, world_ids)

if __name__ == '__main__':
    unittest.main(verbosity=2)
