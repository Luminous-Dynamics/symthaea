import unittest
import numpy as np
from policy_risk_binding import PolicyRiskBindingError, build_manifest, verify_binding
from prior_set_coverage_double_oracle_benchmark import build_risk_tensor

class PolicyRiskBindingTests(unittest.TestCase):
    def setUp(self):
        self.policy_ids = ("policy:alpha", "policy:beta", "policy:gamma")
        self.hypothesis_ids = ("world:0", "world:1", "world:2", "world:3")
        self.matrix = np.asarray([[0.2, 0.4, 0.3, 0.1], [0.1, 0.5, 0.2, 0.4], [0.7, 0.2, 0.1, 0.6]], dtype=np.float64)
        self.manifest = build_manifest(self.policy_ids, self.hypothesis_ids, self.matrix)

    def verify(self, **overrides):
        args = {
            "trusted_manifest": self.manifest,
            "trusted_manifest_sha256": self.manifest.manifest_sha256,
            "runtime_policy_ids": self.policy_ids,
            "runtime_hypothesis_ids": self.hypothesis_ids,
            "runtime_risk_matrix": self.matrix,
            "execution_policy_ids": self.policy_ids,
        }
        args.update(overrides)
        return verify_binding(**args)

    def test_exact_binding_passes_and_is_stable_across_memory_layout(self):
        self.assertEqual(self.verify(), self.manifest.manifest_sha256)
        column_major = np.asfortranarray(self.matrix)
        self.assertEqual(build_manifest(self.policy_ids, self.hypothesis_ids, column_major), self.manifest)

    def test_row_reordering_is_rejected_even_when_shape_is_unchanged(self):
        with self.assertRaisesRegex(PolicyRiskBindingError, "risk_matrix_digest_mismatch"):
            self.verify(runtime_risk_matrix=self.matrix[[2, 0, 1], :])

    def test_reordered_policy_labels_are_rejected(self):
        with self.assertRaisesRegex(PolicyRiskBindingError, "runtime_policy_order_mismatch"):
            self.verify(runtime_policy_ids=("policy:gamma", "policy:alpha", "policy:beta"))

    def test_execution_order_must_match_manifest_order(self):
        with self.assertRaisesRegex(PolicyRiskBindingError, "execution_policy_order_mismatch"):
            self.verify(execution_policy_ids=("policy:beta", "policy:alpha", "policy:gamma"))

    def test_hypothesis_order_mismatch_is_rejected(self):
        with self.assertRaisesRegex(PolicyRiskBindingError, "runtime_hypothesis_order_mismatch"):
            self.verify(runtime_hypothesis_ids=("world:1", "world:0", "world:2", "world:3"))

    def test_trust_anchor_and_self_digest_tampering_are_rejected(self):
        with self.assertRaisesRegex(PolicyRiskBindingError, "manifest_trust_anchor_mismatch"):
            self.verify(trusted_manifest_sha256="0" * 64)
        from dataclasses import replace
        broken = replace(self.manifest, risk_matrix_sha256="0" * 64)
        with self.assertRaisesRegex(PolicyRiskBindingError, "manifest_self_digest_mismatch"):
            verify_binding(broken, self.manifest.manifest_sha256, self.policy_ids, self.hypothesis_ids, self.matrix, self.policy_ids)

    def test_actual_finite_game_risk_rows_are_bound_to_ordered_policy_ids(self):
        tensor, _, _ = build_risk_tensor()
        matrix = np.asarray(tensor["joint"]["risk_by_policy_and_world"], dtype=np.float64)
        policies = tuple(f"toy-policy:{i}" for i in range(matrix.shape[0]))
        hypotheses = tuple(f"toy-world:{i}" for i in range(matrix.shape[1]))
        manifest = build_manifest(policies, hypotheses, matrix)
        self.assertEqual(
            verify_binding(manifest, manifest.manifest_sha256, policies, hypotheses, matrix, policies),
            manifest.manifest_sha256,
        )
        permutation = [3, 1, 7, 0, 6, 2, 5, 4]
        with self.assertRaisesRegex(PolicyRiskBindingError, "risk_matrix_digest_mismatch"):
            verify_binding(manifest, manifest.manifest_sha256, policies, hypotheses, matrix[permutation, :], policies)
        with self.assertRaisesRegex(PolicyRiskBindingError, "runtime_policy_order_mismatch"):
            verify_binding(manifest, manifest.manifest_sha256, tuple(policies[i] for i in permutation), hypotheses, matrix, policies)

    def test_duplicate_ids_nonfinite_and_negative_costs_fail_closed(self):
        with self.assertRaisesRegex(PolicyRiskBindingError, "policy_ids_duplicate"):
            build_manifest(("same", "same", "other"), self.hypothesis_ids, self.matrix)
        with self.assertRaisesRegex(PolicyRiskBindingError, "risk_matrix_nonfinite"):
            build_manifest(self.policy_ids, self.hypothesis_ids, np.full((3, 4), np.nan))
        with self.assertRaisesRegex(PolicyRiskBindingError, "risk_matrix_negative_loss"):
            build_manifest(self.policy_ids, self.hypothesis_ids, -np.ones((3, 4)))

if __name__ == "__main__":
    unittest.main(verbosity=2)
