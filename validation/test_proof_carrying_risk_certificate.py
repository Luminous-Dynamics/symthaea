import copy
import unittest
from proof_carrying_risk_certificate import (
    SCHEMA, CertificateError, build_certificate, verify_certificate, sha256_json
)


def crc_fixture():
    manifest = {
        "schema": SCHEMA,
        "risk_target": {
            "kind": "expected_risk",
            "metric": "hierarchical_classification_error",
            "level": "1/10",
            "time_uniform": False
        },
        "method": {"name": "common_depth_envelope_crc_500_rounds", "version": "research-2026-10-31"},
        "assumptions": {
            "sampling": "finite_exchangeability",
            "model_frozen": True,
            "loss_frozen": True,
            "filtration": "none"
        },
        "scope": {
            "n_calibration": 50,
            "controller_entries": 1326,
            "augmented_state_count": 1378,
            "universe": "all_augmented_count_vectors",
            "time_uniform": False
        },
        "proof_lane": "exact_finite_exchangeability",
        "evidence": {
            "controller_table_sha256": "b987c413fb029fb06348605945498a44a95b84004db5ad35fc1f74c9263b8ecc",
            "loss_spec_sha256": "f7ceab7c677e408ef689d7fa3501756b6fa8efb7de2c3610bbd7c4cfcb0b0626",
            "state_space_sha256": "62709b1a19edb0786f6626c1f52b1c6875a2287f48245656c969c6ebc694d9c2",
            "pool_type_histogram_sha256": "490aa4e715c5d30e6677f8b4c0d5c05b32333bcc8d390c8e75ab50189e438359"
        }
    }
    receipt = {
        "status": "pass",
        "proof_lane": "exact_finite_exchangeability",
        "evidence_digests": manifest["evidence"],
        "verifier": {"name": "symthaea-exact-rational-common-depth-crc", "version": "v1"},
        "checks": {
            "arithmetic": "rational_exact",
            "augmented_state_count": 1378,
            "table_validation_passes": 1009,
            "exact_inequality_checks": 1390402,
            "worst_augmented_count": [0, 0, 51],
            "worst_total": "51/10",
            "budget": "51/10",
            "max_excess": "0/1"
        }
    }
    return manifest, receipt


class ProofCarryingCertificateTests(unittest.TestCase):
    def test_exact_crc_certificate_passes_and_is_bound(self):
        manifest, receipt = crc_fixture()
        cert = build_certificate(manifest, receipt)
        result = verify_certificate(cert)
        self.assertTrue(result["valid"])
        self.assertEqual(result["risk_kind"], "expected_risk")
        self.assertEqual(result["proof_lane"], "exact_finite_exchangeability")
        self.assertEqual(cert["manifest_sha256"], sha256_json(manifest))

    def test_manifest_tampering_fails_digest(self):
        manifest, receipt = crc_fixture()
        cert = build_certificate(manifest, receipt)
        cert["manifest"]["risk_target"]["level"] = "1/20"
        with self.assertRaises(CertificateError):
            verify_certificate(cert)

    def test_receipt_evidence_mismatch_fails(self):
        manifest, receipt = crc_fixture()
        receipt["evidence_digests"] = dict(receipt["evidence_digests"])
        receipt["evidence_digests"]["controller_table_sha256"] = "0" * 64
        with self.assertRaises(CertificateError):
            build_certificate(manifest, receipt)

    def test_positive_budget_excess_cannot_pass(self):
        manifest, receipt = crc_fixture()
        receipt["checks"]["max_excess"] = "1/1000000"
        with self.assertRaises(CertificateError):
            build_certificate(manifest, receipt)

    def test_offline_fdr_cannot_be_relabelled_supfdr(self):
        manifest, _ = crc_fixture()
        manifest["risk_target"]["kind"] = "supfdr"
        manifest["risk_target"]["time_uniform"] = True
        manifest["scope"]["online"] = True
        manifest["scope"]["horizon"] = 100
        manifest["proof_lane"] = "exact_finite_exchangeability"
        with self.assertRaises(CertificateError):
            build_certificate(manifest, {"status":"pass","proof_lane":"exact_finite_exchangeability",
                "evidence_digests":manifest["evidence"],"verifier":{"name":"bad"},
                "checks":{"arithmetic":"rational_exact","augmented_state_count":1,
                          "table_validation_passes":1,"max_excess":"0/1","worst_augmented_count":[0]}})

    def test_supfdr_requires_global_filtration_and_optional_stopping(self):
        manifest, _ = crc_fixture()
        manifest["risk_target"] = {"kind":"supfdr","metric":"sup_fdr","level":"1/10","time_uniform":True}
        manifest["proof_lane"] = "theorem_backed_eprocess"
        manifest["scope"] = {"online":True,"horizon":100}
        manifest["assumptions"] = {
            "filtration":"per_stream_filtrations",
            "predictable_arrivals":True,
            "optional_stopping_valid":True,
            "eprocess_supermartingale":True
        }
        with self.assertRaises(CertificateError):
            build_certificate(manifest, {"status":"pass","proof_lane":"theorem_backed_eprocess",
                "evidence_digests":manifest["evidence"],"verifier":{"name":"bad"},
                "checks":{"theorem_reference":"paper theorem","filtration_contract_checked":True,
                          "optional_stopping_contract_checked":True}})

    def test_simulation_only_cannot_pass_as_a_certificate(self):
        manifest, receipt = crc_fixture()
        manifest["proof_lane"] = "simulation_only"
        receipt["proof_lane"] = "simulation_only"
        with self.assertRaises(CertificateError):
            build_certificate(manifest, receipt)


    def test_receipt_budget_must_match_manifest_risk_target(self):
        manifest, receipt = crc_fixture()
        receipt["checks"]["budget"] = "52/10"
        with self.assertRaises(CertificateError):
            build_certificate(manifest, receipt)

    def test_exact_check_count_must_reconcile(self):
        manifest, receipt = crc_fixture()
        receipt["checks"]["exact_inequality_checks"] = 1
        with self.assertRaises(CertificateError):
            build_certificate(manifest, receipt)

    def test_fixed_family_fdr_certificate_is_not_supfdr(self):
        manifest, _ = crc_fixture()
        manifest["risk_target"] = {
            "kind": "fdr", "metric": "false_discovery_rate",
            "level": "1/20", "time_uniform": False
        }
        manifest["method"] = {"name": "rc_ebh", "version": "research-2026-10-31"}
        manifest["assumptions"] = {
            "marginal_evalues_valid": True,
            "dependence_scope": "arbitrary_dependence"
        }
        manifest["scope"] = {
            "fixed_family": True, "time_uniform": False, "family_size": 5
        }
        manifest["proof_lane"] = "analytical_fdr_theorem"
        receipt = {
            "status": "pass", "proof_lane": "analytical_fdr_theorem",
            "evidence_digests": manifest["evidence"],
            "verifier": {"name": "paper-theorem-contract-checker"},
            "checks": {
                "theorem_reference": "Bai-Ramdas-2026-Proposition-4.2",
                "assumption_contract_checked": True
            }
        }
        cert = build_certificate(manifest, receipt)
        self.assertEqual(verify_certificate(cert)["risk_kind"], "fdr")
        # Reusing the same offline receipt as a time-uniform SupFDR certificate is forbidden.
        cert["manifest"]["risk_target"]["kind"] = "supfdr"
        cert["manifest"]["risk_target"]["time_uniform"] = True
        cert["manifest"]["scope"]["fixed_family"] = False
        cert["manifest"]["scope"]["online"] = True
        cert["manifest"]["scope"]["horizon"] = 100
        cert["manifest"]["proof_lane"] = "theorem_backed_eprocess"
        with self.assertRaises(CertificateError):
            verify_certificate(cert)


    def test_simulation_only_can_be_recorded_as_inconclusive(self):
        manifest, receipt = crc_fixture()
        manifest["proof_lane"] = "simulation_only"
        receipt["proof_lane"] = "simulation_only"
        receipt["status"] = "inconclusive"
        cert = build_certificate(manifest, receipt)
        result = verify_certificate(cert)
        self.assertFalse(result["valid"])
        self.assertEqual(result["receipt_status"], "inconclusive")

    def test_raw_calibration_payloads_are_rejected(self):
        manifest, receipt = crc_fixture()
        manifest["scope"]["calibration_examples"] = [{"x": [1,2,3]}]
        with self.assertRaises(CertificateError):
            build_certificate(manifest, receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
