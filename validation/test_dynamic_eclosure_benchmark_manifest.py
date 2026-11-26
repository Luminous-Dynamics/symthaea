import copy
import unittest

from dynamic_eclosure_benchmark_manifest import (
    build_benchmark_manifest,
    derive_replicate_seed,
    validate_benchmark_manifest,
    verify_comparable_manifests,
)


HASHES = {
    "data_generator_sha256": "1" * 64,
    "filtration_contract_sha256": "2" * 64,
    "arrival_policy_sha256": "3" * 64,
    "null_spec_sha256": "4" * 64,
    "alternative_spec_sha256": "5" * 64,
    "reporting_schedule_sha256": "6" * 64,
    "capacity_utility_sha256": "7" * 64,
}


def manifest(method_id, guarantee="stopped_fdr", contract=None):
    return build_benchmark_manifest(
        method_id=method_id,
        guarantee_class=guarantee,
        alpha_numerator=1,
        alpha_denominator=20,
        source_sha256="a" * 64 if method_id == "method-a" else "b" * 64,
        generator_sha256="c" * 64,
        comparison_contract=contract or HASHES,
        scenario_id="shared-control-null-rho-0.5",
        master_seed=20261101,
        replicate_index=17,
        samples=50000,
        horizon=60,
        method_assumptions={"evidence_family": method_id},
        environment={"python": "3.13.0", "platform": "test"},
        metrics=("fdr", "power", "p95_latency"),
    )


class DynamicEclosureBenchmarkManifestTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_replicate_specific(self):
        a = derive_replicate_seed(20261101, "scenario-a", 0)
        b = derive_replicate_seed(20261101, "scenario-a", 0)
        c = derive_replicate_seed(20261101, "scenario-a", 1)
        d = derive_replicate_seed(20261101, "scenario-b", 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 1 << 64)

    def test_manifest_is_reproducible_and_validates(self):
        first = manifest("method-a")
        second = manifest("method-a")
        self.assertEqual(first, second)
        audit = validate_benchmark_manifest(first)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["manifest_sha256"], first["manifest_sha256"])
        self.assertNotIn("created_at", first)

    def test_tampering_with_alpha_or_seed_breaks_manifest_digest(self):
        original = manifest("method-a")
        tampered = copy.deepcopy(original)
        tampered["alpha"] = [1, 10]
        with self.assertRaisesRegex(ValueError, "manifest_sha256 mismatch"):
            validate_benchmark_manifest(tampered)

        tampered_seed = copy.deepcopy(original)
        tampered_seed["seed"]["derived_seed_u64"] ^= 1
        with self.assertRaisesRegex(ValueError, "derived seed"):
            validate_benchmark_manifest(tampered_seed)

    def test_methods_are_comparable_only_under_identical_contracts(self):
        first = manifest("method-a")
        second = manifest("method-b")
        result = verify_comparable_manifests((first, second))
        self.assertTrue(result["comparable"], result)
        self.assertEqual(result["methods"], ["method-a", "method-b"])

    def test_different_guarantee_classes_must_not_share_leaderboard(self):
        first = manifest("method-a", "stopped_fdr")
        second = manifest("method-b", "supfdr")
        result = verify_comparable_manifests((first, second))
        self.assertFalse(result["comparable"])
        self.assertIn("guarantee_class", [row[1] for row in result["mismatches"]])

    def test_different_filtration_or_arrival_contracts_are_rejected(self):
        contract = dict(HASHES)
        contract["filtration_contract_sha256"] = "8" * 64
        first = manifest("method-a")
        second = manifest("method-b", contract=contract)
        result = verify_comparable_manifests((first, second))
        self.assertFalse(result["comparable"])
        self.assertIn(
            "comparison_contract",
            [row[1] for row in result["mismatches"]],
        )

    def test_invalid_alpha_and_nonfinite_nested_metadata_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            build_benchmark_manifest(
                method_id="method",
                guarantee_class="stopped_fdr",
                alpha_numerator=1,
                alpha_denominator=1,
                source_sha256="a" * 64,
                generator_sha256="b" * 64,
                comparison_contract=HASHES,
                scenario_id="scenario",
                master_seed=1,
                replicate_index=0,
                samples=10,
                horizon=10,
            )
        with self.assertRaisesRegex(ValueError, "canonical-JSON"):
            build_benchmark_manifest(
                method_id="method",
                guarantee_class="stopped_fdr",
                alpha_numerator=1,
                alpha_denominator=20,
                source_sha256="a" * 64,
                generator_sha256="b" * 64,
                comparison_contract=HASHES,
                scenario_id="scenario",
                master_seed=1,
                replicate_index=0,
                samples=10,
                horizon=10,
                method_assumptions={"bad": float("nan")},
            )

    def test_duplicate_method_ids_cannot_form_a_comparison(self):
        first = manifest("same")
        second = manifest("same")
        result = verify_comparable_manifests((first, second))
        self.assertFalse(result["comparable"])
        self.assertTrue(any(row[0] == "duplicate_method_id" for row in result["mismatches"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
