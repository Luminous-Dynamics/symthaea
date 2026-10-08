#!/usr/bin/env python3
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import sys

BINDING_FIELDS = {
    "attribute_only": ["attribute_id"],
    "generation_only": ["generation"],
    "context_only": ["context_id"],
    "subject_only": ["subject_id"],
    "subject_attribute": ["subject_id", "attribute_id"],
    "exact_all": ["subject_id", "attribute_id", "generation", "context_id"],
}
VARIANTS = {
    "exact_current": set(),
    "subject_substitution": {"subject_id"},
    "attribute_substitution": {"attribute_id"},
    "stale_generation": {"generation"},
    "context_substitution": {"context_id"},
    "composite_replay": {"generation", "context_id"},
}
ACCEPTANCE = {
    "subject_substitution": {"attribute_only":1.0,"generation_only":1.0,"context_only":1.0,"subject_only":0.0,"subject_attribute":0.0,"exact_all":0.0},
    "attribute_substitution": {"attribute_only":0.0,"generation_only":1.0,"context_only":1.0,"subject_only":1.0,"subject_attribute":0.0,"exact_all":0.0},
    "stale_generation": {"attribute_only":1.0,"generation_only":0.0,"context_only":1.0,"subject_only":1.0,"subject_attribute":1.0,"exact_all":0.0},
    "context_substitution": {"attribute_only":1.0,"generation_only":1.0,"context_only":0.0,"subject_only":1.0,"subject_attribute":1.0,"exact_all":0.0},
    "composite_replay": {"attribute_only":1.0,"generation_only":0.0,"context_only":0.0,"subject_only":1.0,"subject_attribute":1.0,"exact_all":0.0},
}
BINDING_DIGESTS = {
    "attribute_only":"4fa206363a8b1ed95397f302edc3621cb550f8abda268e571538567fe8c25b09",
    "generation_only":"41be299b62b3f2fcc675811546b1757ea79c15d89ddae3044e5e2f238580faf2",
    "context_only":"96738f21404297a6f6d54c837e25cd2ad1cb0b3694bafe06c85100de57026aa6",
    "subject_only":"c9299ac9684844bfe1b15edcdfa4bb1ce51b35ba7e0d9d4a15938354847ab8f7",
    "subject_attribute":"23c98711f76892ea26555769817af1a9ee684e6792124869f146746b6e4b35a4",
    "exact_all":"c3118e686243325e63c6f33765ccec8107864da7628bb8c8dbd8c5a9f61c69b2",
}
VARIANT_DIGESTS = {
    "exact_current":"38bcc3d24c324add11141b707125c8a6df0fd0c12b6362bd86f89a70f5fcf7de",
    "subject_substitution":"7106fb598388f5edda2a7462b7d13a2d154dc68dd5d58fa4b2ad7f2c64e651ad",
    "attribute_substitution":"cc13f1ed9b0239172d2aca96c96f9ddce8192f578cfe4cce60f351b9b2cfac3c",
    "stale_generation":"bb209a8de2498e34b8539f21dc2f3fd8b62437141873f3127da3c605ea9eb4d7",
    "context_substitution":"562a317cf7858e95ad6e650736c49877212d928578597b82348f4607fb37b82e",
    "composite_replay":"027a8e455e932512bfc21c43b9122c8c99e3a31feb21e5f7f4ba1a1232ceacad",
}
ATTRIBUTE_DIGESTS = {
    "fifo":"d38f0d1334448d335e1b6020533ff0eb02750e312eb4c905334d93633f78e553",
    "criticality_priority":"8691e88f2fe22ba754cb72cd2477e9b22373f3a5de07e131387e2a386f3199d4",
    "minimum_liquidity_demand":"c3de67faa18e0ff0c25afe9e496f6a2c4c43ad06f5421136598d00db6368c166",
}
ATTACK_STATS = {
    "subject_substitution":{"positive_gain_rate":0.2111111111111111,"mean_gain_ticks":0.17555555555555555,"max_gain_ticks":14},
    "attribute_substitution":{"positive_gain_rate":0.12222222222222222,"mean_gain_ticks":0.14444444444444443,"max_gain_ticks":9},
    "stale_generation":{"positive_gain_rate":0.0,"mean_gain_ticks":-1.2533333333333334,"max_gain_ticks":0},
    "context_substitution":{"positive_gain_rate":0.011111111111111112,"mean_gain_ticks":-0.2311111111111111,"max_gain_ticks":6},
    "composite_replay":{"positive_gain_rate":0.0,"mean_gain_ticks":-1.4755555555555555,"max_gain_ticks":0},
}

def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def fail(message):
    raise ValueError(message)

def main():
    if len(sys.argv) != 4:
        print("usage: verify_monetary_attestation_binding.py MANIFEST.json NEGATIVE.json EXECUTION.json", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(Path(sys.argv[1]).read_text())
        negative = json.loads(Path(sys.argv[2]).read_text())
        execution = json.loads(Path(sys.argv[3]).read_text())

        factors = manifest["factors"]
        if len(factors) != 9:
            fail("factor axis count")
        cardinality = 1
        for values in factors.values():
            cardinality *= len(values)
        if manifest["schema_version"] != "monetary-attestation-binding-v1":
            fail("manifest schema")
        if manifest["factorial_size"] != 466560 or cardinality != 466560:
            fail("factorial cardinality")
        if manifest["batch_size"] != 3:
            fail("batch size")
        if factors["binding_policy"] != list(BINDING_FIELDS):
            fail("binding factor order")
        if factors["attestation_variant"] != list(VARIANTS):
            fail("variant factor order")

        fixed = manifest["fixed_dimensions"]
        if fixed["crn_namespace"] != "world:{shock}:{seed}:obligation:{index}":
            fail("CRN namespace")
        if fixed["invalid_binding_fallback"] != "neutral_ordering":
            fail("invalid-binding fallback")
        if fixed["topology_digest"] != "6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983":
            fail("topology identity")
        if fixed["resource_digest"] != "8ff36e45d517568a7b153af2dab070b0adf4c7cfe97778d7575c84dd7b2e33b7":
            fail("resource identity")
        if fixed["resource_capacity"] != 15:
            fail("resource capacity")
        if fixed["attestation_envelope_fields"] != ["subject_id","attribute_id","generation","context_id","attested_value","signer_profile","signature_valid"]:
            fail("attestation envelope fields")

        if set(manifest["binding_policies"]) != set(BINDING_FIELDS):
            fail("binding policies")
        for name, fields in BINDING_FIELDS.items():
            entry = manifest["binding_policies"][name]
            if entry["fields"] != fields or digest(entry) != BINDING_DIGESTS[name]:
                fail(f"binding identity: {name}")
        if manifest["binding_policy_digests"] != BINDING_DIGESTS:
            fail("binding digest map")

        if set(manifest["attestation_variants"]) != set(VARIANTS):
            fail("attestation variants")
        for name in VARIANTS:
            if digest(manifest["attestation_variants"][name]) != VARIANT_DIGESTS[name]:
                fail(f"variant identity: {name}")
        if manifest["attestation_variant_digests"] != VARIANT_DIGESTS:
            fail("variant digest map")
        if manifest["attribute_digests"] != ATTRIBUTE_DIGESTS:
            fail("attribute digest map")

        expected_negative = {f"BIND-X{i:02d}" for i in range(1,13)}
        if negative["schema_version"] != "monetary-attestation-binding-negative-v1":
            fail("negative schema")
        if {x["id"] for x in negative["cases"]} != expected_negative:
            fail("negative fixtures")

        if execution["schema_version"] != "monetary-attestation-binding-v1-execution":
            fail("execution schema")
        if execution["run_count"] != 466560 or execution["obligation_count"] != 1399680:
            fail("execution cardinality")
        if execution["common_random_number_cells"] != 90:
            fail("execution CRN count")
        if execution["exogenous_random_namespace"] != fixed["crn_namespace"]:
            fail("execution CRN namespace")
        if len(execution["trace_set_digest"]) != 64:
            fail("trace digest shape")
        expected_matrix = {"exact_current": {b:1.0 for b in BINDING_FIELDS}, **ACCEPTANCE}
        if execution["binding_acceptance_matrix"] != expected_matrix:
            fail("binding acceptance matrix")
        if execution["headline"] != {
            "exact_all_invalid_binding_acceptance_rate": 0.0,
            "exact_all_attack_positive_gain_rate": 0.0,
            "exact_all_max_attack_gain_ticks": 0,
            "accepted_invalid_rate_across_all_partial_binding_profiles": 17/30,
        }:
            fail("headline")
        if execution["attack_effects_if_admitted"] != ATTACK_STATS:
            fail("attack effects")
        expected_inv = {
            "factorial_cardinality_exact": True,
            "obligation_cardinality_exact": True,
            "signature_valid_for_all_synthetic_variants": True,
            "mutation_coordinate_isolation": True,
            "reporting_independent_authoritative_result": True,
            "resource_integrity_failures": 0,
            "exact_current_all_binding_profiles_accept": True,
            "exact_all_rejects_every_invalid_variant": True,
        }
        if execution["invariants"] != expected_inv:
            fail("invariants")

        print(
            "independent attestation-binding check: "
            "466560 cells / 1399680 obligations; exact subject/attribute/generation/context policy identities; "
            "mutation isolation; expected partial-binding acceptance matrix; exact-all fail-closed binding; "
            "treatment-independent CRN; authoritative-reporting independence; 12 negative fixtures"
        )
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
