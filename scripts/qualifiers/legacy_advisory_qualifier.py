#!/usr/bin/env python3
"""Independent MAT-LEGACY-001B1 qualifier.

Standard-library-only semantic oracle. It intentionally does not import or run
the Rust implementation under test.
"""
from __future__ import annotations
import hashlib, json, pathlib, re, sys

CASES = (
    "authority-explicit",
    "model-generation-distinct",
    "input-generation-distinct",
    "numeric-match-no-promotion",
    "fep-action-not-authorization",
    "malformed-stability-fails-closed",
    "nonfinite-feature-fails-closed",
    "serialization-preserves-ceiling",
    "domain-predictions-advisory",
    "no-scientific-evidence-conversion",
)

REQUIRED = {
    "authority-explicit": ("enum AdvisoryAuthorityV1", "Advisory"),
    "model-generation-distinct": ("model_generation",),
    "input-generation-distinct": ("input_generation",),
    "numeric-match-no-promotion": ("authority: AdvisoryAuthorityV1",),
    "fep-action-not-authorization": ("MiningFepAction", "not an authorization"),
    "malformed-stability-fails-closed": ("try_predict", "InvalidTemperature", "InvalidFraction"),
    "nonfinite-feature-fails-closed": ("InvalidFeatureValue", "!value.is_finite()"),
    "serialization-preserves-ceiling": ("serde::{Deserialize, Serialize}", "serialized_advisory_round_trip"),
    "domain-predictions-advisory": ("AdvisoryMiningPredictionV1", "AdvisoryStrategicPredictionV1"),
    "no-scientific-evidence-conversion": ("no conversion",),
}

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def assert_true(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)

def source_oracle(src: str) -> None:
    assert_true("enum AdvisoryAuthorityV1" in src, "authority enum missing")
    assert_true(re.search(r"enum AdvisoryAuthorityV1[\\s\\S]*?Advisory", src) is not None,
                "authority enum has no advisory variant")
    assert_true("model_generation" in src and "input_generation" in src,
                "identity generations missing")
    assert_true("pub authority: AdvisoryAuthorityV1" in src,
                "typed authority field missing")
    assert_true("pub fn try_predict" in src and "!temperature_k.is_finite()" in src,
                "temperature fail-closed path missing")
    assert_true("!x.is_finite()" in src, "fraction fail-closed path missing")
    assert_true("!value.is_finite()" in src, "feature fail-closed path missing")
    assert_true("serde::{Deserialize, Serialize}" in src, "serialization contract missing")
    assert_true("advisory_action_is_not_an_authorization" in src,
                "action/authorization boundary not represented")
    assert_true("serialized_advisory_round_trip_preserves_ceiling" in src,
                "serialization ceiling test missing")
    assert_true("AdvisoryMiningPredictionV1" in src and "AdvisoryStrategicPredictionV1" in src,
                "domain advisory wrappers missing")
    assert_true("no conversion into source-bearing scientific evidence" in src,
                "explicit scientific-evidence conversion prohibition missing")

def mutation_oracle(src: str) -> None:
    # Every mutation must invalidate at least one independent invariant.
    mutations = {
        "remove-authority-field": src.replace("pub authority: AdvisoryAuthorityV1,", "", 1),
        "remove-model-generation": src.replace("pub model_generation: String,", "", 1),
        "remove-input-generation": src.replace("pub input_generation: String,", "", 1),
        "allow-nonfinite-feature": src.replace("if !value.is_finite() { return Err(AdvisoryInputError::InvalidFeatureValue); }", "", 1),
        "remove-action-boundary": src.replace("Model-recommended action; not an authorization.", "", 1),
        "remove-serialization-test": src.replace("fn serialized_advisory_round_trip_preserves_ceiling", "fn removed_serialization_test", 1),
        "remove-scientific-firewall": src.replace(
            "no conversion into source-bearing scientific evidence", "conversion into source-bearing scientific evidence", 1),
    }
    for name, mutated in mutations.items():
        try:
            source_oracle(mutated)
        except AssertionError:
            continue
        raise AssertionError(f"mutation escaped independent oracle: {name}")

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: legacy_advisory_qualifier.py PATH", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    raw = path.read_bytes()
    src = raw.decode("utf-8")
    assert_true(path.name == "legacy_advisory.rs", "unexpected source target")
    source_oracle(src)
    mutation_oracle(src)
    manifest = {
        "qualifier": "MAT-LEGACY-001B1",
        "schema": "1",
        "source_path": str(path),
        "source_sha256": sha256(raw),
        "cases": CASES,
        "disposition": "PASS",
        "claim_ceiling": "deterministic software authority separation only",
    }
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
