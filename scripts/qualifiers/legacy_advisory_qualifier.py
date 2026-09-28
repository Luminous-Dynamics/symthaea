#!/usr/bin/env python3
"""Independent MAT-LEGACY-001B1 qualifier.

This oracle is intentionally stdlib-only and does not import or execute the
Rust implementation under test. It derives dispositions from raw source text,
then attacks the source with deterministic mutations.
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import re
import sys

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

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def assert_true(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)

def authority_enum_body(src: str) -> str:
    m = re.search(r"enum\s+AdvisoryAuthorityV1\s*\{(?P<body>.*?)\n\}", src, re.S)
    assert_true(m is not None, "authority enum missing")
    return m.group("body")

def source_oracle(src: str) -> None:
    body = authority_enum_body(src)
    variants = re.findall(r"^\s*([A-Z][A-Za-z0-9_]*)\s*,?\s*(?://.*)?$", body, re.M)
    assert_true(variants == ["Advisory"], f"unexpected authority variants: {variants!r}")
    m = re.search(r"LEGACY_ADVISORY_QUALIFIER_CASES_V1: &[&str] = &\\[(?P<body>.*?)\\];", src, re.S)
    assert_true(m is not None, "qualifier manifest missing")
    manifest_cases = tuple(re.findall(r'"([^"]+)"', m.group("body")))
    assert_true(manifest_cases == CASES, "qualifier manifest drifted from independent corpus")
    assert_true("pub authority: AdvisoryAuthorityV1" in src, "typed authority field missing")
    assert_true("model_generation: String" in src and "input_generation: String" in src,
                "identity generations missing")
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
                "scientific-evidence conversion prohibition missing")
    assert_true("authority: AdvisoryAuthorityV1::Advisory" in src,
                "advisory constructors do not visibly force the ceiling")
    assert_true("assert_eq!(decoded.authority, AdvisoryAuthorityV1::Advisory)" in src,
                "round-trip does not explicitly preserve advisory authority")

def mutation_oracle(src: str) -> None:
    mutations = {
        "remove-authority-field": src.replace("pub authority: AdvisoryAuthorityV1,", "", 1),
        "add-evidence-authority": src.replace(
            "pub enum AdvisoryAuthorityV1 {\n    /// Candidate generation, ranking, or hypothesis support only.\n    Advisory,",
            "pub enum AdvisoryAuthorityV1 {\n    Advisory,\n    Evidence,",
            1),
        "remove-model-generation": src.replace("pub model_generation: String,", "", 1),
        "remove-input-generation": src.replace("pub input_generation: String,", "", 1),
        "allow-nonfinite-feature": src.replace(
            "if !value.is_finite() { return Err(AdvisoryInputError::InvalidFeatureValue); }", "", 1),
        "remove-action-boundary": src.replace(
            "fn advisory_action_is_not_an_authorization", "fn removed_action_boundary", 1),
        "remove-serialization-test": src.replace(
            "fn serialized_advisory_round_trip_preserves_ceiling", "fn removed_serialization_test", 1),
        "remove-scientific-firewall": src.replace(
            "no conversion into source-bearing scientific evidence",
            "conversion into source-bearing scientific evidence", 1),
        "manifest-case-mutation": src.replace('"authority-explicit",', '"authority-mutated",', 1),
        "constructor-promotes-authority": src.replace(
            "authority: AdvisoryAuthorityV1::Advisory", "authority: AdvisoryAuthorityV1::Evidence", 1),
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
        "schema": "2",
        "git_sha": os.environ.get("GITHUB_SHA", "unbound-local"),
        "source_path": str(path),
        "source_sha256": sha256(raw),
        "cases": CASES,
        "mutation_count": 9,
        "disposition": "PASS",
        "claim_ceiling": "deterministic software authority separation only",
    }
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
