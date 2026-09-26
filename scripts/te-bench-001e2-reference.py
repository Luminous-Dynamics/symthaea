#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

CORPUS = Path("docs/release/evidence/te-bench-001e1-synthetic-leakage-graph-corpus-v1.json")
EXPECTED_SHA256 = "5a2ae3b12db5c1ad7208fbb761c3ac7aa0098fad6140eaf29907fdef4f2f19b8"
EXPECTED_GIT_BLOB = "ebdb0e14f2864b9bf1f3a45911e02a495ebdf884"
EXPECTED_SCHEMA = "te-bench-001e1-synthetic-leakage-graph-corpus-v1"
EXPECTED_ARCH = "0747b9b850f0e573e4c9fff7e34b2885d2b4571f"
EXPECTED_AUTHORITY = "representation_only_no_scientific_observation_authority"
EXPECTED_IDS = [f"TE-G-{i:02d}" for i in range(1, 28)]
EXPECTED_VECTORS = [
    "collapse_alias",
    "same_canonical_subject",
    "same_parent_family",
    "same_specimen_series",
    "shared_computational_ancestry",
    "independent_evidence_origin",
    "strong_material_holdout_allowed",
    "family_holdout_allowed",
    "controlled_exposure_supported",
    "contradiction_preserved",
    "new_graph_generation_required",
]


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def derive(f):
    out = set()
    alias = f.get("alias")

    if alias == "established" or f.get("mirrored_source") or f.get("secondary_reprint"):
        out.add("collapse_alias")
    if f.get("same_subject") or alias == "established":
        out.add("same_canonical_subject")
    if f.get("same_parent"):
        out.add("same_parent_family")
    if f.get("same_specimen"):
        out.add("same_specimen_series")
    if f.get("same_electronic_structure") or f.get("same_phonon_artifact"):
        out.add("shared_computational_ancestry")

    blockers = (
        alias in {"established", "conflicting"}
        or bool(f.get("mirrored_source"))
        or bool(f.get("secondary_reprint"))
        or bool(f.get("same_source_experiment"))
        or bool(f.get("same_specimen"))
        or bool(f.get("same_electronic_structure"))
        or bool(f.get("same_phonon_artifact"))
        or bool(f.get("training_contains_source"))
        or bool(f.get("edge_policy_changed"))
        or bool(f.get("same_parent") and f.get("same_dopant_series"))
        or bool(
            f.get("same_subject")
            and f.get("same_temperature_trajectory")
            and not f.get("independent_labs")
        )
    )

    positive_independent = (
        bool(f.get("independent_labs"))
        or bool(f.get("cross_specimen_component_mix"))
        or bool(f.get("nominal_vs_analyzed_diff"))
        or bool(f.get("missing_carrier_state"))
        or bool(f.get("missing_direction"))
        or alias == "likely_unresolved"
        or bool(f.get("graph_clean"))
        or bool(
            f.get("same_formula")
            and (f.get("same_phase") is False or f.get("same_dopant_state") is False)
        )
        or f.get("exposure") in {"provider_cutoff_only", "postcutoff_possible"}
        or bool(
            f.get("same_subject") is False
            and f.get("same_parent") is False
            and f.get("same_specimen") is False
            and f.get("same_source_experiment") is False
        )
    )
    if positive_independent and not blockers:
        out.add("independent_evidence_origin")

    strong_holdout = (
        bool(
            f.get("same_subject") is False
            and f.get("same_parent") is False
            and f.get("same_specimen") is False
            and f.get("same_source_experiment") is False
        )
        or bool(f.get("same_formula") and f.get("same_phase") is False)
        or bool(
            f.get("same_formula")
            and f.get("same_phase") is True
            and f.get("same_dopant_state") is False
        )
        or bool(f.get("nominal_vs_analyzed_diff"))
    )
    if strong_holdout and not f.get("training_contains_source"):
        out.add("strong_material_holdout_allowed")

    if (
        f.get("same_subject") is False
        and f.get("same_parent") is False
        and f.get("same_specimen") is False
        and f.get("same_source_experiment") is False
        and f.get("exposure") == "ruled_out_exact"
    ):
        out.add("family_holdout_allowed")

    if f.get("exposure") == "ruled_out_exact":
        out.add("controlled_exposure_supported")

    if alias == "conflicting" or (
        f.get("numeric_conflict")
        and f.get("compatible_conditions")
        and f.get("independent_labs")
    ):
        out.add("contradiction_preserved")

    if f.get("discovered_after_scoring") or f.get("edge_policy_changed"):
        out.add("new_graph_generation_required")

    return out


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    data = CORPUS.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    blob = git_blob_sha1(data)
    if sha256 != EXPECTED_SHA256:
        fail(f"sha256 drift: {sha256}")
    if blob != EXPECTED_GIT_BLOB:
        fail(f"git blob drift: {blob}")

    doc = json.loads(data)
    if doc.get("schema") != EXPECTED_SCHEMA:
        fail("schema drift")
    if doc.get("architecture_head") != EXPECTED_ARCH:
        fail("architecture head drift")
    if doc.get("authority") != EXPECTED_AUTHORITY:
        fail("authority drift")
    if doc.get("vector_keys") != EXPECTED_VECTORS:
        fail("vector vocabulary/order drift")

    cases = doc.get("cases")
    if not isinstance(cases, list):
        fail("cases must be a list")
    ids = [c.get("id") for c in cases]
    if ids != EXPECTED_IDS or len(set(ids)) != len(ids):
        fail(f"case census/order drift: {ids}")

    vector_set = set(EXPECTED_VECTORS)
    for case in cases:
        expected = case.get("expected_true")
        facts = case.get("facts")
        if not isinstance(expected, list) or not isinstance(facts, dict):
            fail(f"{case.get('id')}: malformed facts/expected_true")
        if len(expected) != len(set(expected)):
            fail(f"{case['id']}: duplicate expected vector")
        unknown = set(expected) - vector_set
        if unknown:
            fail(f"{case['id']}: unknown expected vectors {sorted(unknown)}")
        got = derive(facts)
        want = set(expected)
        if got != want:
            fail(
                f"{case['id']}: semantic mismatch "
                f"missing={sorted(want-got)} unexpected={sorted(got-want)}"
            )

    print(f"ok cases={len(cases)} sha256={sha256} git_blob={blob}")


if __name__ == "__main__":
    main()
