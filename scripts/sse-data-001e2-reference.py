#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

CORPUS = Path("docs/release/evidence/sse-data-001e1-synthetic-leakage-graph-corpus-v1.json")
EXPECTED_SHA256 = "336bfad67e621de1ffe543fd811c028c598c4c602b27de49d6e346a6aad7e0a0"
EXPECTED_GIT_BLOB = "deb7746b545946e853c030265239bd2360a61c82"
EXPECTED_SCHEMA = "sse-data-001e1-synthetic-leakage-graph-corpus-v1"
EXPECTED_ARCH = "8321ceaf334eca6a564fa8d4ca9e752acc635a0c"
EXPECTED_AUTHORITY = "representation_only_no_scientific_observation_authority"
EXPECTED_IDS = [f"SSE-G-{i:02d}" for i in range(1, 27)]
EXPECTED_VECTORS = [
    "collapse_alias",
    "same_canonical_subject",
    "same_family",
    "same_specimen_series",
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

    if alias == "established" or f.get("mirrored_source"):
        out.add("collapse_alias")
    if f.get("same_subject") or alias == "established":
        out.add("same_canonical_subject")
    if f.get("same_family"):
        out.add("same_family")
    if f.get("same_specimen") or f.get("same_raw_eis"):
        out.add("same_specimen_series")

    blockers = (
        alias in {"established", "conflicting"}
        or bool(f.get("mirrored_source"))
        or bool(f.get("secondary_reprint"))
        or bool(f.get("same_raw_eis"))
        or bool(f.get("training_contains_source"))
        or bool(f.get("edge_policy_changed"))
        or bool(
            f.get("same_subject")
            and f.get("same_temperature_trajectory")
            and not f.get("independent_labs")
        )
        or bool(f.get("same_specimen") and not f.get("independent_labs"))
    )

    positive_independent = (
        bool(f.get("independent_labs"))
        or alias == "likely_unresolved"
        or bool(f.get("missing_phase"))
        or bool(f.get("graph_clean"))
        or bool(
            f.get("same_formula")
            and (f.get("same_phase") is False or f.get("same_disorder") is False)
        )
        or bool(f.get("same_family"))
        or bool(f.get("origins") and len(set(f.get("origins", []))) > 1)
        or f.get("exposure") in {"provider_cutoff_only", "postcutoff_possible"}
        or bool(
            f.get("same_subject") is False
            and f.get("same_family") is False
            and f.get("same_specimen") is False
            and f.get("same_source_experiment") is False
        )
    )
    if positive_independent and not blockers:
        out.add("independent_evidence_origin")

    strong_holdout = (
        bool(
            f.get("same_subject") is False
            and f.get("same_family") is False
            and f.get("same_specimen") is False
            and f.get("same_source_experiment") is False
        )
        or bool(f.get("same_formula") and f.get("same_phase") is False)
        or bool(f.get("same_formula") and f.get("same_disorder") is False)
        or bool(f.get("same_family") and not f.get("same_dopant_series"))
    )
    if strong_holdout and not f.get("training_contains_source"):
        out.add("strong_material_holdout_allowed")

    if (
        f.get("same_subject") is False
        and f.get("same_family") is False
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
    family_positive = 0
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
        if "same_family" in got:
            family_positive += 1

    if family_positive != 3:
        fail(f"expected 3 exercised same_family cases, got {family_positive}")

    print(
        f"ok cases={len(cases)} family_positive={family_positive} "
        f"sha256={sha256} git_blob={blob}"
    )


if __name__ == "__main__":
    main()
