#!/usr/bin/env python3
"""Static authoring validator for CEB claim-support closure materials.

This validator checks that the public protocol, six-case challenge, and key
remain structurally complete and compositionally bounded. It does not build
or execute a repository graph, inspect candidate code, score claims, or
establish evidence closure in a real project.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROTOCOL = "docs/research/CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_AND_SELECTIVE_INVALIDATION_PROTOCOL_V0_2026-11-08.md"
CHALLENGE = "docs/research/CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CHALLENGE_V0_2026-11-08.md"
KEY = "docs/research/CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_KEY_V0_2026-11-08.md"
CASE_IDS = [f"CCL-S{i:02d}" for i in range(1, 7)]
DISPOSITIONS = ("retain", "revalidate", "invalidate", "supersede", "unresolved")
RERUN_TREATMENTS = ("rerun", "regrade", "reuse")


def extract_blocks(text: str, pattern: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    ids: list[str] = []
    blocks: list[str] = []
    for index, match in enumerate(matches):
        ids.append(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end])
    return ids, blocks


def validate(protocol: str, challenge: str, key: str) -> list[str]:
    errors: list[str] = []
    challenge_ids, challenge_blocks = extract_blocks(
        challenge, r"^## (CCL-S\d{2})\b"
    )
    key_ids, key_blocks = extract_blocks(
        key, r"^## \d+\. (CCL-S\d{2})\b"
    )

    if challenge_ids != CASE_IDS:
        errors.append(f"challenge cases must be CCL-S01..CCL-S06 in order; got {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"key cases must be CCL-S01..CCL-S06 in order; got {key_ids}")

    for case_id, block in zip(challenge_ids, challenge_blocks):
        for term in ("Initial claims:", "Support closure:", "Change event", "Reviewer question:"):
            if term not in block:
                errors.append(f"{case_id}: missing {term}")

    for case_id, block in zip(key_ids, key_blocks):
        if "expected disposition" not in block.lower():
            errors.append(f"{case_id} key: missing expected disposition")
        for term in ("impact witness", "unaffectedness", "claim"):
            if term not in block.lower():
                errors.append(f"{case_id} key: missing {term}")

    required_protocol = (
        "support closure",
        "closure completeness",
        "unaffectedness witness",
        "impact witness",
        "retain",
        "revalidate",
        "invalidate",
        "supersede",
        "unresolved",
        "typed dependency",
        "verifier-only",
        "authority-root",
        "not a second evidence-continuity system",
        "#10630",
        "#10173",
        "#9112",
        "#10314",
        "#10610",
        "#10619",
        "#10064",
        "#10369",
        "no new task schema",
        "no new metric lane",
    )
    for term in required_protocol:
        if term.lower() not in protocol.lower():
            errors.append(f"protocol missing required concept: {term}")

    for term in DISPOSITIONS:
        if term not in challenge or term not in key:
            errors.append(f"challenge/key must preserve continuity disposition {term}")
    for term in RERUN_TREATMENTS:
        if term not in challenge or term not in key:
            errors.append(f"challenge/key must distinguish benchmark treatment {term}")

    if "not a sealed holdout" not in challenge.lower():
        errors.append("challenge must state that it is not a sealed holdout")
    if "not secure blinding" not in key.lower():
        errors.append("key must disclose that it is not secure blinding")
    if "not a sealed holdout" not in key.lower():
        errors.append("key must state that it is not a sealed holdout")
    if "no model performance" not in protocol.lower():
        errors.append("protocol must disclose no model-performance claim")

    return errors


def self_test() -> list[str]:
    protocol = "\n".join((
        "support closure",
        "closure completeness",
        "unaffectedness witness",
        "impact witness",
        *DISPOSITIONS,
        "typed dependency",
        "verifier-only",
        "authority-root",
        "not a second evidence-continuity system",
        "#10630 #10173 #9112 #10314 #10610 #10619 #10064 #10369",
        "no new task schema",
        "no new metric lane",
        "no model performance claim",
    ))
    challenge = "\n".join(
        "\n".join((
            f"## {case_id}",
            "Initial claims:",
            "Support closure:",
            "Change event E",
            "Reviewer question:",
            *DISPOSITIONS,
            *RERUN_TREATMENTS,
        ))
        for case_id in CASE_IDS
    ) + "\nThis is not a sealed holdout."
    key = "\n".join(
        "\n".join((
            f"## {i}. {case_id}",
            "Expected disposition:",
            "Impact witness",
            "Unaffectedness witness",
            "Claim ceiling",
            *DISPOSITIONS,
            *RERUN_TREATMENTS,
        ))
        for i, case_id in enumerate(CASE_IDS, 1)
    ) + "\nnot secure blinding; not a sealed holdout"

    baseline = validate(protocol, challenge, key)
    if baseline:
        return [f"valid minimal support-closure pack rejected: {baseline[0]}"]

    mutants = (
        ("missing case", challenge.replace("CCL-S03", "CCL-SXX", 1), key, protocol),
        ("missing event", challenge.replace("Change event E", "No event", 1), key, protocol),
        ("missing closure", challenge.replace("Support closure:", "No support:", 1), key, protocol),
        ("missing unaffectedness witness", protocol.replace("unaffectedness witness", "impact only"), challenge, key),
        ("missing regrade distinction", challenge.replace("regrade", "rerun-only"), key, protocol),
        ("missing continuity owner", protocol.replace("#10630", "#10629"), challenge, key),
        ("key not secure-blinding declaration", protocol, challenge, key.replace("not secure blinding", "secure blinded")),
    )
    for label, mutated_challenge, mutated_key, mutated_protocol in mutants:
        if not validate(mutated_protocol, mutated_challenge, mutated_key):
            return [f"negative self-test accepted mutation: {label}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        failures = self_test()
        if failures:
            for failure in failures:
                print(f"SELF_TEST_FAIL: {failure}", file=sys.stderr)
            return 1
        print("CEB_REVIEW_SUPPORT_CLOSURE_SELF_TEST_PASS: 7 negative controls")

    try:
        documents = [
            (args.repo_root / relative).read_text(encoding="utf-8")
            for relative in (PROTOCOL, CHALLENGE, KEY)
        ]
    except OSError as exc:
        print(f"ERROR: cannot read closure research pack: {exc}", file=sys.stderr)
        return 1

    failures = validate(*documents)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "CEB_REVIEW_SUPPORT_CLOSURE_PACK_PASS: "
        "cases=6 claim_support_closure=1 completeness=1 impact_witness=1 "
        "unaffectedness_witness=1 continuity_composition=1"
    )
    print(
        "CEB_REVIEW_SUPPORT_CLOSURE_CLAIM_BOUNDARY: static document validation only; "
        "no repository graph executed, no claim disposition scored, no candidate code run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
