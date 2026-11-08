#!/usr/bin/env python3
"""Static authoring validator for the CEB claim-support closure development pack.

This validates public design documents and challenge/key cross-references only.
It does not parse production evidence, execute candidates, infer claim truth,
or replace the #10173/#9112/#10314/#10630 authorities.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROTOCOL = "docs/research/CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_AND_SELECTIVE_INVALIDATION_PROTOCOL_V0_2026-11-08.md"
CHALLENGE = "docs/research/CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CHALLENGE_V0_2026-11-08.md"
KEY = "docs/research/CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_KEY_V0_2026-11-08.md"
CONTINUITY = "docs/research/CEB_EVIDENCE_CONTINUITY_PROTOCOL_V0_2026-11-05.md"
CASE_IDS = [f"CCL-S{i:02d}" for i in range(1, 7)]
DISPOSITIONS = ("retain", "revalidate", "invalidate", "supersede", "unresolved")
EVENT_CLASSES = (
    "Source content",
    "API/contract",
    "Build world",
    "Dependency/runtime",
    "Security/authority",
    "Evaluator/oracle",
    "Data/measurement",
    "Metadata-only",
)
CLAIM_KINDS = (
    "claim",
    "evidence",
    "assumption",
    "authority",
    "environment",
    "policy",
    "method",
)
EDGE_KINDS = (
    "supports",
    "derived_from",
    "assumes",
    "scoped_by",
    "authorized_by",
    "measured_under",
    "produced_under",
    "contradicts",
)


def section_blocks(text: str, pattern: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    ids: list[str] = []
    result: list[str] = []
    for i, match in enumerate(matches):
        ids.append(match.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append(text[match.start():end])
    return ids, result


def validate(protocol: str, challenge: str, key: str, continuity: str) -> list[str]:
    errors: list[str] = []
    challenge_ids, challenge_cases = section_blocks(
        challenge, r"^## (CCL-S\d{2})\b"
    )
    key_ids, key_cases = section_blocks(
        key, r"^## \d+\. (CCL-S\d{2})\b"
    )

    if challenge_ids != CASE_IDS:
        errors.append(f"challenge case set/order mismatch: {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"answer-key case set/order mismatch: {key_ids}")

    for case_id, block in zip(challenge_ids, challenge_cases):
        for required in ("Initial claims:", "Support closure:", "Change event", "Reviewer question:"):
            if required not in block:
                errors.append(f"{case_id}: missing {required}")
        if "C1:" not in block or "C2:" not in block:
            errors.append(f"{case_id}: must exercise at least two distinct claims")

    for case_id, block in zip(key_ids, key_cases):
        if "expected disposition" not in block.lower():
            errors.append(f"{case_id} key: missing expected disposition")
        if "key lesson" not in block.lower():
            errors.append(f"{case_id} key: missing key lesson")
        if not any(disposition in block.lower() for disposition in DISPOSITIONS):
            errors.append(f"{case_id} key: missing continuity disposition")

    for item in (
        "support closure",
        "closure completeness",
        "impact witness",
        "unaffectedness witness",
        "open-world remainder",
        "historical truth",
        "current decision-use",
        "event-to-claim impact",
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
        "not a sealed holdout",
    ):
        if item.lower() not in protocol.lower():
            errors.append(f"protocol missing required concept: {item}")

    for disposition in DISPOSITIONS:
        if disposition not in protocol.lower():
            errors.append(f"protocol missing continuity disposition: {disposition}")

    for event_class in EVENT_CLASSES:
        if event_class.lower() not in protocol.lower():
            errors.append(f"protocol missing event class: {event_class}")

    for kind in CLAIM_KINDS:
        if kind not in protocol.lower():
            errors.append(f"protocol missing claim node kind: {kind}")

    for edge in EDGE_KINDS:
        if edge not in protocol.lower():
            errors.append(f"protocol missing typed edge: {edge}")

    for item in (
        "not hidden ground truth",
        "not secure blinding",
        "not a sealed holdout",
        "no real repository invalidation outcome",
    ):
        if item.lower() not in key.lower():
            errors.append(f"answer key missing public-material boundary: {item}")

    if "not a sealed holdout" not in challenge.lower():
        errors.append("challenge must state it is not a sealed holdout")

    if "## 14.3 Claim-support closure extension" not in continuity:
        errors.append("existing continuity protocol must link the claim-support extension")
    if "CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_AND_SELECTIVE_INVALIDATION_PROTOCOL_V0_2026-11-08.md" not in continuity:
        errors.append("existing continuity protocol missing claim-support extension link")
    if "does not create a second" not in continuity.lower():
        errors.append("continuity protocol must preserve single-owner/non-duplication boundary")

    return errors


def self_test() -> list[str]:
    protocol = "\n".join((
        "support closure closure completeness impact witness unaffectedness witness open-world remainder",
        "historical truth current decision-use event-to-claim impact",
        "not a second evidence-continuity system #10630 #10173 #9112 #10314 #10610 #10619 #10064 #10369 no new task schema not a sealed holdout",
        *DISPOSITIONS,
        *EVENT_CLASSES,
        *CLAIM_KINDS,
        *EDGE_KINDS,
    ))
    challenge = "\n".join(
        "\n".join((
            f"## {case_id}",
            "Initial claims:",
            "C1: claim one",
            "C2: claim two",
            "Support closure:",
            "Change event E",
            "Reviewer question:",
        ))
        for case_id in CASE_IDS
    ) + "\nnot a sealed holdout"
    key = "\n".join(
        "\n".join((
            f"## {i}. {case_id}",
            "Expected disposition: retain / revalidate / invalidate / supersede / unresolved",
            "Key lesson: retain claim scope",
        ))
        for i, case_id in enumerate(CASE_IDS, 1)
    ) + "\nnot hidden ground truth not secure blinding not a sealed holdout not a real repository invalidation outcome"
    continuity = (
        "## 14.3 Claim-support closure extension\n"
        "CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_AND_SELECTIVE_INVALIDATION_PROTOCOL_V0_2026-11-08.md\n"
        "This does not create a second evidence system."
    )

    if validate(protocol, challenge, key, continuity):
        return ["valid minimal claim-support pack rejected"]

    mutations = (
        ("case removed", protocol, challenge.replace("CCL-S03", "CCL-SXX", 1), key, continuity),
        ("support closure removed", protocol, challenge.replace("Support closure:", "Missing:", 1), key, continuity),
        ("unaffectedness proof removed", protocol.replace("unaffectedness witness", "no retain proof"), challenge, key, continuity),
        ("public key boundary removed", protocol, challenge, key.replace("not secure blinding", "secure"), continuity),
        ("continuity link removed", protocol, challenge, key, continuity.replace("claim-support closure", "removed")),
        ("authority revocation omitted", protocol.replace("Security/authority", "Security event"), challenge, key, continuity),
    )
    for label, mutated_protocol, mutated_challenge, mutated_key, mutated_continuity in mutations:
        if not validate(mutated_protocol, mutated_challenge, mutated_key, mutated_continuity):
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
        print("CEB_CLAIM_SUPPORT_CLOSURE_SELF_TEST_PASS: 6 negative controls")

    paths = (PROTOCOL, CHALLENGE, KEY, CONTINUITY)
    try:
        documents = [(args.repo_root / path).read_text(encoding="utf-8") for path in paths]
    except OSError as exc:
        print(f"ERROR: cannot read required claim-support document: {exc}", file=sys.stderr)
        return 1

    failures = validate(*documents)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "CEB_CLAIM_SUPPORT_CLOSURE_PACK_PASS: cases=6 dispositions=5 "
        "claim_node_kinds=7 edge_kinds=8 event_classes=8 continuity_owner=10630"
    )
    print(
        "CEB_CLAIM_SUPPORT_CLOSURE_CLAIM_BOUNDARY: static authoring validation only; "
        "no evidence graph built, no claim disposition inferred, no candidate code or grader executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
