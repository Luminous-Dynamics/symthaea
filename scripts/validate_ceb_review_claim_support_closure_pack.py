#!/usr/bin/env python3
"""Static authoring validator for CEB claim-support closure development materials.

This validator checks public Markdown structure and required claim boundaries.
It does not construct a repository graph, inspect source code, calculate impact,
execute a candidate, or assert that any real claim was retained or invalidated.
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


def extract_blocks(text: str, pattern: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    ids: list[str] = []
    bodies: list[str] = []
    for i, match in enumerate(matches):
        ids.append(match.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        bodies.append(text[match.start():end])
    return ids, bodies


def validate(protocol: str, challenge: str, key: str) -> list[str]:
    errors: list[str] = []
    challenge_ids, challenge_blocks = extract_blocks(challenge, r"^## (CCL-S\d{2})\b")
    key_ids, key_blocks = extract_blocks(key, r"^## \d+\. (CCL-S\d{2})\b")

    if challenge_ids != CASE_IDS:
        errors.append(f"challenge must contain CCL-S01..CCL-S06 once and in order; got {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"key must contain CCL-S01..CCL-S06 once and in order; got {key_ids}")

    for case_id, block in zip(challenge_ids, challenge_blocks):
        for required in ("Initial claims:", "Support closure:", "Change event", "Reviewer question:"):
            if required not in block:
                errors.append(f"{case_id}: missing {required}")
        if "C1" not in block or "C2" not in block:
            errors.append(f"{case_id}: expected at least two separately scoped claims")

    for case_id, block in zip(key_ids, key_blocks):
        for required in ("expected disposition", "Key lesson"):
            if required.lower() not in block.lower():
                errors.append(f"{case_id} key: missing {required}")

    required_protocol = (
        "support closure",
        "Closure completeness is itself a claim",
        "impact witness",
        "unaffectedness witness",
        "shared upstream dependency",
        "historical truth",
        "current decision-use",
        "#10630",
        "#10173",
        "#9112",
        "#10610",
        "#10619",
        "#10314",
        "#10064",
        "#10369",
        "no second repository graph",
        "not a new metric lane",
        "not a sealed holdout",
    )
    for required in required_protocol:
        if required.lower() not in protocol.lower():
            errors.append(f"protocol missing required concept: {required}")

    for disposition in DISPOSITIONS:
        if disposition not in protocol:
            errors.append(f"protocol missing continuity disposition: {disposition}")
    for edge in EDGE_KINDS:
        if edge not in protocol:
            errors.append(f"protocol missing typed edge: {edge}")

    if "not secure blinding" not in key.lower():
        errors.append("public answer key must disclose that it is not secure blinding")
    if "not a sealed holdout" not in challenge.lower():
        errors.append("challenge must disclose that it is not a sealed holdout")
    if "no candidate code" not in protocol.lower():
        errors.append("protocol must preserve source-only/no-candidate-execution boundary")

    # Guard the critical case distinctions so later edits cannot collapse them.
    required_key_phrases = (
        "implementation behavior",
        "documented option contract",
        "shared upstream dependency",
        "build selection and source structural possibility",
        "verifier-only update",
        "historical signed approval bytes",
        "current authorized/admissible",
        "incomplete closure",
    )
    for phrase in required_key_phrases:
        if phrase.lower() not in key.lower():
            errors.append(f"key missing distinction: {phrase}")

    return errors


def self_test() -> list[str]:
    protocol = "\n".join((
        "support closure",
        "Closure completeness is itself a claim",
        "impact witness",
        "unaffectedness witness",
        "shared upstream dependency",
        "historical truth",
        "current decision-use",
        "#10630 #10173 #9112 #10610 #10619 #10314 #10064 #10369",
        "no second repository graph not a new metric lane not a sealed holdout",
        *DISPOSITIONS,
        *EDGE_KINDS,
        "no candidate code",
    ))
    challenge = "\n".join(
        "\n".join((
            f"## {case_id}",
            "Initial claims:",
            "C1",
            "C2",
            "Support closure:",
            "Change event",
            "Reviewer question:",
        ))
        for case_id in CASE_IDS
    ) + "\nnot a sealed holdout"
    key = "\n".join(
        "\n".join((
            f"## {i}. {case_id}",
            "expected disposition",
            "Key lesson",
        ))
        for i, case_id in enumerate(CASE_IDS, 1)
    ) + (
        "\nnot secure blinding "
        "implementation behavior documented option contract shared upstream dependency "
        "build selection and source structural possibility verifier-only update "
        "historical signed approval bytes current authorized/admissible incomplete closure"
    )

    if validate(protocol, challenge, key):
        return ["valid minimal claim-support pack rejected"]

    mutations = (
        ("case removed", challenge.replace("CCL-S03", "CCL-SXX", 1), key, protocol),
        ("support closure removed", challenge.replace("Support closure:", "Closure omitted:", 1), key, protocol),
        ("answer-key distinction removed", challenge, key.replace("historical signed approval bytes", "signature text"), protocol),
        ("unaffectedness witness removed", challenge, key, protocol.replace("unaffectedness witness", "no unaffectedness proof")),
        ("disposition removed", challenge, key, protocol.replace("supersede", "missing")),
        ("execution boundary removed", challenge, key, protocol.replace("no candidate code", "candidate code run")),
    )
    for label, mutated_challenge, mutated_key, mutated_protocol in mutations:
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
        print("CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_SELF_TEST_PASS: 6 negative controls")

    documents: list[str] = []
    for relative in (PROTOCOL, CHALLENGE, KEY):
        try:
            documents.append((args.repo_root / relative).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"ERROR: cannot read {relative}: {exc}", file=sys.stderr)
            return 1

    failures = validate(*documents)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_PACK_PASS: "
        "cases=6 dispositions=5 typed_edges=8 challenge=public-key=separate-public"
    )
    print(
        "CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CLAIM_BOUNDARY: static documentation validation only; "
        "no repository graph, candidate code, tests, builds, graders or claim invalidation executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
