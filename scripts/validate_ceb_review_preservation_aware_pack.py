#!/usr/bin/env python3
"""Static authoring validator for CEB preservation-aware repair verification.

Reads only the three public Markdown documents. It never executes candidate
code, compiles, runs tests, constructs hidden labels, or verifies a repair.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROTOCOL = "docs/research/CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_PROTOCOL_V0_2026-11-08.md"
CHALLENGE = "docs/research/CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_CHALLENGE_V0_2026-11-08.md"
KEY = "docs/research/CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_KEY_V0_2026-11-08.md"
CASE_IDS = [f"PAR-S{i:02d}" for i in range(1, 7)]
PRIMARY_METRICS = (
    "defect_recall_at_5",
    "actionable_precision",
    "reviewer_burden",
)


def blocks(text: str, pattern: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    ids: list[str] = []
    result: list[str] = []
    for index, match in enumerate(matches):
        ids.append(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append(text[match.start():end])
    return ids, result


def validate(protocol: str, challenge: str, key: str) -> list[str]:
    errors: list[str] = []
    challenge_ids, challenge_blocks = blocks(challenge, r"^## (PAR-S\d{2})\b")
    key_ids, key_blocks = blocks(key, r"^## \d+\. (PAR-S\d{2})\b")

    if challenge_ids != CASE_IDS:
        errors.append(f"challenge must contain PAR-S01..PAR-S06 once and in order; got {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"answer key must contain PAR-S01..PAR-S06 once and in order; got {key_ids}")

    for case_id, block in zip(challenge_ids, challenge_blocks):
        for required in (
            "Normative contract:",
            "Canonical implementation:",
            "Mutant implementation:",
            "Preservation boundary:",
            "Reviewer question:",
        ):
            if required not in block:
                errors.append(f"{case_id}: missing {required}")
        if "~~~rust" not in block:
            errors.append(f"{case_id}: missing Rust source example")

    for case_id, block in zip(key_ids, key_blocks):
        for required in (
            "Canonical disposition:",
            "Mutant disposition:",
            "Target witness:",
            "Preservation witness:",
            "Minimal discriminating oracle:",
            "Claim ceiling:",
        ):
            if required not in block:
                errors.append(f"{case_id} key: missing {required}")

    for required in (
        "Target claim",
        "Preservation claim",
        "Independent oracle",
        "Four-way original/patch outcome matrix",
        "Over-correction",
        "P0",
        "P1",
        "P2",
        "specification-first",
        "No self-grading",
        "#10064",
        "#10369",
        "no new primary metric",
        "permanently ineligible as a sealed holdout",
    ):
        if required.lower() not in protocol.lower():
            errors.append(f"protocol missing required concept: {required}")

    for metric in PRIMARY_METRICS:
        if metric not in protocol:
            errors.append(f"protocol missing unchanged primary metric: {metric}")

    if "No candidate code" not in protocol and "no candidate code" not in protocol.lower():
        errors.append("protocol must preserve the no-candidate-execution boundary")

    if "not hidden ground truth" not in key.lower():
        errors.append("answer key must disclose that it is not hidden ground truth")
    if "not a sealed holdout" not in challenge.lower():
        errors.append("challenge must disclose that it is not a sealed holdout")

    return errors


def self_test() -> list[str]:
    protocol = "\n".join((
        "Target claim",
        "Preservation claim",
        "Independent oracle",
        "Four-way original/patch outcome matrix",
        "Over-correction P0 P1 P2 specification-first No self-grading",
        "#10064 #10369 no new primary metric permanently ineligible as a sealed holdout",
        "No candidate code",
        *PRIMARY_METRICS,
    ))
    challenge = "\n".join(
        "\n".join((
            f"## {case_id}",
            "Normative contract:",
            "Canonical implementation:",
            "Mutant implementation:",
            "~~~rust",
            "Preservation boundary:",
            "Reviewer question:",
        ))
        for case_id in CASE_IDS
    ) + "\nnot a sealed holdout"
    key = "\n".join(
        "\n".join((
            f"## {i}. {case_id}",
            "Canonical disposition:",
            "Mutant disposition:",
            "Target witness:",
            "Preservation witness:",
            "Minimal discriminating oracle:",
            "Claim ceiling:",
        ))
        for i, case_id in enumerate(CASE_IDS, 1)
    ) + "\nnot hidden ground truth"

    if validate(protocol, challenge, key):
        return ["valid minimal preservation pack rejected"]

    mutants = (
        ("missing case", challenge.replace("PAR-S03", "PAR-SXX", 1), key, protocol),
        ("missing preservation boundary", challenge.replace("Preservation boundary:", "Boundary omitted:", 1), key, protocol),
        ("missing mutant disposition", challenge, key.replace("Mutant disposition:", "Mutant omitted:", 1), protocol),
        ("missing independent oracle", challenge, key, protocol.replace("Independent oracle", "oracle absent")),
        ("missing P2 arm", challenge, key, protocol.replace("P0 P1 P2", "P0 P1")),
        ("removed execution gate", challenge, key, protocol.replace("#10064", "gate absent")),
    )
    for label, mutated_challenge, mutated_key, mutated_protocol in mutants:
        if not validate(mutated_protocol, mutated_challenge, mutated_key):
            return [f"negative self-test accepted mutation: {label}"]
    return []


def read_required(root: Path, relative: str) -> str:
    try:
        return (root / relative).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {relative}: {exc}") from exc


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
        print("CEB_REVIEW_PRESERVATION_AWARE_SELF_TEST_PASS: 6 negative controls")

    try:
        documents = [
            read_required(args.repo_root, path)
            for path in (PROTOCOL, CHALLENGE, KEY)
        ]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures = validate(*documents)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "CEB_REVIEW_PRESERVATION_AWARE_PACK_PASS: "
        "cases=6 canonical_mutant_pairs=6 dual_witness=1 independent_oracle=1 "
        "answer_key=separate-public primary_metrics=3"
    )
    print(
        "CEB_REVIEW_PRESERVATION_AWARE_CLAIM_BOUNDARY: documentation-only static validation; "
        "no candidate code, builds, tests, PoCs, hidden labels, graders or repair outcomes executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
