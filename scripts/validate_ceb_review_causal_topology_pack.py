#!/usr/bin/env python3
"""Static authoring validator for the public CEB causal-topology research pack.

This validator reads Markdown only. It never loads task fixtures, executes
candidate code, invokes Cargo/tests/graders, or creates benchmark results.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROTOCOL = "docs/research/CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md"
CHALLENGE = "docs/research/CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_V0_2026-11-07.md"
KEY = "docs/research/CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_KEY_V0_2026-11-07.md"
PREREGISTRATION = "docs/research/CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md"
ANNOTATOR_PACK = "docs/research/CEB_REVIEW_DUAL_ANNOTATOR_CALIBRATION_AND_RELIABILITY_PACK_V0_2026-11-07.md"
TRIANGULATION = "docs/research/CEB_REVIEW_BENCHMARK_TRIANGULATION_V0_2026-11-07.md"

CASE_IDS = [f"CTG-S{i:02d}" for i in range(1, 7)]
STATES = ("P", "I", "F", "N", "M", "IRR")
PRIMARY_METRICS = (
    "defect_recall_at_5",
    "actionable_precision",
    "reviewer_burden",
)


def case_blocks(text: str, pattern: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    ids: list[str] = []
    blocks: list[str] = []
    for index, match in enumerate(matches):
        ids.append(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end])
    return ids, blocks


def validate(
    protocol: str,
    challenge: str,
    key: str,
    preregistration: str,
    annotator_pack: str,
    triangulation: str,
) -> list[str]:
    errors: list[str] = []

    challenge_ids, challenge_blocks = case_blocks(
        challenge, r"^## (CTG-S\d{2})\b"
    )
    key_ids, key_blocks = case_blocks(
        key, r"^## \d+\. (CTG-S\d{2})\b"
    )

    if challenge_ids != CASE_IDS:
        errors.append(f"challenge cases must be CTG-S01..S06 in order; got {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"answer-key cases must be CTG-S01..S06 in order; got {key_ids}")

    for case_id, block in zip(challenge_ids, challenge_blocks):
        for state in STATES:
            if not re.search(rf"^\| {state} \|", block, re.MULTILINE):
                errors.append(f"{case_id}: missing {state} source-state row")
        if "Frozen invariant" not in block or "Evidence scope" not in block:
            errors.append(f"{case_id}: missing invariant or evidence-scope declaration")
        if "Reviewer question" not in block:
            errors.append(f"{case_id}: missing reviewer question")

    for case_id, block in zip(key_ids, key_blocks):
        for state in STATES:
            if not re.search(rf"^\| {state} \|", block, re.MULTILINE):
                errors.append(f"{case_id} key: missing {state} expected-state row")
        for required in ("Expected target mechanism", "Canonical topology", "Nearest wrong mechanisms"):
            if required not in block:
                errors.append(f"{case_id} key: missing {required}")

    for required in (
        "typed, directed multigraph",
        "AND gate",
        "OR gate",
        "happens_before",
        "Minimal sufficient paths",
        "Residual reachability",
        "Q1–Q6",
        "#10064",
        "#10369",
        "not a new Rust API",
        "new metric id",
    ):
        if required.lower() not in protocol.lower():
            errors.append(f"protocol: missing required concept {required}")

    for metric in PRIMARY_METRICS:
        if metric not in protocol or metric not in preregistration:
            errors.append(f"existing primary measure {metric} must remain unchanged")

    for required in (
        "CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md",
        "CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_V0_2026-11-07.md",
        "36 P/I/F/N/M/IRR variants",
    ):
        if required not in preregistration:
            errors.append(f"preregistration: missing topology composition {required}")

    if "6.1 Causal-topology addendum" not in annotator_pack:
        errors.append("dual-annotator pack is missing the causal-topology addendum")
    for required in (
        "AND gates",
        "OR gates",
        "Temporal partial order",
        "Residual target paths",
        "first-pass graphs",
    ):
        if required not in annotator_pack:
            errors.append(f"dual-annotator pack: missing topology annotation field {required}")

    for required in (
        "Second live capture",
        "20:14 SAST",
        "82.2 F1",
        "SDRCA",
        "UniVal",
        "AACR-Bench",
        "285%",
        "83.0",
        "no new metric lane",
    ):
        if required.lower() not in triangulation.lower():
            errors.append(f"triangulation: missing current benchmark/methodology note {required}")

    if "no candidate code" not in protocol.lower():
        errors.append("protocol must preserve the no-candidate-execution boundary")
    if "not secure" not in key.lower():
        errors.append("public answer key must disclose that it is not secure blinding")

    return errors


def self_test() -> list[str]:
    protocol = "\n".join((
        "typed, directed multigraph",
        "AND gate OR gate happens_before",
        "Minimal sufficient paths Residual reachability",
        "Q1–Q6 #10064 #10369",
        "not a new Rust API not new metric IDs",
        *PRIMARY_METRICS,
        "no candidate code",
    ))
    challenge = "\n".join(
        "\n".join(
            [f"## {case_id}", "Frozen invariant", "Evidence scope", "Reviewer question"] +
            [f"| {state} |" for state in STATES]
        )
        for case_id in CASE_IDS
    )
    key = "\n".join(
        "\n".join(
            [f"## {i}. {case_id}", "Expected target mechanism", "Canonical topology", "Nearest wrong mechanisms"] +
            [f"| {state} |" for state in STATES]
        )
        for i, case_id in enumerate(CASE_IDS, 1)
    ) + "\nnot secure blinding"
    preregistration = "\n".join((
        "CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md",
        "CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_V0_2026-11-07.md",
        "36 P/I/F/N/M/IRR variants",
        *PRIMARY_METRICS,
    ))
    annotator = "\n".join((
        "6.1 Causal-topology addendum",
        "AND gates OR gates Temporal partial order Residual target paths first-pass graphs",
    ))
    triangulation = "\n".join((
        "Second live capture 20:14 SAST 82.2 F1 SDRCA UniVal AACR-Bench 285% 83.0 no new metric lane",
    ))

    baseline = validate(
        protocol, challenge, key, preregistration, annotator, triangulation
    )
    if baseline:
        return [f"valid minimal pack rejected: {baseline[0]}"]

    mutations = (
        ("missing challenge case", challenge.replace("CTG-S03", "CTG-SXX", 1), key),
        ("missing I state", challenge.replace("| I |", "| X |", 1), key),
        ("missing residual-path key", protocol.replace("Residual reachability", "residual omitted"), key),
        ("missing annotator addendum", annotator.replace("6.1 Causal-topology addendum", "6.1 removed"), key),
        ("stale leaderboard capture", triangulation.replace("82.2 F1", "stale score"), key),
    )
    for label, mutated_challenge, mutated_key in mutations:
        result = validate(
            protocol, mutated_challenge, mutated_key,
            preregistration, annotator, triangulation
        )
        if not result:
            return [f"negative self-test accepted mutation: {label}"]

    return []


def read_required(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
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
        print("CEB_REVIEW_CAUSAL_TOPOLOGY_SELF_TEST_PASS: 5 negative controls")

    try:
        documents = [
            read_required(args.repo_root, relative)
            for relative in (
                PROTOCOL, CHALLENGE, KEY, PREREGISTRATION, ANNOTATOR_PACK, TRIANGULATION
            )
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
        "CEB_REVIEW_CAUSAL_TOPOLOGY_PACK_PASS: "
        "cases=6 states=6 variants=36 graph_protocol=1 answer_key=separate-public "
        "annotator_addendum=1 comparator_refresh=1 primary_metrics=3"
    )
    print(
        "CEB_REVIEW_CAUSAL_TOPOLOGY_CLAIM_BOUNDARY: documentation-only static validation; "
        "no candidate code, Cargo command, tests, PoCs, graders or benchmark tasks executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
