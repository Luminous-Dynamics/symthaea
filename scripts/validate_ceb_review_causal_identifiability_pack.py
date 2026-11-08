#!/usr/bin/env python3
"""Static authoring validator for the CEB causal-identifiability development pack.

Markdown-only validation. Does not execute candidate code, tests, builds,
traces, interventions, graders, benchmark runners or hidden labels.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROTOCOL = "docs/research/CEB_REVIEW_CAUSAL_IDENTIFIABILITY_PROTOCOL_V0_2026-11-07.md"
CHALLENGE = "docs/research/CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md"
KEY = "docs/research/CEB_REVIEW_CAUSAL_IDENTIFIABILITY_KEY_V0_2026-11-07.md"
TOPOLOGY = "docs/research/CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md"
PREREG = "docs/research/CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md"
ANNOTATORS = "docs/research/CEB_REVIEW_DUAL_ANNOTATOR_CALIBRATION_AND_RELIABILITY_PACK_V0_2026-11-07.md"

CASE_IDS = [f"CID-S{i:02d}" for i in range(1, 7)]
DISPOSITIONS = (
    "identified",
    "partially_identified",
    "non_identifiable",
    "contradicted",
    "unresolved",
)
PRIMARY_METRICS = (
    "defect_recall_at_5",
    "actionable_precision",
    "reviewer_burden",
)


def split_cases(text: str, pattern: str) -> tuple[list[str], list[str]]:
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
    topology: str,
    prereg: str,
    annotators: str,
) -> list[str]:
    errors: list[str] = []

    challenge_ids, challenge_blocks = split_cases(
        challenge, r"^## (CID-S\d{2})\b"
    )
    key_ids, key_blocks = split_cases(
        key, r"^## \d+\. (CID-S\d{2})\b"
    )

    if challenge_ids != CASE_IDS:
        errors.append(f"challenge must contain CID-S01..S06 in order; got {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"answer key must contain CID-S01..S06 in order; got {key_ids}")

    for case_id, block in zip(challenge_ids, challenge_blocks):
        for required in ("Family:", "Claim:", "Claim kind:", "Evidence packet:", "Reviewer question:"):
            if required not in block:
                errors.append(f"{case_id}: missing {required}")
        if "Two compatible" not in block and "Two possible" not in block:
            errors.append(f"{case_id}: must expose at least two compatible alternatives")

    for case_id, block in zip(key_ids, key_blocks):
        for required in (
            "Expected disposition:",
            "Claim kind",
            "Material alternatives",
            "Disagreement witness:",
            "Minimum additional evidence:",
            "Claim ceiling:",
        ):
            if required not in block:
                errors.append(f"{case_id} key: missing {required}")

    for disposition in DISPOSITIONS:
        if disposition not in protocol:
            errors.append(f"protocol missing claim disposition {disposition}")

    for required in (
        "mathcal{M}(E,A)",
        "disagreement witness",
        "minimum discriminating evidence",
        "latent_context",
        "selection/censoring ambiguity",
        "repair-entanglement ambiguity",
        "single valid pair",
        "failing to discover a competing model",
        "does not establish",
        "defensible completeness argument",
        "identified conditional on that assumption",
        "Source structural possibility",
        "Build-world selection",
        "Execution realization",
        "Actual-cause attribution",
        "Interventional-effect identifiability",
        "Incident attribution",
        "not a new confirmatory Phase A hypothesis",
        "#10064",
        "#10369",
    ):
        if required.lower() not in protocol.lower():
            errors.append(f"protocol missing required concept {required}")

    for metric in PRIMARY_METRICS:
        if metric not in protocol or metric not in prereg:
            errors.append(f"existing primary measure {metric} must remain unchanged")

    for required in (
        "CEB_REVIEW_CAUSAL_IDENTIFIABILITY_PROTOCOL_V0_2026-11-07.md",
        "CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md",
        "I0",
        "I1",
        "I2",
    ):
        if required not in prereg:
            errors.append(f"preregistration missing composition with identifiability: {required}")

    for required in (
        "Causal-Identifiability",
        "first-pass forms",
        "non-identifiable",
        "discriminating",
    ):
        if required.lower() not in annotators.lower():
            errors.append(f"dual-annotator pack missing identifiability workflow field/concept {required}")

    for required in ("AND gate", "OR gate", "happens_before", "Residual reachability"):
        if required not in topology:
            errors.append(f"causal topology protocol missing composition anchor {required}")

    if "not a sealed holdout" not in challenge.lower():
        errors.append("challenge must disclose that it is not a sealed holdout")
    if "not a gold authority" not in key.lower():
        errors.append("answer key must disclose its non-authoritative synthetic status")
    if "no candidate code" not in protocol.lower():
        errors.append("protocol must preserve source-only no-candidate-execution boundary")

    return errors


def self_test() -> list[str]:
    protocol = "\n".join((
        *DISPOSITIONS,
        "mathcal{M}(E,A)",
        "disagreement witness",
        "minimum discriminating evidence",
        "latent_context",
        "selection/censoring ambiguity",
        "repair-entanglement ambiguity",
        "single valid pair is sufficient",
        "failing to discover a competing model",
        "does not establish",
        "defensible completeness argument",
        "identified conditional on that assumption",
        "Source structural possibility",
        "Build-world selection",
        "Execution realization",
        "Actual-cause attribution",
        "Interventional-effect identifiability",
        "Incident attribution",
        "not a new confirmatory Phase A hypothesis",
        "#10064 #10369",
        *PRIMARY_METRICS,
        "no candidate code",
    ))
    challenge = "\n".join(
        "\n".join((
            f"## {case_id}",
            "Family:",
            "Claim:",
            "Claim kind:",
            "Evidence packet:",
            "Two compatible models",
            "Reviewer question:",
        ))
        for case_id in CASE_IDS
    ) + "\nnot a sealed holdout"
    key = "\n".join(
        "\n".join((
            f"## {i}. {case_id}",
            "Expected disposition:",
            "Claim kind",
            "Material alternatives",
            "Disagreement witness:",
            "Minimum additional evidence:",
            "Claim ceiling:",
        ))
        for i, case_id in enumerate(CASE_IDS, 1)
    ) + "\nnot a gold authority"
    topology = "AND gate OR gate happens_before Residual reachability"
    prereg = "\n".join((
        "CEB_REVIEW_CAUSAL_IDENTIFIABILITY_PROTOCOL_V0_2026-11-07.md",
        "CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md",
        "I0 I1 I2",
        *PRIMARY_METRICS,
    ))
    annotators = "Causal-Identifiability first-pass forms non-identifiable discriminating"

    if validate(protocol, challenge, key, topology, prereg, annotators):
        return ["valid minimal pack rejected"]

    mutants = (
        ("challenge case removed", protocol, challenge.replace("CID-S03", "CID-SXX", 1), key, prereg, annotators),
        ("disposition removed", protocol.replace("partially_identified", "missing_disposition"), challenge, key, prereg, annotators),
        ("missing disagreement witness", protocol, challenge.replace("Two compatible models", "One model", 1), key, prereg, annotators),
        ("missing key claim ceiling", protocol, challenge, key.replace("Claim ceiling:", "Ceiling removed:", 1), prereg, annotators),
        ("missing I2 arm", protocol, challenge, key, prereg.replace("I0 I1 I2", "I0 I1"), annotators),
    )
    for label, mutated_protocol, mutated_challenge, mutated_key, mutated_prereg, mutated_annotators in mutants:
        result = validate(
            mutated_protocol,
            mutated_challenge,
            mutated_key,
            topology,
            mutated_prereg,
            mutated_annotators,
        )
        if not result:
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
        print("CEB_REVIEW_CAUSAL_IDENTIFIABILITY_SELF_TEST_PASS: 5 negative controls")

    try:
        documents = [
            read_required(args.repo_root, relative)
            for relative in (PROTOCOL, CHALLENGE, KEY, TOPOLOGY, PREREG, ANNOTATORS)
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
        "CEB_REVIEW_CAUSAL_IDENTIFIABILITY_PACK_PASS: "
        "cases=6 dispositions=5 alternatives=6 answer_key=separate-public "
        "topology_composition=1 preregistration_composition=1"
    )
    print(
        "CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CLAIM_BOUNDARY: documentation-only static validation; "
        "no candidate code, tests, builds, traces, interventions, graders or benchmarks executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
