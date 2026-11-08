#!/usr/bin/env python3
"""Static authoring validator for the CEB risk-controlled selective acceptance protocol.

Reads one public Markdown research document only. Does not evaluate candidate
claims, calculate empirical risk, calibrate thresholds, or execute code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DOC = "docs/research/CEB_REVIEW_RISK_CONTROLLED_SELECTIVE_CLAIM_ACCEPTANCE_PROTOCOL_V0_2026-11-08.md"
REQUIRED = (
    "False-discovery risk is not false-positive rate",
    "Development / policy selection",
    "Calibration / risk certification",
    "Confirmatory evaluation",
    "Clopper–Pearson",
    "Beta",
    "59 accepted independent calibration claims",
    "299 accepted independent calibration claims",
    "S0 — accept-all baseline",
    "S1 — verbal-confidence threshold",
    "S2 — disagreement/evidence route",
    "S3 — calibrated selective policy",
    "one-sided 95% upper FDR bound at or below 5%",
    "one-sided 95% upper FDR bound at or below 1%",
    "at least 70% coverage",
    "no new task schema",
    "#10064",
    "#10369",
)


def validate(doc: str) -> list[str]:
    errors: list[str] = []
    for item in REQUIRED:
        if item.lower() not in doc.lower():
            errors.append(f"missing required selective-risk concept: {item}")
    if "risk threshold takes precedence over the coverage target" not in doc.lower():
        errors.append("must make risk threshold take precedence over coverage aspiration")
    if "no risk guarantee is claimed" not in doc.lower():
        errors.append("must not imply a current CEB risk guarantee")
    if "clustering" not in doc.lower():
        errors.append("must disclose dependency/cluster assumption")
    if "not candidate-controlled" not in doc.lower() and "candidate-controlled execution remains gated" not in doc.lower():
        errors.append("must retain candidate-execution boundary")
    return errors


def self_test() -> list[str]:
    baseline = "\n".join(REQUIRED) + (
        "\nRisk threshold takes precedence over the coverage target. "
        "No risk guarantee is claimed. Clustering matters. "
        "Candidate-controlled execution remains gated."
    )
    if validate(baseline):
        return ["valid minimal selective-risk protocol rejected"]
    mutations = (
        ("FDR/FPR conflation", baseline.replace("False-discovery risk is not false-positive rate", "FDR equals FPR")),
        ("calibration split removed", baseline.replace("Calibration / risk certification", "Calibration removed")),
        ("risk sample burden removed", baseline.replace("59 accepted independent calibration claims", "58 examples")),
        ("critical risk target removed", baseline.replace("one-sided 95% upper FDR bound at or below 1%", "critical risk unknown")),
        ("risk precedence removed", baseline.replace("Risk threshold takes precedence over the coverage target", "coverage first")),
    )
    for label, mutant in mutations:
        if not validate(mutant):
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
        print("CEB_REVIEW_SELECTIVE_RISK_SELF_TEST_PASS: 5 negative controls")

    try:
        doc = (args.repo_root / DOC).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {DOC}: {exc}", file=sys.stderr)
        return 1

    failures = validate(doc)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "CEB_REVIEW_SELECTIVE_RISK_PROTOCOL_PASS: "
        "dispositions=4 calibration_roles=3 policy_arms=4 risk_targets=2"
    )
    print(
        "CEB_REVIEW_SELECTIVE_RISK_CLAIM_BOUNDARY: static document validation only; "
        "no thresholds calibrated, no risk guarantee established, no candidate claims evaluated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
