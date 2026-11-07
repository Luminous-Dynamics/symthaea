#!/usr/bin/env python3
"""Static validator for the public CEB code-review research pack.

This tool validates only authoring documents. It never loads task fixtures,
runs Cargo, executes repository-controlled code, invokes a judge, or creates
benchmark tasks or results.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TRIANGULATION = "docs/research/CEB_REVIEW_BENCHMARK_TRIANGULATION_V0_2026-11-07.md"
GOLD_COMPLETENESS = "docs/research/CEB_REVIEW_GOLD_COMPLETENESS_AND_OPEN_SET_DISCOVERY_V0_2026-11-07.md"
SEQUENTIAL_EVIDENCE = "docs/research/CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md"
EVIDENCE_CROSSWALK = "docs/research/CEB_REVIEW_EVIDENCE_ACTION_CROSSWALK_V0_2026-11-07.md"
PREREGISTRATION = "docs/research/CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md"
CHALLENGE = "docs/research/CEB_REVIEW_EVIDENCE_GROUNDING_CHALLENGE_V0_2026-11-06.md"
ADJUDICATION = "docs/research/CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md"
SCORECARD = "docs/research/CEB_ENGINEERING_QUALITY_SCORECARD_V0_2026-11-04.md"

PRIMARY_METRICS = (
    "defect_recall_at_5",
    "actionable_precision",
    "reviewer_burden",
)
EVIDENCE_CLASSES = (
    "source_declared",
    "syntax_derived",
    "compiler_resolved",
    "runtime_observed",
    "intervention_observed",
    "inferred / unresolved",
)
COMPARATORS = (
    "ReviewBench",
    "SecPRBench",
    "CR-Bench-v2",
    "CRBugBench",
    "CodeReviewBench",
)
EXECUTION_GATES = ("#10064", "#10369")


def scenario_blocks(challenge: str) -> tuple[list[int], list[str]]:
    matches = list(re.finditer(r"^### ERG-S(\d{2})\b", challenge, re.MULTILINE))
    ids: list[int] = []
    blocks: list[str] = []
    for index, match in enumerate(matches):
        ids.append(int(match.group(1)))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(challenge)
        blocks.append(challenge[match.start():end])
    return ids, blocks


def validate_documents(
    triangulation: str,
    challenge: str,
    adjudication: str,
    scorecard: str,
    preregistration: str,
    gold_completeness: str,
    sequential_evidence: str,
    evidence_crosswalk: str,
) -> list[str]:
    errors: list[str] = []

    ids, blocks = scenario_blocks(challenge)
    expected_ids = list(range(1, 27))
    if ids != expected_ids:
        errors.append(
            f"evidence challenge must contain ERG-S01..ERG-S26 exactly once and in order; got {ids}"
        )
    if len(blocks) != 26:
        errors.append(f"evidence challenge must contain 26 scenario blocks; got {len(blocks)}")
    for scenario_id, block in zip(ids, blocks):
        label = f"ERG-S{scenario_id:02d}"
        if "**Evidence pack:**" not in block:
            errors.append(f"{label}: missing Evidence pack")
        if not re.search(r"\*\*Expected(?: review)? behavior:\*\*", block):
            errors.append(f"{label}: missing Expected behavior")
        if not re.search(r"\*\*Invalid behaviors?:\*\*", block):
            errors.append(f"{label}: missing Invalid behavior(s)")

    for comparator in COMPARATORS:
        if comparator not in triangulation:
            errors.append(f"triangulation document is missing comparator {comparator}")
    for metric in PRIMARY_METRICS:
        if metric not in triangulation or metric not in adjudication:
            errors.append(f"existing primary metric {metric} must remain mapped in both documents")
    for url in (
        "https://review-bench.ai/",
        "https://arxiv.org/abs/2610.33509",
        "https://arxiv.org/abs/2611.02566",
        "https://arxiv.org/abs/2610.13668",
    ):
        if url not in triangulation:
            errors.append(f"triangulation document is missing source URL {url}")

    for stage in (
        "Stage 1 — Discovery",
        "Stage 2 — Localization",
        "Stage 3 — Causal mechanism",
        "Stage 4 — Evidence authority and freshness",
        "Stage 5 — Decision utility and disposition",
    ):
        if stage not in adjudication:
            errors.append(f"adjudication card is missing {stage}")
    for authority in EVIDENCE_CLASSES:
        if authority not in adjudication:
            errors.append(f"adjudication card is missing existing evidence class {authority}")

    if "MECHANISM_MISDIAGNOSIS" not in triangulation:
        errors.append("triangulation must preserve the mechanism-misdiagnosis failure category")
    if "grounded" not in triangulation.lower() or "augmented" not in triangulation.lower():
        errors.append("triangulation must distinguish grounded and augmented review findings")
    if "fixed companion" not in triangulation.lower():
        errors.append("triangulation must retain fixed-companion limitations")

    for gate in EXECUTION_GATES:
        if gate not in challenge or gate not in adjudication or gate not in triangulation:
            errors.append(f"execution/regrade gate {gate} must remain explicit")
    if not any(phrase in triangulation for phrase in ("does not create another review lane", "No new metric lane")):
        errors.append("triangulation must state that it does not create another review/metric lane")
    if "does not create metrics" not in adjudication:
        errors.append("adjudication card must state that it does not create new metrics")
    if "3.10 From code location to causal mechanism diagnosis" not in scorecard:
        errors.append("engineering scorecard is missing the causal-diagnosis paradigm shift")
    if "3.11 From human-found gold to independently audited open-set discovery" not in scorecard:
        errors.append("engineering scorecard is missing the gold-completeness paradigm shift")
    if "3.12 From tool use to sequential evidence acquisition" not in scorecard:
        errors.append("engineering scorecard is missing the sequential evidence-acquisition paradigm shift")

    required_states = (
        "P — Parent / pre-introduction",
        "I — Introducing",
        "F — Fixed companion",
        "N — Neighboring-mechanism control",
        "M — Minimal causal intervention",
        "IRR — irrelevant intervention",
    )
    for state in required_states:
        if state not in preregistration:
            errors.append(f"causal-diagnosis preregistration is missing counterfactual state {state}")
    for phase in ("Phase A — Source-only diagnosis contract", "Phase B — Evidence acquisition value", "Phase C — Human complementarity"):
        if phase not in preregistration:
            errors.append(f"causal-diagnosis preregistration is missing {phase}")
    for required in (
        "paired task-level analysis",
        "repository-cluster bootstrap",
        "80% power",
        "family-wise alpha 0.05",
        "same total inference budget",
        "no new primary metric lane",
        "#10064",
        "#10369",
        "No tasks are admitted by this preregistration",
    ):
        if required not in preregistration.lower() and required not in preregistration:
            errors.append(f"causal-diagnosis preregistration is missing required safeguard: {required}")
    for metric in PRIMARY_METRICS:
        if metric not in preregistration:
            errors.append(f"causal-diagnosis preregistration must preserve existing primary metric {metric}")
    if "candidate-specific and appears in both numerator and denominator" not in triangulation:
        errors.append("ReviewBench augmented-recall denominator caveat is missing")
    if "CodeReviewBench human-flagged bug recall" not in triangulation:
        errors.append("triangulation must distinguish CodeReviewBench human-flagged-bug recall")
    if "Gold-Completeness and Open-Set Discovery" not in triangulation:
        errors.append("triangulation must link the gold-completeness protocol")
    if "Sequential Evidence Acquisition" not in triangulation:
        errors.append("triangulation must link the sequential evidence-acquisition protocol")
    if "Gold-Completeness and Open-Set Discovery" not in preregistration:
        errors.append("causal-diagnosis preregistration must compose with gold-completeness protocol")
    if "Sequential Evidence Acquisition" not in preregistration:
        errors.append("causal-diagnosis preregistration must compose with sequential evidence policy")

    for required in (
        "Frozen target set",
        "Adjudicated open-set discoveries",
        "Unobserved defect remainder",
        "A naïve capture–recapture estimate",
        "zero-finding PRs",
        "producer-dependence graph",
        "No new primary metric is introduced.",
        "#10064",
        "#10369",
    ):
        if required not in gold_completeness:
            errors.append(f"gold-completeness protocol is missing {required}")

    for required in (
        "Evidence action ladder",
        "Sequential policy arms",
        "Matched-budget accounting",
        "Stopping and escalation policy",
        "smallest applicable observation",
        "No activity credit",
        "#10064",
        "#10369",
    ):
        if required not in sequential_evidence:
            errors.append(f"sequential evidence protocol is missing {required}")

    crosswalk_rows = [
        line for line in evidence_crosswalk.splitlines()
        if re.match(r"^\\| ERG-S\\d{2} \\|", line)
    ]
    crosswalk_ids = [
        int(re.match(r"^\\| ERG-S(\\d{2}) \\|", line).group(1))
        for line in crosswalk_rows
    ]
    if crosswalk_ids != list(range(1, 27)):
        errors.append(
            f"evidence-action crosswalk must map ERG-S01..ERG-S26 exactly once and in order; got {crosswalk_ids}"
        )
    if len(crosswalk_rows) != 26:
        errors.append(f"evidence-action crosswalk must contain 26 rows; got {len(crosswalk_rows)}")
    for scenario_id, row in zip(crosswalk_ids, crosswalk_rows):
        if row.count("|") < 6:
            errors.append(f"ERG-S{scenario_id:02d}: crosswalk row is missing a required column")
    for required in (
        "Competing hypotheses / claim to resolve",
        "Minimum discriminating action",
        "Strongest authority this action can establish",
        "Stop / escalate condition",
        "E3/E4 remain design-only until #10064 qualification",
        "#10369",
    ):
        if required not in evidence_crosswalk:
            errors.append(f"evidence-action crosswalk is missing {required}")

    return errors


def self_test() -> list[str]:
    triangulation = "\n".join((
        *COMPARATORS,
        *PRIMARY_METRICS,
        "https://review-bench.ai/",
        "https://arxiv.org/abs/2610.33509",
        "https://arxiv.org/abs/2611.02566",
        "https://arxiv.org/abs/2610.13668",
        "https://www.codereviewbench.com/leaderboard",
        "MECHANISM_MISDIAGNOSIS grounded augmented fixed companion does not create another review lane",
        "candidate-specific and appears in both numerator and denominator",
        "CodeReviewBench human-flagged bug recall",
        "Gold-Completeness and Open-Set Discovery",
        "Sequential Evidence Acquisition",
        "#10064 #10369",
    ))
    challenge = "\n\n".join(
        f"### ERG-S{i:02d} — Synthetic\n\n**Evidence pack:** X\n\n"
        "**Expected review behavior:** Y\n\n**Invalid behaviors:** Z"
        for i in range(1, 27)
    ) + "\n#10064 #10369"
    adjudication = "\n".join((
        *PRIMARY_METRICS,
        *EVIDENCE_CLASSES,
        "Stage 1 — Discovery",
        "Stage 2 — Localization",
        "Stage 3 — Causal mechanism",
        "Stage 4 — Evidence authority and freshness",
        "Stage 5 — Decision utility and disposition",
        "does not create metrics",
        "#10064 #10369",
    ))
    scorecard = (
        "3.10 From code location to causal mechanism diagnosis\n"
        "3.11 From human-found gold to independently audited open-set discovery\n"
        "3.12 From tool use to sequential evidence acquisition"
    )
    preregistration = "\n".join((
        "P — Parent / pre-introduction",
        "I — Introducing",
        "F — Fixed companion",
        "N — Neighboring-mechanism control",
        "M — Minimal causal intervention",
        "IRR — irrelevant intervention",
        "Phase A — Source-only diagnosis contract",
        "Phase B — Evidence acquisition value",
        "Phase C — Human complementarity",
        "paired task-level analysis repository-cluster bootstrap 80% power family-wise alpha 0.05",
        "same total inference budget no new primary metric lane #10064 #10369 No tasks are admitted by this preregistration",
        "Gold-Completeness and Open-Set Discovery",
        "Sequential Evidence Acquisition",
        *PRIMARY_METRICS,
    ))
    gold_completeness = "\n".join((
        "Frozen target set",
        "Adjudicated open-set discoveries",
        "Unobserved defect remainder",
        "A naïve capture–recapture estimate",
        "zero-finding PRs",
        "producer-dependence graph",
        "No new primary metric is introduced.",
        "#10064 #10369",
    ))
    sequential_evidence = "\n".join((
        "Evidence action ladder",
        "Sequential policy arms",
        "Matched-budget accounting",
        "Stopping and escalation policy",
        "smallest applicable observation",
        "No activity credit",
        "#10064 #10369",
    ))

    def run(t: str, ch: str, a: str, s: str, p: str, g: str, e: str, x: str) -> list[str]:
        return validate_documents(t, ch, a, s, p, g, e, x)

    crosswalk = "\n".join((
        "| Scenario | Competing hypotheses / claim to resolve | Minimum discriminating action | Strongest authority this action can establish | Stop / escalate condition |",
        *(
            f"| ERG-S{i:02d} | target vs alternative | E1 smallest action | source_declared | stop or ask |"
            for i in range(1, 27)
        ),
        "E3/E4 remain design-only until #10064 qualification; #10369",
    ))
    valid = run(
        triangulation, challenge, adjudication, scorecard,
        preregistration, gold_completeness, sequential_evidence, crosswalk
    )
    if valid:
        return [f"minimal valid authoring pack rejected: {valid[0]}"]

    mutations = (
        ("duplicate scenario identity", "challenge", challenge.replace("ERG-S02", "ERG-S01", 1)),
        ("missing evidence pack", "challenge", challenge.replace("**Evidence pack:** X", "Missing", 1)),
        ("missing expected behavior", "challenge", challenge.replace("**Expected review behavior:** Y", "Expected", 1)),
        ("missing comparator", "triangulation", triangulation.replace("SecPRBench", "OtherBench")),
        ("missing CodeReviewBench", "triangulation", triangulation.replace("CodeReviewBench", "OtherBench")),
        ("missing evidence authority", "adjudication", adjudication.replace("compiler_resolved", "deleted_class")),
        ("missing isolation gate", "challenge", challenge.replace("#10064", "removed_gate")),
        ("missing primary metric mapping", "adjudication", adjudication.replace("reviewer_burden", "removed_metric")),
        ("missing counterfactual state", "preregistration", preregistration.replace("M — Minimal causal intervention", "M-state-removed")),
        ("missing power rule", "preregistration", preregistration.replace("80% power", "unspecified power")),
        ("missing phase C", "preregistration", preregistration.replace("Phase C — Human complementarity", "phase omitted")),
        ("missing frozen target separation", "gold", gold_completeness.replace("Frozen target set", "target missing")),
        ("missing no-naive-capture warning", "gold", gold_completeness.replace("A naïve capture–recapture estimate", "estimate missing")),
        ("missing zero-finding audit", "gold", gold_completeness.replace("zero-finding PRs", "no clean sample")),
        ("missing evidence action ladder", "evidence", sequential_evidence.replace("Evidence action ladder", "ladder missing")),
        ("missing stop rule", "evidence", sequential_evidence.replace("Stopping and escalation policy", "stop missing")),
        ("missing cost accounting", "evidence", sequential_evidence.replace("Matched-budget accounting", "budget missing")),
        ("duplicate crosswalk scenario", "crosswalk", crosswalk.replace("ERG-S02", "ERG-S01", 1)),
        ("missing crosswalk scenario", "crosswalk", crosswalk.replace("| ERG-S02 |", "| ERG-SXX |", 1)),
        ("missing crosswalk execution gate", "crosswalk", crosswalk.replace("#10064", "gate removed")),
    )
    for label, target, mutant in mutations:
        args = {
            "triangulation": triangulation,
            "challenge": challenge,
            "adjudication": adjudication,
            "scorecard": scorecard,
            "preregistration": preregistration,
            "gold": gold_completeness,
            "evidence": sequential_evidence,
            "crosswalk": crosswalk,
        }
        args[target] = mutant
        result = run(
            args["triangulation"],
            args["challenge"],
            args["adjudication"],
            args["scorecard"],
            args["preregistration"],
            args["gold"],
            args["evidence"],
            args["crosswalk"],
        )
        if not result:
            return [f"negative self-test was accepted: {label}"]

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
        print("CEB_REVIEW_RESEARCH_SELF_TEST_PASS: 20 negative controls")

    try:
        triangulation = read_required(args.repo_root, TRIANGULATION)
        challenge = read_required(args.repo_root, CHALLENGE)
        adjudication = read_required(args.repo_root, ADJUDICATION)
        scorecard = read_required(args.repo_root, SCORECARD)
        preregistration = read_required(args.repo_root, PREREGISTRATION)
        gold_completeness = read_required(args.repo_root, GOLD_COMPLETENESS)
        sequential_evidence = read_required(args.repo_root, SEQUENTIAL_EVIDENCE)
        evidence_crosswalk = read_required(args.repo_root, EVIDENCE_CROSSWALK)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures = validate_documents(triangulation, challenge, adjudication, scorecard, preregistration, gold_completeness, sequential_evidence, evidence_crosswalk)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    ids, _ = scenario_blocks(challenge)
    print(
        "CEB_REVIEW_RESEARCH_PACK_PASS: "
        f"scenarios={len(ids)} comparators={len(COMPARATORS)} "
        f"adjudication_stages=5 primary_metrics={len(PRIMARY_METRICS)} "
        "counterfactual_states=6 experimental_phases=3 gold_protocol=1 evidence_policy=1 crosswalk_rows=26"
    )
    print(
        "CEB_REVIEW_RESEARCH_CLAIM_BOUNDARY: documentation validation only; "
        "no candidate code, Cargo command, tests, graders, benchmark tasks or model evaluations executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
