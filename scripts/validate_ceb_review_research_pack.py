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
PRODUCTION_IMPACT = "docs/research/CEB_REVIEW_PRODUCTION_IMPACT_PROTOCOL_V0_2026-11-07.md"
CROSS_FRAMING = "docs/research/CEB_REVIEW_CROSS_FRAMING_ANCHOR_PROTOCOL_V0_2026-11-07.md"
CROSS_FRAMING_CALIBRATION_PACK = "docs/research/CEB_REVIEW_CROSS_FRAMING_CALIBRATION_PACK_V0_2026-11-07.md"
COUNTERFACTUAL_QUALIFICATION = "docs/research/CEB_REVIEW_COUNTERFACTUAL_CONTROL_QUALIFICATION_V0_2026-11-07.md"
GOLD_COMPLETENESS = "docs/research/CEB_REVIEW_GOLD_COMPLETENESS_AND_OPEN_SET_DISCOVERY_V0_2026-11-07.md"
SEQUENTIAL_EVIDENCE = "docs/research/CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md"
EVIDENCE_CROSSWALK = "docs/research/CEB_REVIEW_EVIDENCE_ACTION_CROSSWALK_V0_2026-11-07.md"
CALIBRATION_PACK = "docs/research/CEB_REVIEW_HUMAN_CALIBRATION_PACK_V0_2026-11-07.md"
CALIBRATION_KEY = "docs/research/CEB_REVIEW_HUMAN_CALIBRATION_KEY_V0_2026-11-07.md"
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
    "Kodus State of AI Code Review",
    "Martian Code Review Benchmark",
    "SWE-sweep",
    "LoopControlBench",
    "Saving SWE-Bench",
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
    calibration_pack: str,
    calibration_key: str,
    counterfactual_qualification: str,
    cross_framing: str,
    cross_framing_calibration_pack: str,
    production_impact: str,
) -> list[str]:
    errors: list[str] = []

    ids, blocks = scenario_blocks(challenge)
    expected_ids = list(range(1, 33))
    if ids != expected_ids:
        errors.append(
            f"evidence challenge must contain ERG-S01..ERG-S32 exactly once and in order; got {ids}"
        )
    if len(blocks) != 32:
        errors.append(f"evidence challenge must contain 32 scenario blocks; got {len(blocks)}")
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
        "https://arxiv.org/abs/2510.08996",
        "https://www.codereviewbench.com/leaderboard",
        "https://www.codereviewbenchmark.com/",
        "https://swesweep.com/paper",
        "https://swesweep.com/",
        "https://github.com/kltownsend/loopcontrolbench",
        "https://arxiv.org/abs/2510.08996",
        "https://www.microsoft.com/en-us/research/publication/saving-swe-bench-a-benchmark-mutation-approach-for-realistic-agent-evaluation/",
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
    if "3.13 From uptake to verified engineering impact" not in scorecard:
        errors.append("engineering scorecard is missing the adoption-versus-correctness paradigm shift")
    if "3.14 From historical fix commits to qualified counterfactual controls" not in scorecard:
        errors.append("engineering scorecard is missing the counterfactual-control qualification paradigm shift")
    if "3.15 From fixed benchmark prompts to matched information framings" not in scorecard:
        errors.append("engineering scorecard is missing the cross-framing paradigm shift")

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
    if "Martian production review adoption" not in triangulation:
        errors.append("triangulation must distinguish Martian production adoption")
    if "suggestion adoption is not ground-truth correctness" not in triangulation.lower():
        errors.append("triangulation must distinguish suggestion adoption from factual correctness")
    if "### 2.7 SWE-sweep" not in triangulation or "4,068 bugs across 100 repositories and 22 languages" not in triangulation:
        errors.append("triangulation must include SWE-sweep open-world discovery with its exact population")
    if "4.7%" not in triangulation or "9.5%" not in triangulation or "different model/harness snapshot" not in triangulation:
        errors.append("SWE-sweep paper result and live leaderboard snapshot must remain separate")
    if "### 2.8 LoopControlBench" not in triangulation or "46/70 = 65.7%" not in triangulation:
        errors.append("triangulation must include LoopControlBench's scoped deterministic result")
    if "not 65.7% of software bugs generally" not in triangulation.lower():
        errors.append("LoopControlBench narrow-population caveat is missing")
    if "validator, not the model, decides done" not in triangulation.lower():
        errors.append("LoopControlBench deterministic completion-authority lesson is missing")
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
        if re.match(r"^\| ERG-S\d{2} \|", line)
    ]
    crosswalk_ids = [
        int(re.match(r"^\| ERG-S(\d{2}) \|", line).group(1))
        for line in crosswalk_rows
    ]
    if crosswalk_ids != list(range(1, 33)):
        errors.append(
            f"evidence-action crosswalk must map ERG-S01..ERG-S32 exactly once and in order; got {crosswalk_ids}"
        )
    if len(crosswalk_rows) != 32:
        errors.append(f"evidence-action crosswalk must contain 32 rows; got {len(crosswalk_rows)}")
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

    case_ids = re.findall(r"^## \d+\. Case (CAL-\d{2})", calibration_pack, re.MULTILINE)
    expected_cases = ["CAL-01", "CAL-02", "CAL-03"]
    if case_ids != expected_cases:
        errors.append(f"human calibration pack must contain CAL-01..CAL-03 exactly once and in order; got {case_ids}")
    for case_id in expected_cases:
        match = re.search(
            rf"^## \d+\. Case {case_id}\b([\s\S]*?)(?=^## \d+\. Case CAL-|^## \d+\. Cross-case|\Z)",
            calibration_pack,
            re.MULTILINE,
        )
        block = match.group(1) if match else ""
        variants = [int(x) for x in re.findall(r"^#### Variant X([1-6])\b", block, re.MULTILINE)]
        if variants != [1, 2, 3, 4, 5, 6]:
            errors.append(f"{case_id}: expected exactly neutral variants X1..X6; got {variants}")
        for phrase in ("Normative contract", "Candidate finding to assess", "Reviewer worksheet"):
            if phrase not in block:
                errors.append(f"{case_id}: missing {phrase}")
    key_rows = re.findall(
        r"^\| (CAL-\d{2}) \| X([1-6]) \| (P|I|F|N|M|IRR) —",
        calibration_key,
        re.MULTILINE,
    )
    key_pairs = [(case_id, int(variant)) for case_id, variant, _ in key_rows]
    if len(key_rows) != 18 or len(set(key_pairs)) != 18:
        errors.append(f"calibration key must map 18 unique case/variant pairs; got {len(key_rows)} rows and {len(set(key_pairs))} unique")
    if set(case_id for case_id, _, _ in key_rows) != set(expected_cases):
        errors.append("calibration key must cover all three calibration cases")
    for case_id in expected_cases:
        states = [state for row_case, _, state in key_rows if row_case == case_id]
        if sorted(states) != sorted(["P", "I", "F", "N", "M", "IRR"]):
            errors.append(f"{case_id}: key must assign each P/I/F/N/M/IRR state exactly once")
    for required in (
        "public synthetic development/calibration material",
        "not an admitted task set",
        "permanently ineligible as a sealed CEB holdout",
        "No code was compiled or executed",
    ):
        if required.lower() not in calibration_pack.lower():
            errors.append(f"human calibration pack is missing claim boundary: {required}")
    if "public development answer key" not in calibration_key.lower():
        errors.append("calibration answer key must remain labelled public development material")
    if "not hidden ground truth" not in calibration_key.lower():
        errors.append("calibration answer key must not claim hidden/oracle authority")

    if "### 2.9 Saving SWE-Bench" not in triangulation:
        errors.append("triangulation is missing Saving SWE-Bench prompt-framing evidence")
    if "Cross-Framing Anchor Protocol V0" not in triangulation or "Cross-Framing Anchor Protocol V0" not in preregistration:
        errors.append("triangulation and causal preregistration must link the cross-framing protocol")

    for required in (
        "F0 — Formal issue-conditioned repair",
        "F1 — Realistic user request",
        "F2 — Diff-only review",
        "F3 — Repository-only single-defect discovery",
        "F4 — Failing-test-guided repair",
        "F5 — Open-world multi-bug sweep",
        "Stratum A — Single-defect common anchor",
        "Stratum B — Multi-defect repository anchor",
        "same source/defect anchor",
        "wording-only mutation",
        "information content",
        "randomize condition order",
        "fresh isolated contexts",
        "clustered under the original anchor",
        "do not create a composite",
        "not a universal correction factor",
        "#10064",
        "#10369",
        "no task is admitted",
    ):
        if required.lower() not in cross_framing.lower():
            errors.append(f"cross-framing protocol is missing {required}")

    for required in (
        "C1 — Contract preservation",
        "C2 — Target identity",
        "C3 — Information inventory",
        "C4 — Objective/output",
        "C5 — Interaction realism",
        "C6 — Target leakage",
        "W — Wording-only",
        "R — Information-reduced",
        "A — Information-added",
        "O — Objective-changed",
        "X — Invalid / unresolved",
        "two reviewers independently",
        "not convert these labels into a new metric",
    ):
        if required.lower() not in cross_framing.lower():
            errors.append(f"cross-framing prompt-fidelity protocol is missing {required}")

    anchor_headings = (
        "## 3. Single-defect anchor CF-A01",
        "## 4. Single-defect anchor CF-A02",
        "## 5. Single-defect anchor CF-A03",
    )
    for heading in anchor_headings:
        if cross_framing_calibration_pack.count(heading) != 1:
            errors.append(f"cross-framing calibration pack must contain exactly one {heading}")
    for condition in range(5):
        if cross_framing_calibration_pack.count(f"### F{condition} —") != 3:
            errors.append(f"cross-framing calibration pack must contain F{condition} on each of three single-defect anchors")
    if cross_framing_calibration_pack.count("### F5 — Open-world multi-bug sweep") != 1:
        errors.append("cross-framing calibration pack must contain exactly one separate F5 multi-bug sweep")
    if cross_framing_calibration_pack.count("## 6. Multi-defect repository anchor CF-B01") != 1:
        errors.append("cross-framing calibration pack must contain exactly one multi-defect CF-B01 anchor")
    for required in (
        "Cross-framing information matrix",
        "wording-only mutation",
        "information-package contrast",
        "fresh isolated conversation/context",
        "permanently ineligible as a sealed CEB holdout",
        "No code was compiled or executed",
        "not an admitted task set",
    ):
        if required.lower() not in cross_framing_calibration_pack.lower():
            errors.append(f"cross-framing calibration pack is missing {required}")

    for required in (
        "Reviewer 1 C1–C6 dispositions",
        "Reviewer 2 C1–C6 dispositions",
        "Final framing class: W / R / A / O / X",
        "Information ledger (present / absent / added)",
        "F4 is normally A",
        "Do not label F1 as W",
    ):
        if required.lower() not in cross_framing_calibration_pack.lower():
            errors.append(f"cross-framing calibration pack is missing prompt-fidelity field: {required}")

    for required in (
        "six independent control-qualification dimensions",
        "Q1 — Identity and lineage",
        "Q2 — Change disentanglement",
        "Q3 — Causal mechanism and reachability",
        "Q4 — Intervention fidelity",
        "Q5 — Environment and applicability",
        "Q6 — Outcome and oracle validity",
        "historical source-state observations",
        "minimal in the causal model",
        "not presumed to be a minimal intervention",
        "confounders",
        "equivalent or likely equivalent",
        "source-only structural claims",
        "#10064",
        "#10369",
        "No new primary metric",
    ):
        if required.lower() not in counterfactual_qualification.lower():
            errors.append(f"counterfactual-control protocol is missing {required}")

    for required in (
        "Four states",
        "Truth",
        "Utility",
        "Uptake",
        "Outcome",
        "cluster-randomized presentation experiment",
        "intention-to-treat",
        "blinded",
        "before treatment",
        "Never hide, delay or downgrade critical",
        "No uptake score is added",
        "No new metric",
        "no causal claim",
        "#10064",
        "#10369",
    ):
        if required.lower() not in production_impact.lower():
            errors.append(f"production-impact protocol is missing {required}")
    for required in (
        "https://kodus.io/data",
        "https://www.codereviewbenchmark.com/",
        "https://review-bench.ai/",
    ):
        if required not in production_impact:
            errors.append(f"production-impact protocol is missing external source {required}")
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
        "https://www.codereviewbenchmark.com/",
        "MECHANISM_MISDIAGNOSIS grounded augmented fixed companion does not create another review lane",
        "candidate-specific and appears in both numerator and denominator",
        "CodeReviewBench human-flagged bug recall",
        "Martian production review adoption; suggestion adoption is not ground-truth correctness",
        "SWE-sweep LoopControlBench 4,068 bugs across 100 repositories and 22 languages 4.7% 9.5% different model/harness snapshot",
        "### 2.7 SWE-sweep ### 2.8 LoopControlBench 46/70 = 65.7% not 65.7% of software bugs generally validator, not the model, decides done",
        "https://swesweep.com/paper https://swesweep.com/ https://github.com/kltownsend/loopcontrolbench",
        "Gold-Completeness and Open-Set Discovery",
        "Sequential Evidence Acquisition",
        "#10064 #10369",
    ))
    challenge = "\n\n".join(
        f"### ERG-S{i:02d} — Synthetic\n\n**Evidence pack:** X\n\n"
        "**Expected review behavior:** Y\n\n**Invalid behaviors:** Z"
        for i in range(1, 33)
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
        "3.12 From tool use to sequential evidence acquisition\\n"
        "3.13 From uptake to verified engineering impact\n"
        "3.14 From historical fix commits to qualified counterfactual controls"
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

    calibration_pack = "\n".join((
        "public synthetic development/calibration material; not an admitted task set",
        *(
            f"## {idx + 2}. Case CAL-{case:02d}\nNormative contract\nCandidate finding to assess\n"
            + "\n".join(f"#### Variant X{variant}\nsource excerpt" for variant in range(1, 7))
            + "\nReviewer worksheet"
            for idx, case in enumerate(range(1, 4))
        ),
        "permanently ineligible as a sealed CEB holdout; No code was compiled or executed",
    ))
    calibration_key = "\n".join((
        "public development answer key; not hidden ground truth",
        *(
            f"| CAL-{case:02d} | X{variant} | {state} — state |"
            for case in range(1, 4)
            for variant, state in enumerate(["P", "I", "F", "N", "M", "IRR"], 1)
        ),
    ))
    counterfactual_qualification = "\n".join((
        "six independent control-qualification dimensions",
        "Q1 — Identity and lineage",
        "Q2 — Change disentanglement",
        "Q3 — Causal mechanism and reachability",
        "Q4 — Intervention fidelity",
        "Q5 — Environment and applicability",
        "Q6 — Outcome and oracle validity",
        "historical source-state observations",
        "minimal in the causal model; not presumed to be a minimal intervention",
        "confounders; equivalent or likely equivalent; source-only structural claims",
        "No new primary metric; #10064 #10369",
    ))

    cross_framing = "\\n".join((
        "F0 — Formal issue-conditioned repair",
        "F1 — Realistic user request",
        "F2 — Diff-only review",
        "F3 — Repository-only single-defect discovery",
        "F4 — Failing-test-guided repair",
        "F5 — Open-world multi-bug sweep",
        "Stratum A — Single-defect common anchor",
        "Stratum B — Multi-defect repository anchor",
        "same source/defect anchor; wording-only mutation; information content",
        "randomize condition order; fresh isolated contexts; clustered under the original anchor",
        "do not create a composite; not a universal correction factor; no task is admitted; #10064 #10369",
        "C1 — Contract preservation; C2 — Target identity; C3 — Information inventory; C4 — Objective/output; C5 — Interaction realism; C6 — Target leakage",
        "W — Wording-only; R — Information-reduced; A — Information-added; O — Objective-changed; X — Invalid / unresolved; two reviewers independently; not convert these labels into a new metric",
    ))

    cross_framing_calibration_pack = "\n".join((
        "not an admitted task set; public development calibration; permanently ineligible as a sealed CEB holdout",
        "## 3. Single-defect anchor CF-A01",
        "### F0 — Formal issue-conditioned repair",
        "### F1 — Realistic user request",
        "### F2 — Diff-only review",
        "### F3 — Repository-only discovery",
        "### F4 — Failing-test-guided repair",
        "## 4. Single-defect anchor CF-A02",
        "### F0 — Formal issue-conditioned repair",
        "### F1 — Realistic user request",
        "### F2 — Diff-only review",
        "### F3 — Repository-only discovery",
        "### F4 — Failing-test-guided repair",
        "## 5. Single-defect anchor CF-A03",
        "### F0 — Formal issue-conditioned repair",
        "### F1 — Realistic user request",
        "### F2 — Diff-only review",
        "### F3 — Repository-only discovery",
        "### F4 — Failing-test-guided repair",
        "## 6. Multi-defect repository anchor CF-B01",
        "### F5 — Open-world multi-bug sweep",
        "Cross-framing information matrix; wording-only mutation; information-package contrast; fresh isolated conversation/context; No code was compiled or executed",
        "Reviewer 1 C1–C6 dispositions; Reviewer 2 C1–C6 dispositions; Final framing class: W / R / A / O / X; Information ledger (present / absent / added)",
        "F4 is normally A; Do not label F1 as W",
    ))

    production_impact = "\\n".join((
        "Truth Utility Uptake Outcome",
        "cluster-randomized presentation experiment",
        "intention-to-treat",
        "blinded before treatment",
        "Never hide, delay or downgrade critical security findings",
        "No uptake score is added; No new metric",
        "no causal claim #10064 #10369",
        "https://kodus.io/data",
        "https://www.codereviewbenchmark.com/",
        "https://review-bench.ai/",
    ))

    def run(t: str, ch: str, a: str, s: str, p: str, g: str, e: str, x: str, cp: str, ck: str, q: str, cf: str, cfp: str, pi: str) -> list[str]:
        return validate_documents(t, ch, a, s, p, g, e, x, cp, ck, q, cf, cfp, pi)

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
        preregistration, gold_completeness, sequential_evidence, crosswalk,
        calibration_pack, calibration_key, counterfactual_qualification, cross_framing, cross_framing_calibration_pack, production_impact
    )
    if valid:
        return [f"minimal valid authoring pack rejected: {valid[0]}"]

    mutations = (
        ("duplicate scenario identity", "challenge", challenge.replace("ERG-S02", "ERG-S01", 1)),
        ("missing evidence pack", "challenge", challenge.replace("**Evidence pack:** X", "Missing", 1)),
        ("missing expected behavior", "challenge", challenge.replace("**Expected review behavior:** Y", "Expected", 1)),
        ("missing comparator", "triangulation", triangulation.replace("SecPRBench", "OtherBench")),
        ("missing CodeReviewBench", "triangulation", triangulation.replace("CodeReviewBench", "OtherBench")),
        ("missing Kodus state report", "triangulation", triangulation.replace("Kodus State of AI Code Review", "OtherBench")),
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
        ("missing uptake scorecard shift", "scorecard", scorecard.replace("3.13 From uptake to verified engineering impact", "shift removed")),
        ("missing calibration case", "calibration_pack", calibration_pack.replace("Case CAL-02", "Case CAL-XX", 1)),
        ("missing calibration variant", "calibration_pack", calibration_pack.replace("#### Variant X6", "#### Variant X7", 1)),
        ("missing calibration key mapping", "calibration_key", calibration_key.replace("| CAL-02 | X3 |", "| CAL-02 | X8 |", 1)),
        ("duplicate calibration key mapping", "calibration_key", calibration_key.replace("| CAL-02 | X1 |", "| CAL-01 | X1 |", 1)),
        ("missing control disentanglement", "counterfactual_qualification", counterfactual_qualification.replace("Q2 — Change disentanglement", "Q2 removed")),
        ("missing SWE-sweep population", "triangulation", triangulation.replace("4,068 bugs across 100 repositories and 22 languages", "population removed")),
        ("missing LoopControlBench scope caveat", "triangulation", triangulation.replace("not 65.7% of software bugs generally", "claim removed")),
        ("missing F5 framing condition", "cross_framing", cross_framing.replace("F5 — Open-world multi-bug sweep", "F5 removed")),
        ("missing CF-A02 F4", "cross_framing_calibration_pack", cross_framing_calibration_pack.replace("### F4 — Failing-test-guided repair", "### F4 removed", 2)),
        ("missing W disposition", "cross_framing", cross_framing.replace("W — Wording-only", "W removed")),
        ("missing second prompt reviewer", "cross_framing_calibration_pack", cross_framing_calibration_pack.replace("Reviewer 2 C1–C6 dispositions", "Second reviewer removed")),
        ("collapsed truth and uptake", "production_impact", production_impact.replace("Truth Utility Uptake Outcome", "Review success")),
        ("missing intention-to-treat", "production_impact", production_impact.replace("intention-to-treat", "per-protocol only")),
        ("randomized critical findings", "production_impact", production_impact.replace("Never hide, delay or downgrade critical", "May delay critical")),
        ("uptake becomes metric", "production_impact", production_impact.replace("No uptake score is added", "Uptake primary score")),
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
            "calibration_pack": calibration_pack,
            "calibration_key": calibration_key,
            "counterfactual_qualification": counterfactual_qualification,
            "cross_framing": cross_framing,
            "cross_framing_calibration_pack": cross_framing_calibration_pack,
            "production_impact": production_impact,
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
            args["calibration_pack"],
            args["calibration_key"],
            args["counterfactual_qualification"],
            args["cross_framing"],
            args["cross_framing_calibration_pack"],
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
        print("CEB_REVIEW_RESEARCH_SELF_TEST_PASS: 37 negative controls")

    try:
        triangulation = read_required(args.repo_root, TRIANGULATION)
        challenge = read_required(args.repo_root, CHALLENGE)
        adjudication = read_required(args.repo_root, ADJUDICATION)
        scorecard = read_required(args.repo_root, SCORECARD)
        preregistration = read_required(args.repo_root, PREREGISTRATION)
        gold_completeness = read_required(args.repo_root, GOLD_COMPLETENESS)
        sequential_evidence = read_required(args.repo_root, SEQUENTIAL_EVIDENCE)
        evidence_crosswalk = read_required(args.repo_root, EVIDENCE_CROSSWALK)
        calibration_pack = read_required(args.repo_root, CALIBRATION_PACK)
        calibration_key = read_required(args.repo_root, CALIBRATION_KEY)
        counterfactual_qualification = read_required(args.repo_root, COUNTERFACTUAL_QUALIFICATION)
        cross_framing = read_required(args.repo_root, CROSS_FRAMING)
        cross_framing_calibration_pack = read_required(args.repo_root, CROSS_FRAMING_CALIBRATION_PACK)
        production_impact = read_required(args.repo_root, PRODUCTION_IMPACT)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures = validate_documents(
        triangulation, challenge, adjudication, scorecard, preregistration,
        gold_completeness, sequential_evidence, evidence_crosswalk,
        calibration_pack, calibration_key, counterfactual_qualification, cross_framing,
        cross_framing_calibration_pack, production_impact
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    ids, _ = scenario_blocks(challenge)
    print(
        "CEB_REVIEW_RESEARCH_PACK_PASS: "
        f"scenarios={len(ids)} comparators={len(COMPARATORS)} "
        f"adjudication_stages=5 primary_metrics={len(PRIMARY_METRICS)} "
        "counterfactual_states=6 experimental_phases=3 gold_protocol=1 evidence_policy=1 crosswalk_rows=26 "
        "human_calibration_cases=3 variants=18 answer_key=separate-public "
        "counterfactual_controls=6 qualification_dimensions=6 cross_framing_conditions=6 cross_framing_anchors=4 prompt_variants=16 production_impact_protocol=1"
    )
    print(
        "CEB_REVIEW_RESEARCH_CLAIM_BOUNDARY: documentation validation only; "
        "no candidate code, Cargo command, tests, graders, benchmark tasks or model evaluations executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
