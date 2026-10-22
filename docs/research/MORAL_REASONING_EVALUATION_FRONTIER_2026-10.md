# Moral Reasoning Evaluation Frontier — October 2026

**Status:** Research and implementation plan. This document does not claim moral competence or benchmark superiority.

## Executive principle

Do not build a second ethics engine or treat the Love Coherence scalar as an ethical verdict. Extend the existing EthicsEngine with a qualification program that can fail closed on invalid evidence, distinguish hard constraints from trade-offs, retain principled disagreement, and measure consequences independently.

The current Love Coherence implementation contained two contract defects addressed by the companion code change:

- remapping a neutral harmony mean of zero into the interval midpoint assigned a score of 0.5 despite no positive coordinate evidence;
- the scalar API had no explicit validation result for non-finite/out-of-range coordinates or malformed interaction matrices.

The patch adds a checked API, preserves the old API as a compatibility wrapper that defaults to a zero score on invalid input, and records mathematical invariants. This improves score hygiene; it is not evidence that the score tracks actual care or moral competence.

**Important residual risk:** the existing EthicsEngine still uses Love Coherence to modulate confidence. This change makes the score's neutral/invalid behavior more conservative, but does not independently validate that coupling or remove the feedback path. The exact-head review must inspect downstream effects, and confidence-boosting should remain disabled or separately gated until an ablation shows calibrated benefit on held-out outcomes.


## What recent evaluation work changes

Moral reasoning evaluation is moving beyond one-answer accuracy:

- **MoReBench (January 2026)** emphasizes reasoning process and pluralism rather than simple verdict matching: https://labs.scale.com/blog/morebench
- **Beyond Verdicts (AAAI 2026)** identifies gaps in moral sensitivity to noisy contexts, reasoning quality, steerability, and evaluations that assume a single ground truth: https://ojs.aaai.org/index.php/AAAI/article/view/41131
- **PluriHarms (ICLR 2026)** measures both perceived harm and human disagreement, using 150 prompts and 15,000 ratings from 100 annotators: https://proceedings.iclr.cc/paper_files/paper/2026/hash/04cea5800f4d71fe0d89d5f64e33408e-Abstract-Conference.html
- **OVERTONBENCH (ICLR 2026)** measures coverage of diverse viewpoints rather than assuming agreement is always the target: https://proceedings.iclr.cc/paper_files/paper/2026/hash/4444eb0b68174180d2a46841c951e7f3-Abstract-Conference.html
- **CORDA (August 2026)** tests how a system prioritizes principles when every option has a cost: https://www.raillab.org/publication/singh-2026-corda/
- The 2025 Nature roadmap recommends adversarial and confirmatory evaluations that distinguish appropriate reasoning from merely appropriate-seeming explanations: https://www.nature.com/articles/s41586-025-10021-1
- A 2026 position paper identifies a gap between representing moral values and correctly identifying/applying context-sensitive norms: https://arxiv.org/abs/2608.14566

These benchmarks answer complementary questions. None by itself is a general certificate of moral competence, and leaderboard results cannot be compared unless versions, prompts, decoding, splits, aggregation, and cost are matched.

## Paradigm shifts to test

### 1. From verdicts to evidence-bearing decisions

For each case, record separately:
- observed facts and provenance;
- uncertain or missing facts;
- the affected parties and their expressed preferences;
- applicable normative premises or policy clauses;
- candidate actions and predicted consequences;
- constraints, disagreements, and reasons for abstention;
- what evidence would change the conclusion.

A fluent rationale does not count as proof that the decision process used those considerations. Use interventions that change one morally relevant fact at a time and verify that the action changes only when a declared rule requires it.

### 2. From scalar ethics to constrained, plural objectives

Maintain a vector such as welfare/harm, consent/autonomy, truthfulness, distributional fairness, agency, and reversibility. Keep hard constraints outside compensatory sums. A weighted average must not let creativity, warmth, or predicted benefit numerically cancel explicit refusal or an independently established severe-harm constraint.

Where legitimate frameworks conflict, retain their separate conclusions and identify the premise that changes the recommendation. Do not infer a universal moral priority order from the list order of the harmonies or from empirical majority preference alone.

### 3. From benchmark memorization to disconfirming tests

Use paired counterfactuals and adversarial perturbations:
- care-language manipulation versus verifiable care;
- clear refusal versus unspecified consent;
- truthful, gentle feedback versus comforting deception;
- intervention that improves average welfare while violating a protected party's boundary;
- same scenario under a different legitimate framework profile;
- negation, quotation, role reversal, hypothetical framing, and irrelevant emotionally loaded wording;
- a missing key fact that should trigger clarification rather than confident inference;
- actions with equal predicted mean outcomes but different downside risk.

The public suite is for development. Keep a sealed, independently curated confirmatory set; do not tune on the final benchmark.

### 4. From internal coherence to external consequences

Track four different quantities: semantic recognition, policy reasoning, conduct, and affected-party outcomes. Keep them distinct in every report. Internal harmony, low anomaly score, confidence, and rationale quality are explanatory features—not outcome labels.

## Proposed benchmark ladder

| Gate | Test | Required report |
|---|---|---|
| M0 | Input/math contracts | Invalid-input rejection, neutral-zero property, bounds, finite outputs, permutation invariance, matrix symmetry |
| M1 | Norm sensitivity | Relevant-fact recall, false-positive/false-negative norm application, counterfactual sensitivity |
| M2 | Constraint integrity | Violation rate by severity; explicit refusal, withdrawal, scope and missing-evidence handling |
| M3 | Pluralism | Per-profile accuracy/calibration, viewpoint coverage, disagreement calibration, worst-group slices |
| M4 | Adversarial robustness | Paired consistency, negation/role/irrelevance flips, distribution shift, prompt injection against values |
| M5 | Action-grounded utility | Independently adjudicated harm/welfare/autonomy outcomes, downside risk, reversibility and recovery |
| M6 | Calibration & cost | Abstention quality, confidence calibration, latency, energy/memory/cost, and full versioned provenance |

Compare the current engine, current Love Coherence, a simple rules/constraint-only baseline, a calibrated weighted-score baseline, ablations, and frontier model baselines on identical cases. Report macro and per-domain measures, confidence intervals, calibration, failure severity, and all dataset/model digests. Use task-matched metrics rather than forcing consent, virtue, justice, and harm detection into one accuracy number.

## Advance criteria (proposed, not achieved)

1. 100% pass on declared deterministic mathematical properties and hard-constraint fixture cases.
2. No invalid/non-finite input may produce an ordinary positive coherence score.
3. On a preregistered held-out primary metric, the candidate must beat the strongest simple baseline with uncertainty reported. A relative improvement of 10% is an aspirational target, not evidence until measured.
4. No aggregate gain can mask regressions on explicit refusal, severe foreseeable harm, coercion/manipulation, privacy, or autonomy.
5. Include negative results, uncertainty, annotation disagreement, subgroup slices, efficiency and all baseline outputs.

## Implementation sequence

1. Land the small math-contract correction only after exact-head Rust formatting, tests, Clippy and downstream compatibility checks pass.
2. Add a versioned, typed case contract and deterministic Rust runner in a separate evaluation module; keep the cases task-specific and human-auditable.
3. Replay the same case set with current Love Coherence enabled, disabled, shuffled, and adversarially injected to test for self-confirming confidence/learning feedback.
4. Evaluate the existing EthicsEngine rather than creating a competing engine. Integrate existing plural-ethics and first-principles work through explicit adapters and preserve the existing action/safety gate.
5. Run external comparators only after licensing, task semantics, splits and evaluation settings are reconciled. Do not claim SOTA from a synthetic contract suite.

## Evidence boundary

The code-level tests in this tranche are authored but were not compiled or run in this environment. The document's targets are proposed gates, not results. Even a passing test suite would validate the declared math contract only; it would not establish real-world moral competence, subjective love, or improved welfare.
