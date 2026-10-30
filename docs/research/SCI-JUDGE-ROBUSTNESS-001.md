# SCI-JUDGE-ROBUSTNESS-001 — Qualifying the evaluator

**Status:** new protocol and executable audit layer under development. Example artifacts are non-scoring. No Symthaea judge-performance result is claimed.

**Tracking:** [Issue #9059](https://github.com/Luminous-Dynamics/symthaea/issues/9059). This composes with [SCI-EVIDENCE-CONTRACTS-001 / PR #9005](https://github.com/Luminous-Dynamics/symthaea/pull/9005) but audits the evaluator, not the candidate system.

## 1. Paradigm shift: the evaluator is part of the system

An evaluator can be perfectly deterministic yet wrong, or accurate on a fixed set yet unstable under changes that should be irrelevant. Consequently, evaluator quality must separate at least three constructs:

1. **Repeatability:** does the same judge give the same semantic decision on the same input?
2. **Intervention invariance/sensitivity:** does the decision remain stable under a semantics-preserving transformation and change under a meaning-changing control?
3. **Validity:** does the judge agree with independent reference adjudication?

A single judge score confounds these. This protocol reports them separately and makes the judge's model revision, prompt, rubric, decision policy, input variants, reference commitments and repetition schedule precommitted.

## 2. Research basis

- [The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation (GroundLM 2026)](https://aclanthology.org/2026.groundlm-1.64/) reports an 11–14% mean verdict-flip rate on identical reruns across its tested judges, and a large position-swap experiment in which 21–41% of matched comparisons depended on candidate slot rather than content. These are the paper's measurements, not ours.
- [The Judge Can Debias Itself (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.2098/) studies authority, confidence and formatting biases and evaluates bias-aware prompting and self-debiasing. Its reported gains are setting-specific, not guaranteed to transfer.
- [Judging the Judges (IJCNLP 2025)](https://aclanthology.org/2025.ijcnlp-long.18/) studies repetition stability, position consistency and preference fairness across judges and tasks.
- [NIST AI 800-3 (February 2026)](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models) distinguishes fixed-benchmark from generalized accuracy and explains why uncertainty and task-level variation matter.

These sources justify measuring evaluator behavior explicitly. They do not show that any Symthaea judge is unreliable or reliable.

## 3. Audit families and controls

| Family | Controlled change | Invariant or changing property |
|---|---|---|
| repeatability | Repeat identical input several times | Semantic decision should be stable; report pairwise rerun disagreement |
| position_swap | Swap candidate A/B display slots, preserve candidate identity | Semantic winner should remain the same after mapping left/right back to candidate identity |
| formatting_invariance | Change superficial format only | Semantic decision should remain invariant |
| confidence_style_invariance | Change assertiveness or confidence language while factual content remains fixed | Semantic decision should remain invariant |
| authority_cue_invariance | Add a fabricated authority cue that policy says is not evidence | Semantic decision should remain invariant |
| semantic_sensitivity_control | Change evidence so the warranted winner changes | Decision should change; a judge that always returns the same label must fail |

Every input variant has its own digest. Candidate order is recorded separately from the semantic decision: raw left/right judgments are normalized to candidate A/B before computing position sensitivity. The reference decision is evaluator-only and is revealed in the post-run report; the pre-run plan commits to its digest without exposing the raw label.

An authority-cue control is meaningful only if the rubric explicitly establishes that the cue is fabricated or non-authoritative. A real change in source authority is not an invariance case. Likewise, semantic-sensitivity controls must change the underlying evidence in a way that independent adjudicators agree changes the correct decision.

## 4. Machine-enforced invariants

The Rust validator must reject:

- a zero/placeholder digest or a qualified plan whose judge source revision differs from the inspected build;
- a plan with repeated case IDs, mismatched family/intervention semantics, equal input digests where a meaning change is expected, or changed reference decisions in an invariance case;
- a position-swap case that does not actually swap candidate order;
- missing, duplicate, unplanned or extra observations; wrong input digest, condition, order, repeat index or judge receipt digest;
- a reference decision that does not hash to the plan's precommitted digest;
- post-hoc report mutations, a wrong plan byte digest, a wrong judge revision, or absent required observations.

A valid report may faithfully expose an unreliable judge. Report validity authenticates internal accounting and commitments; it is not a quality pass.

## 5. Metrics (separate by family)

- **Exact-repeat disagreement:** fraction of unordered pairs of decisions on the identical input that disagree; abstention and tie are explicit decisions, not missing zeros.
- **Paired variant disagreement:** proportion of matched repeat indices where the normalized semantic decision changes under a transformation expected to preserve meaning.
- **Sensitivity-control pass rate:** proportion of matched repeats whose normalized decisions change under the predeclared meaning-changing control.
- **Position sensitivity:** disagreement after candidate-identity normalization in the position-swap family; report whether raw judgments systematically prefer the left or right slot too.
- **Gold agreement:** proportion of observations matching the independent reference decision, reported independently of stability.
- **Abstention rate:** abstentions / observed judgments, separate from correctness and stability.
- **Coverage/integrity:** planned cases, scheduled observations, observed observations, missing/duplicate/invalid receipts and exact artifact digests.
- **Cost:** judge calls, tokens, wall time and measured cost, where captured; do not report missing metrics as zero.

Integer rates are rounded to millionths. If the relevant denominator is zero, the rate is null, not zero. A stable-but-wrong judge can score well on repeatability and poorly on gold agreement; both results must remain visible. A judge that never changes its decision fails the semantic-sensitivity control.

## 6. Targets and competitive comparison

### Deterministic engineering gate

- Reject 100% of the enumerated malformed, tampered, duplicate, missing, wrong-revision and wrong-order controls in the committed synthetic test suite.
- Accept all clean synthetic invariance and semantic-sensitivity controls in that suite.
- Produce byte-deterministic verification output for the same committed artifacts.
- Preserve runner_execution_proven=false and external_anchor_required=true.

These targets apply only to known synthetic fixtures. They do not estimate judge reliability on open-world inputs.

### Empirical judge qualification

For every judge model/prompt/rubric version, publish per-family rates with confidence intervals. Use repeated trials to quantify nondeterminism. Blind human reference adjudication, predeclare the disagreement resolution process, and report inter-rater agreement. Audit sample selection must include random cases as well as high-disagreement/high-consequence cases so the audit does not overfit the judge's easiest errors.

Compare at least: the unmodified baseline judge prompt, a bias-aware prompt, and a self-critique/debias variant under matched token/call budgets. Where possible, compare multiple judge model families. Freeze versions and sampling parameters. No winner should be selected from a single dataset or by pooling repeat trials as if they were independent task samples.

Use mixed-effects or another design-appropriate model for broader claims, accounting for task/case family and repeated judgements. Report fixed-suite measurements separately from any inference to a larger task population, consistent with NIST AI 800-3.

## 7. Acceptance sequence and claim boundary

1. Validate schemas and illustrative fixtures.
2. Implement typed commitments, exact observation accounting and family-specific invariants.
3. Add synthetic positive controls and hostile mutations exercised through the actual CLI.
4. Pass exact-head formatting, unit tests, Clippy, schema checks and E2E controls.
5. Run a blind evaluator audit with independent gold labels before trusting it to adjudicate the model contracts in #9005.

This protocol does not infer universal truth, causal reasoning or general intelligence. It measures an evaluator's behavior on declared transformations and reference labels. Stability is not correctness, and correctness on a fixed audit is not generalized validity.
