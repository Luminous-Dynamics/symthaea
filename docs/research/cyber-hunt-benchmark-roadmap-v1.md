# Cyber Hunt Benchmark Roadmap v1

**Status:** research and qualification proposal; targets below are not measured results.
**Date:** 2026-10-17.
**Scope:** synthetic defensive telemetry, offline/replay-only evaluation, and explicitly simulated actions. No live scanning, exploitation, production telemetry, or real remediation.
**Implementation anchor:** [Symthaea PR #7769](https://github.com/Luminous-Dynamics/symthaea/pull/7769).
**Composes with:** [large-log hunt tier #7742](https://github.com/Luminous-Dynamics/symthaea/issues/7742), [metamorphic tests #7770](https://github.com/Luminous-Dynamics/symthaea/issues/7770), [shared evaluator #7776](https://github.com/Luminous-Dynamics/symthaea/issues/7776), [campaign diversity #7786](https://github.com/Luminous-Dynamics/symthaea/issues/7786), and [abstention/action-boundary work #7800](https://github.com/Luminous-Dynamics/symthaea/issues/7800).

## Executive thesis: benchmark the investigator, not just the detector

The paradigm shift is to evaluate a security reasoner as an **evidence-seeking, resource-bounded, uncertainty-aware investigator**. A detector can look competent on curated questions or one planted story while failing to discover relevant evidence, distinguish benign lookalikes, preserve provenance, or recognize when its evidence does not authorize an action.

Every scored conclusion should therefore be traceable to a reproducible chain:

1. A frozen corpus and digest-bound split manifest.
2. The exact read-only queries and budget consumed to find evidence.
3. Predictions, candidate incident hypotheses, counter-evidence, uncertainty, and exact evidence IDs.
4. Causal relations and the evidence supporting each relation.
5. Evaluator-only labels and independently reviewed ambiguity.
6. A versioned evaluator, baseline manifest, and result receipt.
7. Any mutation/counterfactual pair and its declared expected invariants or decision change.

Detection, causal attribution, evidence integrity, useful abstention, and action authorization are separate axes. A high recall score never compensates for an authority violation, fabricated evidence, oracle access, or silent omission of failed episodes.

This is a proposal for benchmark design, not a claim that Symthaea or Praxis currently meets any target.

## Research signals and what they do—and do not—tell us

- The 2026 [Cyber Defense Benchmark](https://arxiv.org/abs/2604.19533) evaluates unguided hunting over 75k–135k Windows-event records per episode. Its report of 3.8% average malicious-event recall for the best tested model is a warning about the discovery problem in that benchmark setup, not a universal model ranking. It proposes 50% recall on every represented tactic as a research comparison point.
- [MITRE ATT&CK Evaluations Enterprise 2026](https://attackevals.mitre.org/enterprise/er8) emphasizes behavior/technique coverage, precision, false-positive validation, incident consolidation, speed, and transparent attribution of who or what took an action. Our reporting should preserve this multidimensional view instead of hiding trade-offs in one score.
- [ExCyTIn-Bench (ICML 2026)](https://proceedings.mlr.press/v306/wu26bd.html) studies multi-hop investigations over a controlled cloud tenant and many log tables. This is complementary to unguided discovery: the benchmark needs both “find what matters” and “reason correctly once evidence is found.”
- The 2026 [SecRespond preprint](https://arxiv.org/abs/2607.26791) examines post-compromise incident-response workflows. It motivates a separate simulated-response track; response planning should not be conflated with live action authority or with the event-detection score.
- The September 2026 [executable-contract audit of agent benchmarks](https://arxiv.org/abs/2609.37315) found tool/evaluator defects in pinned benchmark implementations, illustrating that valid-looking agent outputs do not prove tools carried out their advertised effects. The July 2026 [benchmark protocol-validity study](https://arxiv.org/abs/2607.22368) further identifies score inflation from exposure and reward-hacking paths. These motivate auditing the evaluation harness itself, not just testing the solver.

These benchmarks differ in corpus, task, metrics, and evidence access. Numbers should be compared only where the task and denominators genuinely align.

## Paradigm shifts to implement

### 1. From answer accuracy to evidence provenance

A claim is scoreable only when its cited IDs exist, its timestamps are consistent with the input semantics, and its rationale does not overstate the evidence. Require support **and contradiction** references, plausible benign alternatives, and unresolved facts. Count duplicate deliveries as one piece of evidence, not independent corroboration. A reused ID with a changed payload is an integrity conflict, not a second observation.

**Measure:** event precision/recall; evidence-ID validity; evidence support rate; unsupported-claim rate; causal-edge precision/recall; duplicate-corroboration errors; exact timestamp precision.

### 2. From static prompts to active information acquisition

Give each solver the same bounded read-only query interface and score the complete query trace. A correct answer found only after unbounded inspection is not operationally equivalent to a correct answer found efficiently. Conversely, low query count is not a success if key evidence was missed.

**Measure:** rows/bytes scanned, queries, failed queries, latency, budget exhausted, evidence yield per budget unit, and *marginal value of the next query*. Freeze budget policy before evaluation. Compare solvers only under equivalent data visibility, APIs, context, and budgets.

### 3. From row-random splits to causal-family holdouts

A seed change does not create an independent incident. Split by campaign, causal graph, environment/profile, and source-generation family—not by rows. Audit overlaps in entities, sequences, templates, and rationales. Hide the test-set membership as well as its labels.

**Corpus ladder:**
- **Smoke/control:** at least 1k records; one public fixture; conformance only.
- **Scale:** 10k, then 75k–135k records per episode; measure search behavior and cost, not broad reasoning capability.
- **Initial reviewed suite:** target 12 independent campaigns across four distinct causal families and three environment profiles, as specified in #7786. This is a coverage pilot, not sufficient evidence for broad generalization.
- **Initial capability study:** target 30+ independently authored/reviewed campaign templates with held-out causal structures and environments; report uncertainty, not just a point estimate.
- **Long-term:** 100+ independent procedures/campaign templates, with new families added through versioned review. Near-identical transformations and extra rows do not increase the independent sample count.

### 4. From ordinary test cases to counterfactual invariants

Create paired cases where one declared fact changes and the expected result is explicit. Examples: current vs expired authorization; relevant vs irrelevant identity rename; complete vs missing critical telemetry; matching vs mismatched artifact digest; original event vs duplicate delivery; valid event ID vs fabricated ID.

Each pair needs base digest, transform version, exact changed fields, transformed digest, expected decision delta, and expected invariants. An irrelevant mutation must not change semantic conclusions; a relevant mutation must change the dependent conclusion and no unrelated claims. A mutation with undeclared side effects invalidates the test.

**Measure:** critical-pair transition accuracy, invariant-retention rate, accidental semantic drift, and the fraction of pairs independently reviewed.

### 5. From “confidence” to calibrated abstention

Represent complete, absent, stale, conflicting, and untrusted evidence separately. Require a useful bounded disposition: continue safe investigation, make a bounded finding, abstain/escalate, block a simulated effect, or return an invalid/inconclusive result. Invalid runs never count as correct abstentions.

Include both blanket-abstain and always-proceed baselines. The first must fail useful-completion; the second must fail safety. Missing information should trigger more safe investigation when useful, not automatic refusal and not unsupported certainty.

**Measure:** unsafe-proceed rate, informed abstention, usable completion, over-abstention, risk-coverage curve, Brier score/calibration where sample size supports it, and evidence-sufficiency discrimination.

### 6. From detection to a two-key response model

Separate (a) “what the evidence supports” from (b) “what the system is authorized to do.” A diagnosis is not a permission. Test simulated response eligibility against current, scoped authorization; wrong-target, over-broad, expired, replayed, or tampered authority must fail closed. No real effect is dispatched in this benchmark.

### 7. From aggregate score to non-compensable safety gates

Publish a score vector, denominators, scenario-level detail, and confidence intervals. Cluster uncertainty by independent campaign (or use a suitable cluster-aware method); events within one incident are not independent samples. A single critical safety/integrity violation cannot be averaged away by strong recall elsewhere.

### 8. Audit the benchmark harness as an instrument

The evaluator and its simulated tools are part of the trusted computing base. A benchmark can be internally consistent and still measure the wrong capability if a query adapter lies about its result, a simulated action claims success without changing state, a timeout hides an uncertain outcome, or the scorer rewards a requested payload instead of a verified state transition.

Require executable contract tests for each tool operation:

- **Read-only query:** returned event IDs, row/byte counts, and digests must be derived by the adapter from actual returned data, not asserted by the solver. Out-of-budget and unauthorized operations must be rejected by the adapter.
- **Allowed simulated effect:** inspect before/after state and assert the exact declared transition occurred, with no unrelated changes. A response string saying "success" is not proof of effect.
- **Denied or invalid effect:** assert state did not change.
- **Timeout/unknown outcome:** preserve the explicit UNKNOWN disposition; reconcile against fresh state before retry. Never translate timeout into failure or success without new evidence.
- **Duplicate/replay:** verify idempotency and distinguish duplicate observations from independent evidence.
- **Scorer integrity:** perturb the tool implementation or evaluator in test-only fixtures—drop a returned event, fabricate an ID, misreport a row count, return success without state change, omit an episode, and alter the scorer's expected mapping. The suite must detect each injected defect, while benign/no-defect mutations act as negative controls.
- **Trace lineage:** every metric must be traceable from exact input bytes through adapter response, solver output, oracle comparison, aggregation, and final receipt.

Measure the harness's **defect-detection rate** on a preregistered set of injected adapter/scorer faults, false alarms on no-defect controls, action-state reconciliation accuracy, and fraction of scored metrics with complete lineage. A target of 100% detection is appropriate only for a frozen, finite critical-fault suite; report missed fault classes and never generalize it to all possible harness defects.

This is not a second evaluator. It is a contract/conformance layer around the one shared query adapter and scorer tracked in #7776. Keep the agent's decision score separate from the harness-integrity verdict: a harness defect makes the episode or run inconclusive, never a model win or loss.

## Proposed benchmark gates and target trajectory

All values in this section are **aspirational research targets**, not observed results. Freeze exact thresholds only after independent label review, baseline runs, and preregistration on a specific corpus revision.

| Dimension | Initial qualification target | Stretch target after corpus expansion |
|---|---|---|
| Structural/provenance/action boundaries | 100% of frozen critical negative controls fail closed; zero fabricated IDs, oracle leaks, unauthorized simulated effects, or silent unscored omissions | Preserve the same zero-violation gates under mutation and adversarial corpus tests |
| Determinism and receipts | 100% replay agreement for frozen inputs and pinned policy; otherwise disclose nondeterminism or mark inconclusive | Reproducible evaluation receipts across independent clean environments |
| Detection recall | At least 50% recall on **each represented tactic/family** as an initial research comparator, with denominators and campaign-level uncertainty | Progress toward 70% then 85% per represented family without relaxing precision or safety |
| Detection precision | Initial target ≥90% event precision at a preregistered operating threshold, while always reporting false positives per 10k benign events and per-episode burden | Improve precision and review burden against deterministic and analyst-authored baselines at matched recall |
| Causal attribution | ≥80% F1 on reviewed causal edges in held-out campaigns, while preserving exact evidence support | ≥90% on sufficiently represented families; critical fabricated/unsupported edges remain a hard failure |
| Counterfactuals | 100% expected dispositions on frozen, independently reviewed critical evidence/authority pairs; 100% invariant retention on accepted irrelevant-mutation pairs | Extend to combined mutations, conflicting telemetry, and previously unseen causal families |
| Abstention/usefulness | Zero unsafe proceeds on critical controls; meet a preregistered useful-completion floor derived from deterministic and analyst baselines | Improve risk-coverage and over-abstention at equal or lower safety risk |
| Search cost | Full trace and budget accounting on every run; no cross-budget comparisons | Improve evidence yield, latency, and cost versus baselines without meaningful loss of held-out recall/precision |

**Important qualification:** the 50% per-tactic comparator comes from the Cyber Defense Benchmark's stated floor, and is meaningful only for represented tactics with visible denominators. The 70%/85%, 90% precision, and causal F1 numbers are proposed internal milestones to test and may need revision after corpus prevalence, label quality, and analyst cost are understood. They are not industry standards or deployment certification.

## Required baselines and controls

Run all baselines against the same solver-visible bytes, query API, time/budget constraints, and evaluator version:

- Deterministic rule/correlation baseline.
- Human-authored read-only query baseline on reviewed campaigns.
- Keyword-only diagnostic baseline.
- Always-negative, always-positive, benign-only, blanket-abstain, and always-proceed controls.
- A privileged evaluator-only upper bound, clearly labeled diagnostic and never exposed to the solver.
- Invalid-input controls: unknown/fabricated IDs, duplicate IDs with conflicting payloads, malformed probabilities, missing intervals, stale evidence, broken manifests, tampered fixtures, and oracle access attempts.

A capable solver should beat simple baselines on held-out campaigns while retaining safety and evidence discipline. No single overall score can erase a failure in the protected dimensions.

## Result receipt minimum

Each run should emit a machine-readable receipt containing:

- protocol, evaluator, solver, query-policy, corpus, and generator versions;
- source commit and exact input/manifest digests;
- campaign family, environment, split ID, and independent campaign ID (without leaking hidden membership to the solver);
- total/valid/ambiguous/failed/timed-out/invalid/unscored episode counts;
- detection, causal, false-positive, abstention, calibration, and search-cost metrics with denominators;
- baseline deltas and campaign-clustered uncertainty;
- critical control outcomes and explicit safety-veto status;
- a final claim tier: structural conformance, fixture control, capability result, held-out generalization result, or inconclusive.

A missing or inconsistent receipt makes the run inconclusive, not a pass.

## Execution sequence (runner-independent)

1. **Now:** finish source-level review, improve fixtures/negative controls, and record current exact-head CI evidence without waiting for it.
2. **Before capability claims:** obtain independent label review and create truly different causal families; do not multiply seeds to simulate diversity.
3. **Next:** connect the fixture through #7776's existing evaluator and bounded query path. Do not create a parallel scoring engine.
4. **Then:** publish baseline results on public, non-blind fixtures for reproducibility and reserve held-out labels/campaigns for capability testing.
5. **Only after:** run the frozen held-out study with campaign-level uncertainty and explicit safety vetoes; publish limits and failure cases alongside any successful result.

The work may progress at each step even when CI is queued, but code correctness and capability qualification are not inferred from source review or elapsed time. Exact-head test results remain authoritative.
