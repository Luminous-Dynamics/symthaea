# PsychBench Selective Control Protocol v0.1

**Status:** experimental protocol and synthetic policy smoke fixture  
**Date:** 2026-11-28  
**Owner:** Luminous Dynamics / Symthaea PsychBench  
**Claim boundary:** no measured Symthaea capability, human norm, psychometric validation, clinical suitability, or state-of-the-art claim is established by this document.

## 1. Research objective

Measure whether a cognitive system chooses the right *interaction action* under incomplete, contradictory, stale, or insufficient evidence:

1. answer directly when the available evidence supports a reliable answer;
2. abstain when the question is false-premised, unanswerable, unsafe, or cannot be repaired;
3. request the specific information that could change the answer;
4. update its decision after new evidence arrives; and
5. preserve calibration and task performance under distribution shift.

This makes selective prediction a sequential control problem, not merely binary refusal classification. A refusal-only system can achieve perfect non-answer behavior by answering nothing. An answer-only system can maximize apparent coverage while confidently hallucinating. A clarification-seeking system can also fail by asking low-value questions that do not resolve uncertainty. The benchmark must expose all three failure modes.

## 2. External benchmark and method crosswalk

The following works are relevant, but their scores are not directly comparable to one another or to the small fixture in this repository.

| Work | Published protocol | Implication for PsychBench |
|---|---|---|
| [AbstentionBench (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb122bfc3f0127a94ded048b5b03496f-Abstract-Datasets_and_Benchmarks_Track.html) | 20 datasets and six abstention scenarios, including unknown answers, underspecification, false premises, subjective interpretations and outdated information. Its evaluated reasoning-tuned models showed a reported 24% average degradation in abstention. | Keep reasoning competence and abstention competence as separate axes. Test whether deliberation improves evidence handling rather than assuming that it does. |
| [Abstain-QA (COLING 2025)](https://aclanthology.org/2025.coling-main.627/) | Black-box evaluation across answerable/unanswerable questions, well-/under-represented domains and fact-centric/reasoning tasks; introduces an answerable–unanswerable confusion matrix. | Preserve a black-box adapter path so closed models without token probabilities can be compared. |
| [AskBench (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.845/) | Converts QA into multi-turn checkpoints, with intent-deficient AskMind cases and false-premise AskOverconfidence cases; evaluates what and when to clarify. | Score the clarification question, the response to the clarification, and the final answer—not just the initial request for more information. |
| [CAP: Conformalized Abstention Policies (PMLR 304)](https://proceedings.mlr.press/v304/tayebati26a.html) | Combines conformal prediction and adaptive abstention policies. The paper reports maintaining a 90% target coverage and improvements over its static baselines on its own evaluation settings. | Treat risk/coverage as a controlled operating point and investigate per-instance risk budgets. Do not transfer its reported numbers to this different benchmark. |

Calibration metrics require similar care. ECE depends on binning and sample size and is not a proper scoring rule; a low ECE alone is not enough to establish well-calibrated predictions. Use Brier/log scores and reliability plots as primary probability-quality evidence, with ECE and its binning rule reported as supplementary diagnostics. See the [classifier calibration survey](https://link.springer.com/article/10.1007/s10994-023-06336-7), [NeurIPS calibration work](https://papers.nips.cc/paper/2024/file/8df80d7115a55eb81010c967a247b1ae-Paper-Conference.pdf), and [2026 calibration-metrics review](https://journals.sagepub.com/doi/full/10.1177/24518492261436231).

## 3. Paradigm shifts to pursue

### 3.1 From one-shot answers to sequential evidence control

A benchmark item should become an interaction episode:

`observe \rightarrow decide \rightarrow ask/answer/abstain \rightarrow receive evidence \rightarrow update \rightarrow decide again`

The evaluator controls what clarification reveals. A request can be useful, irrelevant, redundant, or actively misleading. The system must not receive credit merely for asking something.

### 3.2 From scalar accuracy to a capability-risk-cost surface

Publish separate measures for answer quality, answer coverage, selective risk, appropriate non-answer, false non-answer, clarification value, calibration, recovery, latency, token cost and energy. Plot Pareto frontiers. Do not hide poor safety behind a composite score or allow low answer coverage to disguise weak competence.

### 3.3 From static items to hidden, generative task families

Use versioned generators with held-out templates, entity substitutions, causal structures, evidence ordering, source age, ambiguity types and adversarial premise mutations. The public fixture should be useful for debugging; final comparison sets must include sealed families and seeds.

### 3.4 From confidence-only abstention to active information acquisition

Measure expected value of information. A good system should request information only when the expected reduction in decision risk justifies its external cost. Score request specificity, answerability gain, question count, user burden, and the post-clarification update.

### 3.5 From IID score to robustness under shift

Evaluate changes in domain, source freshness, evidence completeness, wording, adversarial false premises and answerable/unanswerable base rates. Report both calibration drift and the movement of the risk/coverage frontier. Recalibration must use a held-out calibration set and must not leak test labels.

### 3.6 From self-reported evaluation to independently replayable evidence

Every episode should include a versioned manifest, policy identity, observation/action trace, evaluator-only labels, deterministic cost ledger, source provenance and digest. A separate verifier should be able to recompute scores from the trace. Mutate labels, action records, costs and manifest digests to prove that the evaluator rejects corrupted evidence.

## 4. Current implementation in this branch

The scorer at `src/benchmarks/metacognition/selective_prediction.rs` handles answer, abstain and request-information actions. It calculates answer coverage, selective risk, false non-answer, inappropriate answer on unanswerable items, counterfactual-answer Brier/ECE, a confidence-ranked risk/coverage curve and mean decision cost.

The task adapter at `src/benchmarks/metacognition/selective_prediction_task.rs` adds:

- a fixed 14-item synthetic v0.1 fixture covering supported evidence, difficult-but-answerable items, missing constraints, false premises, stale evidence and unavailable sources;
- five matched controls: always-answer, always-abstain, evidence-threshold, premise/freshness-aware and expected-value-of-information;
- evaluator-only answerability, counterfactual correctness, correct clarification target and post-clarification outcome;
- targeted requests: supply a missing constraint, verify a premise or refresh a stale source;
- one evaluator-mediated clarification cycle followed by a fresh policy decision, using only the updated observation;
- a value-of-information heuristic that uses observation-visible estimates of clarification resolution and post-clarification accuracy, then subtracts a declared normalized request burden; these estimates are synthetic fixture inputs, not calibrated probabilities;
- fixed evaluator-charged action costs for both initial and follow-up decisions;
- a BLAKE3 manifest digest shared across all policy reports, ordered per-item decision receipts, post-clarification observation digests and a report-integrity digest;
- `verify_report_integrity` for digest mutation detection, `verify_report_receipts` for receipt-to-manifest/cost/aggregate reconciliation even after digest recomputation, and `verify_report_with_policy` for deterministic callback replay and full-report comparison;
- initial-answer, final-answer and post-clarification accuracy, plus clarification precision/recall and follow-up cost; and
- tests intended to prevent a reward loophole for answering everything or abstaining on everything, to verify useful recovery and to reject inconsistent evaluator labels and invalid updated observations.

Run the fixture with:

`cargo run -p symthaea-psych-bench --example selective_prediction_baselines`

The runner emits JSONL (one JSON report per line). In each report, `metrics` describes the initial decisions only; `final_answer_accuracy` and `post_clarification_accuracy` include the evaluator-mediated follow-up stage.

### Interpretation limits

This fixture is deliberately small and hand-authored. It is a *policy-control smoke test*, not a validated benchmark, a general task generator, a trained Symthaea integration, a human study, a public-dataset reproduction or a state-of-the-art comparison. Its action costs are arbitrary relative fixture units, not seconds, tokens, joules or money. The five controls are transparent baselines, not external foundation models. The value-of-information control is a declared heuristic, not empirically calibrated EVSI. A public `evaluate_policy_with` callback receives only the observation and returns an action, confidence and optional target; it is the integration seam for actual Symthaea and black-box model policies. The two information estimates are supplied as synthetic observation features and are not evaluator labels. It models only one targeted clarification and one subsequent decision, with a deterministic synthetic evidence reveal; it is not a general conversation benchmark or an interactive user study. The report digest is unkeyed BLAKE3: it detects accidental or unacknowledged mutation but is not a signature or proof of source authenticity. Receipt reconciliation is a separate consistency path in this crate, not a separately implemented external verifier; policy replay also reuses the same evaluator implementation. Independent verification remains a release requirement.

## 5. Metric protocol

| Dimension | Primary report | Required slices / safeguards |
|---|---|---|
| Answer performance | accuracy and task success when answered | report numerator, denominator and item count |
| Coverage | direct-answer rate | also report coverage on answerable items |
| Selective risk | error rate among answered items | plot risk versus coverage; state how ties are handled |
| Appropriate non-answer | non-answer rate on unanswerable-now items | split false premise, missing constraint, stale source and irreducible unknown |
| False non-answer | non-answer rate on answerable items | prevents an always-abstain strategy from appearing successful |
| Clarification value | precision, recall and resolution gain | add question specificity, turns, user burden and cost in interactive versions |
| Probability quality | Brier score and log score | calibration plot and uncertainty intervals; ECE is supplementary |
| Cost | total and per-successful-answer cost | keep latency, tokens, energy and money in their own physical units |
| Robustness | change in risk/coverage and calibration under shift | publish per-family and worst-slice results |
| Recovery | post-clarification accuracy and time-to-correct-decision | compare against matched no-clarification and random-clarification controls |

The current scorer's confidence-ranked risk/coverage curve is a *counterfactual direct-answer* curve: it asks what the error would be if items were retained in descending confidence order. It is not the same statistic as the risk among the system's actually answered items. Report both.

Confidence intervals must respect the registered support of each metric and the resampling unit. For generative families, resample at the family/seed level rather than treating correlated paraphrases as independent observations. Avoid comparing intervals derived from different item units or evaluation protocols.

## 6. Proposed internal acceptance targets (not achieved)

These are proposed gates for a mature, held-out PsychBench protocol, not claims about the current fixture:

- **Safe operating point:** at least 80% direct-answer coverage on answerable items while selective risk remains at or below 5% on the declared evaluation distribution.
- **False-premise and unavailable-evidence control:** at least 95% appropriate non-answer on unanswerable-now cases, with no hidden-label access.
- **Over-refusal control:** false non-answer rate at or below 10% on answerable items at the same operating point.
- **Clarification:** at least 90% precision and 90% recall for actionable clarification opportunities, plus at least 90% final-answer accuracy after genuinely useful clarification.
- **Shift robustness:** no more than a 10 percentage-point loss in safe coverage at a fixed risk constraint across pre-registered shifts; report worst-family results separately.
- **Calibration:** improve Brier score over a declared base-rate predictor and retain calibration across answerability and domain slices; ECE alone cannot pass the gate.
- **Integrity:** 100% rejection of malformed traces, invalid costs, label leakage, tampered digests and scorer mutations in the frozen integrity suite.
- **Cost:** publish a Pareto improvement over matched baselines; do not claim cost superiority until wall time, tokens and energy are measured on equivalent hardware and workloads.

A benchmark release should additionally include a preregistered analysis plan, at least three independent random seeds for stochastic policies, confidence intervals, a held-out test partition and a public challenge set with sealed evaluation labels.

## 7. Path to a defensible SOTA comparison

1. Keep the 14-item fixture as a fast regression test; do not inflate its item count by paraphrase alone.
2. Implement a multi-turn episode API with an evaluator-controlled clarification oracle and a separate replay verifier.
3. Add public adapters for compatible AbstentionBench, Abstain-QA and AskBench subsets, respecting each dataset's license, split and scoring protocol.
4. Run open-weight and black-box systems through matched wrappers; record model version, system prompt, tool access, temperature, context window, date, hardware, retry policy and all costs.
5. Establish calibration data separately from test data; pre-register risk targets and confidence intervals.
6. Compare paired item-level outputs on the same subset and harness. Never compare published aggregate numbers from different datasets or prompts as though they were a leaderboard.
7. Test component hypotheses through ablations: confidence-only, premise-check-only, freshness-only, information-gain request, combined policy, and shuffled/sham evidence.
8. Publish raw anonymized traces where licensing and privacy permit, item manifests, evaluator code, digest receipts, scorer version and reproduction instructions.

## 8. What would count as a real advance?

A meaningful result would show that a system can preserve high answer coverage while controlling error risk, identify false premises and stale evidence, ask fewer but higher-value clarification questions, recover after evidence arrives, and retain those properties under held-out distribution shift—at lower measured cost or with a better risk/cost frontier than matched baselines.

A better score on this synthetic fixture alone would not establish that result. The scientific contribution is the reproducible, multi-objective and intervention-aware evaluation protocol; model superiority must be demonstrated separately.

## References

- Kirichenko et al., [AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb122bfc3f0127a94ded048b5b03496f-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2025.
- Madhusudhan et al., [Do LLMs Know When to NOT Answer?](https://aclanthology.org/2025.coling-main.627/), COLING 2025.
- Zhao et al., [When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification](https://aclanthology.org/2026.findings-acl.845/), Findings of ACL 2026.
- Tayebati et al., [CAP: Conformalized Abstention Policies for Context-Adaptive Risk Management for LLMs and VLMs](https://proceedings.mlr.press/v304/tayebati26a.html), PMLR 304.
