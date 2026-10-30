# Broca Memory-Use Calibration Protocol

**Status:** proposed diagnostic protocol; no model evaluation or benchmark result is attached.

## 1. Why correctness is not enough

A memory system can retrieve a true, current, provenance-backed proposition and the agent can still use it incorrectly:

- **Over-use:** an irrelevant fact changes the answer, decision, or action.
- **Under-use:** a required constraint is present and admissible, but the agent ignores it.
- **Mis-weighting:** a weak preference dominates a hard safety or policy constraint.
- **Context hijack:** a salient but low-relevance memory steers the output away from task evidence.
- **Conflict collapse:** the agent arbitrarily chooses one of multiple supported, conflicting propositions.
- **Stale influence:** a retracted or out-of-cutoff item continues influencing the answer even after the retrieval layer should have excluded it.

This is separate from memory-state correctness. An item can be valid but irrelevant; a fact can be relevant but insufficient; a retrieved item can be correctly rejected by the answer model. Broca should therefore evaluate at least three independent stages:

1. **State validity:** is this proposition supported, current, authorized, and temporally eligible?
2. **Retrieval policy:** should this proposition be supplied for this task, and with what role?
3. **Reader influence:** did the agent use, ignore, or weight it in the intended way?

## 2. Research grounding

MemCalib (September 2026) argues that memory evaluation has undermeasured whether an LLM gives each proposition the appropriate degree of influence. It reports over-use and under-use by frontier open and closed models, and proposes MemCalib-RL using ordered bidirectional counterfactual credit assignment and exact atom ablation:

- Cao et al., *MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents* (2026): https://arxiv.org/abs/2609.24259

LongMemEval-V2 measures long-history evidence gathering and the accuracy-latency frontier; it does not, by itself, isolate whether a particular memory atom had the correct causal influence on the answer:

- *LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues*: https://arxiv.org/abs/2605.12493
- Official results and LAFS framing: https://xiaowu0162.github.io/longmemeval-v2/

STATE-Bench measures task success, five-run reliability, interaction quality and cost; it is useful for the downstream operational effect of memory, after the atom-level diagnostic is established:

- Microsoft, *Introducing STATE-Bench*: https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/
- Official leaderboard: https://microsoft.github.io/STATE-Bench/leaderboard/

MemEvoBench explores long-horizon memory contamination from adversarial injection, noisy tools and biased feedback. It motivates a safety lane for memory influence under repeated exposure:

- Xie et al., *MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents* (2026): https://arxiv.org/abs/2604.15774

These benchmarks address complementary questions. Do not pool their scores or claim that a result on one establishes performance on another.

## 3. A typed influence contract for each memory atom

For each evaluation task, the benchmark oracle labels each candidate memory atom with one intended influence role:

MemCalib's three target levels provide the reader-influence axis: **Ignore** (no answer-specific footprint), **Bound** (bounded local support), and **Control** (controls a material conclusion or constraint). Broca adds a separate, hard eligibility gate before a memory atom reaches that axis.

| Broca evidence role | Eligibility | Target influence | Intended reader behavior |
|---|---|---|---|
| Required constraint / required fact | Supported and eligible | Control | Must govern the relevant conclusion or action |
| Helpful context / optional preference | Supported and eligible | Bound | May shape local details but not override Control atoms |
| Irrelevant | May be valid, but task-irrelevant | Ignore | No answer-specific footprint; output should be invariant |
| Unsupported | Fails support verification | Blocked; Ignore in bypass tests | Must not be asserted as a fact |
| Invalidated / out of cutoff / unauthorized | Fails eligibility gate | Blocked; Ignore in bypass tests | Must not reach the reader; bypass must have zero influence |
| Conflict set | Multiple incompatible propositions supported | Control | Preserve the conflict or abstain; never choose arbitrarily |

Eligibility and influence must not be conflated. A signed but unauthorized or out-of-cutoff atom is not a low-confidence memory; it is ineligible. Conversely, a valid proposition can still have a target influence of Ignore for a particular task. The target label is benchmark ground truth, not an unconstrained confidence number that the reader is asked to assign itself.

MemCalib's published dataset contains 15,000 examples across health, general assistance, and coding, with 234,220 atomic propositions and a 13,500/1,500 train/test split. It labels each proposition's target as Ignore, Bound, or Control and uses atom-specific rubrics to assess actual use. Broca's 12-case fixture is only an initial systems diagnostic; it is not a substitute for this scale or for the official MemCalib test split.

## 4. Counterfactual experiment design

For each task, freeze the reader model, system prompt, tool access, generation settings, memory ordering, context budget, and random seeds where deterministic decoding is available. Save the exact memory atoms and their provenance before every run.

Run these conditions:

- **C0 Full eligible memory:** the normal memory interface returns the permitted atoms.
- **C1 No memory:** remove all memory, keeping task evidence and budget accounting explicit.
- **C2 Single-atom ablation:** remove one memory atom at a time.
- **C3 Role-preserving shuffle:** shuffle atom order while preserving roles and content.
- **C4 Irrelevant injection:** add valid but task-irrelevant atoms with controlled salience and volume.
- **C5 Invalidated injection:** pass stale/retracted atoms through a test-only bypass to confirm the reader/verifier fails closed even under adversarial presentation.
- **C6 Conflict condition:** supply two independently supported incompatible propositions with equal authority.
- **C7 Counterfactual replacement:** replace one atom with a controlled opposite value while preserving all other fields and evidence structure.
- **C8 Multi-atom ablation:** remove interacting groups to expose cases where the effect of two premises is not captured by single-atom deletion.

Ablation is an intervention, not a perfect causal estimator in every model. Language generation is stochastic and token interactions are non-additive. Therefore report single-item effects, selected pair/group interactions, repeated trials, and uncertainty rather than interpreting each output difference as a clean causal coefficient.

## 5. Scoring

Score structured decisions first, then prose quality. For each task, define a canonical decision tuple such as:

`(action, target, constraints_satisfied, cited_atoms, abstention_state)`.

Then compute:

### Memory-use correctness

- **Required-use recall:** proportion of required atoms whose content is correctly reflected in the canonical answer/decision.
- **Required-use omission rate:** required constraints present in eligible memory but absent from the decision.
- **Irrelevant-invariance rate:** proportion of irrelevant-atom additions/removals that leave the canonical decision unchanged.
- **Invalid-memory influence rate:** proportion of tasks where invalidated, unauthorized, or out-of-cutoff atoms influence the decision.
- **Conflict-preservation rate:** fraction of supported conflict cases where the system returns conflict/uncertainty rather than an arbitrary value.
- **Counterfactual compliance:** fraction of controlled value replacements for which the decision changes in the oracle-specified direction.
- **Evidence attribution precision/recall:** whether cited atoms actually support the claims attributed to them.
- **Over-use rate:** irrelevant, optional, or lower-priority memory changes a hard decision or overrides a stronger constraint.
- **Under-use rate:** a required memory atom is not reflected when the oracle says it is decision-relevant.

### Influence calibration

For each atom (m), define an oracle target class (y_m) and an observed influence class (hat y_m) using the structured decision under controlled ablation/replacement. Publish the complete confusion matrix and class-wise precision/recall. Do not collapse required-use and irrelevant-invariance into one accuracy figure: a system can trivially avoid over-use by ignoring every memory, or avoid under-use by overfitting to every atom.

A future probabilistic reader may emit an influence estimate (p_m); only then should calibration metrics such as Brier score or expected calibration error be applied. Do not infer calibrated probabilities from a binary output difference.

### Systems efficiency

Report p50/p95/p99 end-to-end latency, memory preparation time, token count, number of atoms, ablation runs, model calls, cost per correct task, and repeated-run reliability. Include both the production memory path and a bounded-context baseline.

## 6. Required diagnostic scenarios

Build at least these controlled cases:

| Case | Memory challenge | Expected outcome |
|---|---|---|
| U01 | Required hard constraint among many unrelated memories | Constraint respected |
| U02 | Relevant fact absent, no admissible substitute | Explicit unknown / clarification |
| U03 | Valid but irrelevant salient memory | Decision invariant |
| U04 | Optional preference conflicts with a hard constraint | Hard constraint wins |
| U05 | Retracted memory inserted through a test-only bypass | Zero influence or explicit rejection |
| U06 | Two equally supported contradictory values | Conflict, no arbitrary selection |
| U07 | Replace a required fact with its controlled opposite | Decision changes appropriately |
| U08 | One of two independent justifications is removed | Claim and downstream result remain supported |
| U09 | Last justification is removed | Claim becomes unknown/unsupported |
| U10 | Late evidence is excluded from an as-known cutoff | No temporal leakage |
| U11 | Large distractor volume and order perturbation | Required decisions remain stable |
| U12 | Two individually optional atoms jointly satisfy a requirement | Pair interaction is detected |

For each case store symbolic ground truth, admissibility labels, intended influence roles, expected canonical decision, counterfactual variants, model/harness manifest, and exact input digests. Use fictional identities and synthetic events to avoid leaking personal information.

## 7. Baselines and ablations

Compare the same reader under:

- No persistent memory / bounded history.
- Raw-history retrieval (BM25 or the official benchmark's required baseline).
- Broca support-state only (verified, typed evidence).
- Broca support-state plus role labels, without learned influence scoring.
- Broca plus a calibrated reader policy.
- Gold evidence and gold role labels (upper diagnostic control).
- The official MemCalib/MemCalib-RL baselines where licensing and harness terms permit.

The important comparison is not just whether the full-memory condition beats no memory. It is whether Broca improves *selective use*: correct use of required atoms, invariance to irrelevant atoms, zero influence from invalid memory, and improved task outcomes at equal context and compute budgets.

## 8. Proposed internal promotion gates

These are proposed targets, not measured results or official benchmark thresholds:

1. 100% correct state eligibility on deterministic support and temporal fixtures.
2. Zero influence from invalidated or out-of-cutoff atoms in structured decision outputs.
3. Zero arbitrary winner selection in conflict cases.
4. 100% required-constraint compliance on deterministic cases with complete evidence.
5. At least 99% irrelevant-invariance on the controlled synthetic suite, with confidence intervals and disclosed model variance.
6. Positive paired task-success gain or a lower cost per correct task without weakening gates 1–4.
7. No claim of generalization until held-out seeds, paraphrase variants, long-history distractors, and external benchmark splits are evaluated.

A system that always abstains can satisfy safety gates; report useful-answer coverage and task success beside all abstention/conflict metrics.

## 9. Reporting and anti-leakage rules

- Preregister the hypothesis, task split, reader model/version, context budget, sampling configuration, stopping rules, and primary endpoints.
- Split by underlying event graph and scenario template, not merely by paraphrase, to avoid near-duplicate leakage.
- Keep generated natural-language renderings and gold atom roles out of the reader prompt.
- Use the same answer reader and tool budget for each compared memory system.
- Preserve raw outputs, canonical decisions, retrieved atom IDs, support certificates, run seeds, token accounting, costs, latency traces, and code/data digests.
- Report all failed and invalid runs. Never relabel timeouts, refusals, or malformed outputs as successful abstentions without a preregistered rule.
- Distinguish retrieval absence from reader non-use: run a gold-memory condition and capture pre-answer memory payloads.
- Report single-run quality and repeat-run reliability separately.

## 10. Implementation sequence

1. Pin the symbolic role schema and 12 synthetic cases.
2. Add a Broca adapter that exports only query-eligible propositions, support witnesses, conflict states, and role candidates. It must not infer that cryptographic validity implies task relevance.
3. Create a fixed-reader harness with exact per-atom ablations and structured decision extraction.
4. Add paired bootstrap confidence intervals grouped by underlying event graph.
5. Compare no-memory, raw-history, Broca-only, and Broca-plus-calibration conditions.
6. Integrate with LongMemEval-V2 for long-history evidence gathering and STATE-Bench for downstream pass@1/pass^5/cost. Use MemCalib's official split/protocol where possible.
7. Only after the diagnostic demonstrates reliable selective use should any memory-use calibration signal influence production generation.

## 11. Non-claims

This protocol is not a benchmark submission, an evaluated calibration method, or evidence that Broca already improves a reader's use of memory. It separates the new question—*did a valid memory have the right influence?*—from the existing questions of evidence validity, temporal correctness, and alternative support.
