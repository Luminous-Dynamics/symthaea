# MACE v0.1 — Decision-Value Retrieval Controller and Counterfactual Evaluation

**Status:** research / implementation contract. This document specifies the next controller increment; it is not evidence that the controller, adapters, or benchmark runs are implemented.  
**Related:** [#13477](https://github.com/Luminous-Dynamics/symthaea/issues/13477), [#13455](https://github.com/Luminous-Dynamics/symthaea/issues/13455), [#13261](https://github.com/Luminous-Dynamics/symthaea/issues/13261), [#13284](https://github.com/Luminous-Dynamics/symthaea/issues/13284), [PR #13436](https://github.com/Luminous-Dynamics/symthaea/pull/13436).

## 1. Problem statement

MACE already has retrieval trace and strategy-memory types, but the LongMemEval-V2 sidecar does not yet operationally promote and consume advisory strategy notes. Its current retrieval path is BM25-only. Repeated successful traces are useful support evidence, but they do not establish that a learned strategy improves held-out decisions.

The engineering objective is to turn retrieval from a single ranked lookup into a **budgeted sequential evidence-acquisition policy**. The policy may choose among typed retrieval operations, inspect the result, update its estimate of what remains useful, and either continue, stop, or abstain.

Three boundaries are non-negotiable:

1. Strategy memory proposes *where and how to search*. It does not qualify evidence.
2. The independent qualifier determines which source observations may enter a qualified evidence set. The retrieval backend cannot qualify its own output.
3. A decision receipt, action authorization, consume-once state, and outcome receipt remain distinct downstream objects. Evidence qualification is not permission to act.

## 2. Formal decision-value objective

Let:

- (s_t) be the current decision state, containing the DecisionNeed, the evidence already qualified for this invocation, unresolved contradictions, source/channel coverage, the remaining resource budget, and current lifecycle context;
- (a) be a typed retrieval action;
- (o_a) be the observation returned by that action;
- (E_t) be the currently qualified evidence set;
- (R(E)) be downstream decision regret under a frozen, independently controlled evaluator;
- (C(a)) be the all-in incremental cost of action (a);
- (H(a,s_t)) be a risk estimate for privacy exposure, scope violation, stale state, poisoning, or an inadmissible source.

The ideal value of sample information is:

[
\operatorname{EVSI}(a\mid s_t)
= \mathbb{E}[R(E_t)-R(E_t\cup Q(o_a))\mid s_t] - \lambda_C C(a) - \lambda_H H(a,s_t),
]

where (Q(o_a)) is the output after independent evidence qualification. An unqualified observation is not inserted into (E_t).

The live controller cannot inspect hidden gold answers or evaluator labels. It therefore uses a calibrated *prediction* of action value, not the evaluator's realized regret:

[
a_t = \arg\max_{a\in A_{\mathrm{safe}}(s_t)}
\operatorname{LCB}_{\alpha}[\widehat{\Delta R}(a\mid s_t)]
-\lambda_C C(a)-\lambda_H H(a,s_t).
]

The lower confidence bound makes uncertainty explicit. The controller stops when no safe action has positive risk-adjusted lower-bound value, when its budget is exhausted, or when the evidence-sufficiency rule is met. It abstains when the qualified evidence does not support a decision.

**Important distinction:** the estimator's prediction is not a benchmark result. Only a separate evaluator with sealed outcomes can measure whether a policy actually reduced regret.

## 3. Typed action space

The first real multi-action increment should add at least one genuinely different retrieval mechanism to BM25. The target portfolio is:

| Action family | Purpose | Required output discipline |
|---|---|---|
| BM25 / lexical | Exact terms and rare identifiers | Preserve source spans and offsets |
| Dense retrieval | Semantic paraphrase and related concepts | Pin embedding model and revision |
| Hybrid rank fusion | Combine lexical and dense candidate rankings | Use rank fusion or calibrated scores; never add incomparable raw scores |
| Entity / literal | Names, IDs, values, versions and exact strings | Preserve normalized and original forms |
| Temporal / state | Dates, event order, supersession and current state | Bind valid-time and observation/transaction-time |
| Causal / lineage graph | Parent dependencies, interventions and consequences | Return traversable, digest-bound edges |
| HDC / typed association | Structured relational and compositional recall | Report separately from conventional embedding retrieval |
| Stop / abstain | Avoid low-value or unsupported acquisition | Produce an explicit reason and budget snapshot |

The adapter contract must expose action identity, version, query-parameter digest, backend revision, candidate-set digest, source references, latency, resource accounting, and an outcome status. A strategy note must never contain arbitrary executable code or an unconstrained shell/network operation. It may select only from an allowlisted action vocabulary and validated parameters.

Channel scores are not naturally comparable. Begin with per-channel ranks and an explicit fusion rule (for example, reciprocal-rank fusion); introduce calibrated score fusion only after measuring calibration on a separate training split. Log channel-level candidates so that a fused result remains auditable.

## 4. Strategy-note contract

A strategy note is an advisory policy proposal, not an instruction with authority. It should contain only:

- stable note ID and immutable content digest;
- typed action sequence or conditional action template;
- DecisionNeed/task-family applicability scope;
- environment, corpus, backend, embedding and tool-profile digests;
- support-trace receipt digests (not raw labels, answer keys, or retrieved source payloads);
- estimated utility distribution and uncertainty, with estimator version;
- observed cost envelope (p50/p95/p99 and resource accounting);
- known failure and negative-transfer envelope;
- creation time, expiry, revocation/lifecycle state and parent policy digest.

At use, revalidate scope, tool and backend revisions, environment, policy version, expiry, revocation state and lifecycle snapshot. A mismatch means **do not suggest the note**. Do not silently fall back to a stale note.

The existing three-independent-successful-query promotion threshold may remain as a *minimum support threshold for an exploratory proposal*. It must not be treated as proof of positive causal utility or production eligibility.

Use three explicit statuses:

1. **Exploratory:** supported by trace observations; never allowed to outrank the safe default without an explicit experiment.
2. **Evaluation-eligible:** frozen and admitted to a preregistered, held-out counterfactual trial.
3. **Policy-eligible:** passes the held-out utility, cost, transfer and safety gates below. Still advisory and still subject to use-time applicability checks.

## 5. Label separation and counterfactual evaluation

The policy process must not read benchmark answer keys, gold answers, evaluator labels, hidden question IDs, or evaluator signing credentials. It may observe its own retrieval actions, source/tool outcomes, qualification dispositions, elapsed time, measured resource use and explicitly permitted operational feedback.

The independent evaluator may use sealed labels to calculate decision utility. It writes immutable evaluation receipts and aggregate metrics only after the candidate policy and run manifest are frozen. Do not feed per-question answer correctness back into the policy-memory process during the same evaluation.

### Paired experiment protocol

For each benchmark family:

1. Freeze the corpus, trajectories, query set, reader/model, prompt, tools, embedding and reranker revisions, environment, timeouts, seeds and total budget.
2. Split by session / trajectory / project / domain as appropriate. Do not split near-duplicate turns from one trajectory across support and held-out evaluation.
3. Run paired conditions on identical held-out tasks: no-note baseline, fixed BM25, fixed hybrid BM25+dense, random safe action, strategy-note controller, and (where implemented) EVSI-guided controller.
4. Keep model, reasoning tier, reader, evaluator and total resource budget identical. Randomize policy assignment/order where possible and pin deterministic replay inputs.
5. Include an abstain/no-additional-evidence arm. A controller must not receive credit for retrieving more material if decision regret increases.
6. Record action sequences, returned evidence digests, qualification dispositions, provenance closures, candidate coverage, latency, context tokens, storage, embedding/reranking cost, CPU/GPU time, network bytes and consolidation cost.
7. Evaluate per task family and operating point. Report paired absolute differences, confidence intervals, sample counts, missing runs and all exclusions. Use a preregistered paired bootstrap or randomization test; correct for multiple comparisons when promoting among many candidate policies.
8. Publish exact manifests and raw result receipts. A score without a reproducible source/data/model/evaluator/budget manifest is not a reproduced baseline.

For policy \(\pi\) and held-out task \(q\), define realized uplift:

[
\Delta_q = R_q(\pi_{\mathrm{baseline}})-R_q(\pi).
]

Report the mean paired uplift and its confidence interval, but also report accuracy, abstention, evidence sufficiency, per-family regret, p50/p95/p99 latency and all-in cost. Do not collapse independent benchmarks into a composite score.

### Counterfactual controls

Include at least:

- a no-note controller and a fixed-policy controller;
- a random policy over the same safe action set;
- query-shuffled and task-family-shuffled strategy notes;
- note removal and stale-profile controls;
- poisoned trace and repeated-query controls;
- a strategy that increases evidence volume but increases regret;
- a strategy that only helps one task family and harms another;
- a policy trained on support traces but evaluated on unseen sessions/domains;
- a baseline with the same total token, compute and latency budget.

For evidence-level attribution, use randomized channel / evidence suppression and rerun the frozen decision pipeline. Report this as an intervention on the evaluated pipeline, not as proof that a retrieved source is universally causal.

## 6. Promotion and stopping rules

A strategy becomes policy-eligible only when all of the following hold:

- positive paired held-out uplift against no-note, fixed-policy and random controls, with a preregistered confidence criterion;
- no pre-registered task family regresses beyond the allowed margin;
- matched-budget utility is maintained or improved;
- all-in cost and latency are within their declared operating envelope;
- the note survives domain, environment and tool-profile negative-transfer tests;
- all strategy use is traceable to a validated note digest and allowlisted action;
- no evidence qualification, provenance, privacy, lifecycle or authorization invariant is weakened.

Initial project targets (not measured results):

| Measure | Initial target |
|---|---:|
| Paired held-out accuracy uplift at matched budget | ≥2 percentage points, or a preregistered alternative utility improvement |
| Alternative efficiency gate | ≥20% lower p95 query latency with ≤1 pp accuracy loss |
| All-in cost reduction at matched utility | ≥25% |
| Worst task-family regression | ≤2 pp |
| Invalid / expired / mismatched strategy note used | 0 |
| Strategy action outside allowlist | 0 |
| Unqualified evidence surfaced as qualified | 0 |

A policy that fails the utility gate remains exploratory or is rejected. A policy that fails a safety gate is rejected regardless of retrieval accuracy.

## 7. Integration boundary with independent qualification

The intended path is:

```
LMEV2 sidecar
  -> typed action planner (advisory)
  -> allowlisted retrieval adapter(s)
  -> untrusted candidate batch
  -> independent qualifier (#13455)
  -> standard signed qualification receipt
  -> exact-batch projection (#13436)
  -> current-at-use binding / lifecycle validation
  -> decision-sufficiency controller
  -> separate decision and authorization protocols
```

The controller must not skip the qualifier to save latency. If the qualifier is unavailable, receipt validation fails, a candidate is Indeterminate, or dependency closure is incomplete, the system returns an auditable abstention or a separately tracked retry. It must never expose the raw candidate batch as if it were qualified.

Prefer a DSSE envelope carrying a versioned in-toto Statement predicate for the first interoperable receipt prototype. Fix the payload type, statement predicate type, canonical payload schema, signer identity and verification policy. Publish positive, negative, malformed, duplicate-field, unknown-key, revoked-key, multi-signature and replay vectors. If COSE is evaluated, explicitly choose the structure required by the signer model: COSE_Sign supports one or more signatures; COSE_Sign1 is single-signer and is not wire-compatible with COSE_Sign. The receipt attests scoped evidence qualification; it is not an action approval.

## 8. Benchmark portfolio

Keep official benchmark results separate:

| Benchmark | What it establishes | MACE evaluation requirement |
|---|---|---|
| Agent Memory Leaderboard | Track-specific textual, multimodal and coding memory utility | Implement the actual Add/Search protocol; compare only within matching track and division |
| LongMemEval-V2 | Experience acquisition and memory-query accuracy/latency frontier | Report official ΔLAFS against the frozen reference frontier and every operating point |
| AMA-Bench | Long-horizon agentic memory, causal inference and tool-augmented retrieval | Reproduce the strongest matched-backbone baseline; report each task family |
| MemoryArena | Interdependent multi-session Memory-Agent-Environment task performance | Measure downstream task completion and transfer across sessions, not dialogue recall alone |
| MACE decision-sufficiency lane | Whether qualified evidence changes decisions safely and economically | Independent evaluator, causal interventions, lifecycle/rights attacks and regret-cost frontier |

The official LongMemEval-V2 LAFS definition integrates the best reachable accuracy over log-scaled latency budgets from 1 to 200 seconds. Preserve the official reference frontier and benchmark protocol; do not compare a locally modified frontier to an official score.

Project ambition targets already proposed for LongMemEval-V2 are ΔLAFS ≥ +3.2 initially and ≥ +4.0 as a stretch. These remain targets, not achieved results. For AMA-Bench and MemoryArena, first reproduce exact matched settings; then preregister uplift targets against the strongest reproduced control rather than comparing unrelated paper numbers.

### Assurance metrics (separate from leaderboard utility)

- unauthorized / cross-scope evidence accepted: 0;
- invalid, replayed, expired, revoked or rebound qualification accepted: 0;
- cyclic, dangling or incomplete lineage accepted: 0;
- stale lifecycle evidence used at consequential time: 0;
- unsupported cases produce explicit abstention;
- harmful-memory influence removal ≥90%;
- benign-memory utility retention ≥95%;
- report selective-risk versus coverage, calibration, abstention utility and subgroup/task-family performance;
- report p50/p95/p99 latency and all-in CPU, memory, storage, network and verification cost.

A reject-everything controller fails the utility-retention and decision-quality gates. Security metrics cannot be traded against answer accuracy.

## 9. Implementation sequence — runner-independent work first

1. **Source audit:** locate the exact sidecar, trace store, strategy promotion and applicability functions; map the current public API before modifying it.
2. **Pure controller contract:** define typed action, observation, plan, policy note and use-receipt structures. Keep planning deterministic for fixed state and seed.
3. **Two real actions:** wire BM25 plus one independent channel (dense or entity/literal) behind explicit adapters. A second alias of BM25 does not count.
4. **Advisory integration:** call applicability and promotion in the sidecar; record plan digest, action receipt and policy-use decision. Keep BM25-only fallback explicit and versioned.
5. **Deterministic fake adapters:** test action ordering, budget stop, abstention, duplicate support traces, task-family overfitting, note expiry, backend/environment mismatch, poisoned proposals and restart/replay.
6. **Paired evaluation harness:** freeze manifests and implement paired runs, confidence intervals, task-family guardrails, cost accounting and sealed evaluator custody.
7. **Qualification composition:** integrate the independent qualifier and exact-batch projection; test timeouts, ambiguous outcomes, replays, corrections and current-at-use lifecycle checks.
8. **External reproductions:** implement raw-trajectory adapters and pinned reproductions for Resurface, Khipu and InsightMem. Keep AML official protocol runs separate from local causal-intervention experiments.

Because the authoring environment previously lacked the Rust toolchain, source authoring and static review can proceed independently, but compilation, formatting, Clippy, tests and integration are not claimed until executed on a pinned toolchain or exact-head CI. A skipped workflow is not a pass.

## 10. Decision record

The design deliberately rejects four shortcuts:

- **No promotion by repetition alone:** three successful traces support a hypothesis; held-out counterfactual uplift supports a policy.
- **No self-qualification:** retrieval policy cannot issue evidence authority.
- **No uncalibrated score soup:** different retrieval channels require rank fusion or calibrated combination.
- **No one-score benchmark story:** utility, latency, all-in cost, evidence integrity and lifecycle resilience remain independently reported.

The research claim MACE should attempt to establish is:

> Under matched task, reader, evaluator and all-in resource budgets, decision-value-guided retrieval improves held-out decision utility or the accuracy–latency frontier, while independently qualified evidence, explicit abstention and lifecycle controls preserve evidence integrity.

This claim is falsifiable. It is not established by the existence of the architecture or by authored tests.
