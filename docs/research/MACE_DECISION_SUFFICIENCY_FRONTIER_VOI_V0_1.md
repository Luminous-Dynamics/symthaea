# MACE decision-sufficiency frontier and active evidence acquisition v0.1

**Status:** research design; proposed protocol and engineering sequence, not an executed benchmark.  
**Created:** 2026-11-22  
**Tracking:** [#13284](https://github.com/Luminous-Dynamics/symthaea/issues/13284)  
**Starting point:** [PR #13242](https://github.com/Luminous-Dynamics/symthaea/pull/13242), head `a763d73e40f118fcee02d466f3c3b7aea20295c4`  
**Related:** #13221, #13241, #13242, #13261, #13264, #13265

## 1. Proposed contribution

MACE should treat memory as a **decision-sufficient evidence representation**, rather than as a ranked collection of semantically relevant records.

For a particular decision need, the system should find a compact representation that preserves decision quality, satisfies hard evidence and authority constraints, and exposes exactly where compression becomes unsafe. When evidence is missing, it should estimate whether acquiring a new observation is worth its cost. When neither existing nor obtainable evidence is sufficient, it should abstain.

The target capability is therefore:

> Given a decision, its admissible evidence universe, a fixed policy, and an explicit resource budget, identify the least-cost evidence representation that remains decision-sufficient; show the dependencies that make it sufficient; and decide whether to compress, acquire more evidence, or abstain.

This is distinct from claiming universal memory sufficiency. A representation can be sufficient for one decision, policy, loss function, scope, and time horizon while being insufficient for another.

The intended research outputs are:

1. A cost–regret Pareto frontier, not just accuracy at one top-k.
2. A verified, transitive dependency closure for each decision capsule.
3. A counterfactual account of necessity, complementarity, redundancy, and harmful evidence.
4. An evidence-acquisition policy evaluated by expected decision improvement after cost.
5. A proof-carrying decision capsule that makes support, uncertainty, lifecycle, and abstention replayable.
6. An external benchmark suite that reproduces strong methods under their own protocols.

## 2. Formal definition

Let:

- n be a candidate-visible DecisionNeed, including scope, objective, allowed actions, estimand, deadline, and policy version.
- E(n) be the full candidate-visible evidence universe after exact scope and authority partitioning.
- S be a retained subset or a compressed, derived representation of that evidence.
- pi(n, S) be the frozen decision policy applied to the evidence resolved from (S).
- y be evaluator-only world truth.
- U_y(a) be the declared utility of action a in world y.

Define absolute regret:

```text
R_y(S) = max over a in A(n) of U_y(a) - U_y(pi(n, S)).
```

Define compression-induced regret relative to the full candidate-visible evidence reference:

```text
G_y(S) = R_y(S) - R_y(E(n)).
```

The second quantity isolates degradation caused by compression relative to the declared reference. It must not replace absolute regret: the full-evidence policy may itself make a poor decision, and the candidate-visible evidence universe may be incomplete with respect to the real world.

A representation is epsilon-decision-sufficient only when both of the following hold:

1. Every hard admission, authority, scope, provenance, independence, and lifecycle constraint passes.
2. The preregistered regret criterion is met, for example `G_y(S) <= epsilon`, together with an absolute-regret ceiling.

For a population of tasks, report at least:

- Mean paired absolute regret and mean paired compression-regret gap.
- Probability that the regret gap exceeds (epsilon).
- A tail-risk statistic such as CVaR at a preregistered level.
- Worst task-family and regime outcomes.
- Correct and incorrect abstention, unsupported action, and hard-gate failure rates.

The benchmark's current synthetic zero-utility abstention convention is only a fixture convention. A real task must declare the value or cost of abstention, including delay, missed opportunity, and escalation cost where applicable.

### 2.1 Why the full-evidence reference is not enough

Decision agreement with a full-evidence decision is not a correctness metric: both systems can make the same wrong choice. Conversely, a compressed system may differ from the full-evidence decision and be better because the full pool contains misleading or superseded evidence.

Every result must therefore report:
- agreement with the full-evidence decision;
- absolute regret against evaluator-only optimal utility;
- compression-regret gap;
- no-memory comparison;
- hard evidence qualification;
- abstention behavior.

Evidence utility is not assumed to be monotonic in set size. More records can increase utility, do nothing, or make a decision worse.

### 2.2 Decision sufficiency is not universal statistical sufficiency

The intended property is conditional on a declared decision problem. If Q_E(a) is expected utility for action a using all evidence and Q_S(a) is the corresponding value using a representation, then a useful sufficient condition is:

```text
max over a in A(n) of abs(Q_E(a) - Q_S(a)) <= delta.
```

If both policies choose the utility-maximizing action under their own value estimates, this uniform value error bounds the full-evidence regret of the compressed choice by (2delta). This is a theoretical guide for future probabilistic policies, not a property established by the current synthetic runner.

## 3. Optimize a Pareto frontier, not one arbitrary score

The naive formulation is:

```text
minimize C(S) over S subseteq E(n), subject to G_y(S) <= epsilon.
```

In practice, C(S) is not one natural scalar. Keep an explicit cost vector:

```text
C(S) = (memory_bytes, sidecar_bytes, context_tokens, index_time, update_time, verification_time, p50_latency, p95_latency, p99_latency, model_calls, monetary_cost, energy).
```

Include source receipts, lifecycle history, tombstones, compaction, canonical evidence revalidation, and background consolidation. A vector index that appears small but depends on a large unreported sidecar is not a low-cost memory.

Construct the nondominated frontier across:
- total retained bytes;
- context tokens and model calls;
- query and end-to-end latency;
- verification/replay cost;
- absolute regret and compression-regret gap;
- decision coverage and abstention.

Only collapse costs into a scalar when the deployment declares the exchange rates in advance. Report the unscalarized vector as well.

### 3.1 Selection is not always greedy

Do not assume that evidence utility is submodular or monotone. Multi-parent support creates complementarity; duplicate sources create redundancy; stale or wrong-scope records create interference. A greedy top-k procedure may therefore miss essential coalitions or retain harmful evidence.

Use a staged evaluation:
- Exact subset enumeration on small fixtures as a correctness oracle.
- Deterministic coalition sampling on larger fixtures, with the seed and Monte Carlo error recorded.
- Bounded search or integer programming for selected tractable cases.
- Learned selectors trained only on training/calibration clusters.
- Locked policy selection before held-out cluster evaluation.

Use leave-one-dependency-group-out effects and second-order interactions as descriptive intervention statistics. Shapley estimates may summarize allocation over sampled coalitions, but must not be mislabeled as causal effects absent the appropriate randomized interventions.

## 4. Verified transitive evidence dependencies

The current #13242 implementation is a useful direct-root/direct-family synthetic audit. It is not yet transitive lineage verification or source authentication.

The next graph should contain canonical evidence nodes and derivation edges. Each node should bind:
- canonical evidence ID and payload digest;
- one or more exact parent IDs and parent digests;
- transformation identity, version, and output digest;
- source-root and source-family ancestry;
- source receipt and verifier profile;
- authority/rights and scope/estimand;
- event time and knowledge/transaction time;
- lifecycle state and references to corrections, supersession, retraction, or revocation.

A digest proves equality to a byte sequence, not the identity or authority of its publisher. Root identity must be resolved against trusted source receipts and issuer/key profiles. A signature can establish integrity and origin under a trust policy; it does not establish that the signed claim is true.

### 4.1 Closure semantics

If (C) is derived from parents (A) and (B), and (B) is derived from (D), then the verified closure of (C) includes (A), (B), and (D). Knocking out (D) invalidates (B) and (C) unless an independently verified alternative derivation reconstructs them.

Required fail-closed cases:
- duplicate or aliased canonical IDs;
- dangling parent references;
- cycles;
- conflicting payloads for one canonical ID;
- untrusted or ambiguous source roots/families;
- a derived view whose declared inputs do not match its verified receipts;
- a retraction/correction that is known at consumption time but absent from a cached view;
- a parent knockout that leaves an unsupported child visible;
- a source-family alias that falsely inflates independent support.

Unknown ancestry contributes zero independent-root/family support. The graph must distinguish historical validity (what was true/knowable at a past time) from current validity (what may be relied on now). Replaying a historical decision must not rewrite what the agent knew then; using evidence today must not resurrect a fact that has since been retracted.

This work belongs with #13264 and should precede any claim that counterfactual knockouts cover real derived memories.

## 5. Active evidence acquisition

Retrieval asks, “Which stored item should I fetch?” Active acquisition asks, “Which new observation should I obtain, and is it worth obtaining?”

For a candidate query or observation q, define expected value of sample information (EVSI) in decision-risk units:

```text
EVSI(q) = R(b) - E_{o ~ P(o | b, q)}[R(b | o)].
```

Here b is the candidate-visible belief state, o is a possible observation, and R(b) is minimum expected decision risk under the declared policy and action set.

Net value must account for acquisition, delay, privacy, safety, and operational costs. If those costs are not legitimately convertible to the same utility unit, report a vector of expected risk reduction and costs rather than inventing an exchange rate.

The planner may use only:
- the current candidate-visible evidence and uncertainty;
- declared source availability and trust profiles;
- the query/observation model;
- declared acquisition costs and deadlines.

It must not inspect evaluator truth, future observations, hidden scenario utilities, intervention labels, or test-set answer keys.

Compare these acquisition policies:
1. EVSI-guided query selection.
2. Random query selection.
3. Highest uncertainty.
4. Missing-dependency / evidence-gap first.
5. Retrieve more from existing memory.
6. Immediate abstention/escalation.
7. Evaluator-only oracle query selection as an upper bound, never as a deployable baseline.

Required adversarial fixtures include:
- the cheapest query is uninformative;
- the most uncertain query does not change the best action;
- a second independent source resolves a common-cause ambiguity;
- a source query returns a retraction rather than confirmation;
- query latency makes the information arrive too late;
- abstention is better than acquisition;
- an apparently useful source is not authorized for the decision.

The key endpoint is realized regret reduction after all acquisition costs, not the number of queries issued or documents retrieved.

## 6. Proof-carrying decision capsule

A capsule is a compact, replayable statement of why a decision was or was not justified. It should bind:

- DecisionNeed digest and policy/version digest;
- candidate evidence-set digest and canonical evidence references;
- content digests and verified transitive dependency closure;
- source receipts, issuer/key profile, authority and rights checks;
- scope, objective, estimand, horizon, and deadline;
- event-time and knowledge-time validity;
- transformation and derivation records;
- unresolved ancestry, missing evidence, and uncertainty;
- retrieval/acquisition actions and their costs;
- selected action or abstention and the policy's reason;
- evaluator-independent replay and verification profile.

A capsule is not a proof of real-world truth. It is a proof-carrying provenance and qualification object: it makes the claimed support, its lineage, the verifier used, and the limits of the decision inspectable.

The system must retain a route to canonical evidence or explicitly report when that evidence is no longer available. Do not count a hash pointer as retained evidence if the underlying source cannot be re-observed or replayed.

## 7. External comparison: separate protocols, strong controls

MACE should reproduce the method, not copy a headline score or compare unlike metrics.

| Lane | What it tests | Required comparison |
|---|---|---|
| AML Cycle 2 | Add/Search on Textual, Coding, Multimodal | Official protocol; Resurface, InsightMem, Khipu; distinct track/division |
| LongMemEval-V2 | Context gathering for a fixed reader; accuracy and latency | Insert/Query/Reader protocol; AgentRunbook-R/C/V2; exact model and reasoning tier |
| AMA-Bench | Long-horizon agent trajectories and causal questions | AMA-Agent causal graph and tool-augmented retrieval |
| WorldMemArena | Multimodal observe/write/maintain/retrieve/use in action-world loops | Stage-level diagnosis and original task protocol |
| StateMemBench | Current state, supersession, relational dependencies | State-first memory and same-backbone/length-matched controls |
| FAMA / ACL Memora benchmark | Forgetting and invalidated-memory reuse | Current-vs-obsolete memory scoring and original benchmark protocol |
| TimlMemBench | Revocation, restriction, relevance, governed activation | Canonical-pool and original governance controls |
| LoCoMo | Long dialogue, abstraction, cue anchors, multi-hop questions | Microsoft's Memora representation and other pinned baselines |

Naming caution: Microsoft's **Memora** is an ICML 2026 memory-representation method based on primary abstractions and cue anchors. The ACL 2026 **Memora** is a long-term memory benchmark that introduces FAMA. They are different works and must not be conflated.

The official AML Cycle 2 open-method anchors published 2026-11-19 are:
- Textual: Resurface, 61.22.
- Coding resolved: InsightMem, 63.33%.
- Multimodal: Khipu, 43.26.

Commercial MemoraX is a separate division: 69.26 textual, 68.00% coding resolved, and 53.68 multimodal. Proposed next-cycle open-method ambitions of +5 points are 66.22, 68.33%, and 48.26. They are targets, not MACE results.

LongMemEval-V2 must be reported as an accuracy–latency frontier. The published AgentRunbook-C V2 research update reports, on the small split with GPT-5.4-mini, 65.41% at 26.11 seconds in its low-reasoning setting and 75.61% at 130.54 seconds in its xhigh setting. Do not merge these points with results from another model, reasoning tier, split, or controller version. A fair claim is a held-out frontier improvement under a pinned protocol, not “higher accuracy” from a different compute regime.

All baseline manifests must pin code commit, data/split checksum, model and embedding revision, reranker, tokenizer, prompt, reader/judge, API semantics, top-k, fusion weights, context packing, hardware/container, resource limits, concurrency, and query-time accounting. The MACE synthetic Experience records are not a substitute for raw trajectory/session adapters.

## 8. Proposed MACE qualification targets

These are research targets, not achieved results:

| Capability | Proposed gate |
|---|---|
| Known retracted/revoked/wrong-scope support used | 0 |
| Deterministic lineage/alias/correction/common-cause fixtures | 100% expected rejection or propagation |
| Unknown/unverified ancestry counted as independent | 0 |
| Compression-induced task-success loss at selected operating point | ≤1 percentage point, with preregistered margin and interval |
| Worst-family regression | ≤2 percentage points |
| All-in resource cost reduction at matched task quality | ≥25%, including sidecars and verification |
| EVSI over random and uncertainty-only acquisition | Positive held-out paired effect after costs |
| Selective behavior | Report coverage-risk, correct/incorrect abstention, and calibration |
| External benchmark performance | +5 open-method points over Cycle 2 leaders on the exact official tracks, pursued independently |

The operating point must be selected on training/calibration data and then locked. Do not tune the epsilon, top-k, acquisition policy, or abstention threshold on held-out outcomes.

## 9. Preregistered statistical design

Before evaluation:
1. Name one primary endpoint and one primary contrast.
2. Fix the (epsilon) regret margin and absolute-regret ceiling.
3. Fix the eligible task families, worlds, source clusters, model, policy, and exclusions.
4. Freeze intervention manifests without using evaluator outcomes.
5. Lock all selectors, prompt/model settings, budgets, and acquisition policies.
6. Calculate power using independent world/source clusters, not the number of intervention rows.

Analysis:
- Keep paired task/world blocks intact.
- Aggregate within task blocks and world clusters before resampling.
- Use paired cluster bootstrap or cluster-level randomization inference.
- If model-fit seeds are nested, resample seeds first and clusters within seed second.
- Report point estimates, intervals, cluster counts, denominators, attrition, missing blocks, and worst-family outcomes.
- Control multiplicity for confirmatory intervention families; label per-root/per-family exploratory analyses.
- Preserve evaluator custody independently of candidate code and candidate memory.

A bootstrap interval over a small number of named clusters does not prove that those clusters are independent. Report the sampling frame and justify the cluster definition; perform sensitivity analyses where common-source structure may cross nominal world boundaries.

## 10. Engineering sequence and acceptance criteria

### Step 1 — Lineage substrate (#13264)
Implement canonical evidence IDs, signed source receipts under explicit trust profiles, transitive multi-parent closure, bitemporal lifecycle, and correction-at-use. Add metamorphic/adversarial fixtures for aliases, cycles, dangling edges, root/family convergence, multi-parent derivation, retractions, and historical replay.

**Acceptance:** every tested knockout and lifecycle mutation has the expected transitive effect; unknown ancestry never counts as independent; no unverified source label upgrades authority.

### Step 2 — Frozen frontier contract (this issue)
Define versioned schemas for cost vectors, regret summaries, frontier points, selector manifests, and preregistration metadata. Keep candidate and evaluator streams separate.

**Acceptance:** a summary can be recomputed from receipts; each point binds candidate code, policy, data, selector, and budget digests; the evaluator can prove which clusters and rows were included.

### Step 3 — Raw-trajectory baseline adapters (#13261)
Implement Add/Search-compatible adapters for raw sessions and agent trajectories. Reproduce at least lexical BM25, dense, RRF hybrid, and neighboring-context retrieval before attempting larger system reproductions.

**Acceptance:** each adapter preserves original text, timestamps, session/speaker identifiers and source references; baseline outputs can be compared under pinned reader/context/latency budgets.

### Step 4 — Frontier runner
Sweep deterministic budgets and produce cost–regret frontiers for exact reference, no-memory, recency, BM25, dense, hybrid, causal graph, state-first, hierarchical abstraction/cue anchors, typed HDC, and event-ID HDC.

**Acceptance:** exact subset fixtures match an exhaustive oracle; non-monotone evidence cases are retained; no held-out label reaches a selector; sidecars and verification work appear in the reported cost.

### Step 5 — Active acquisition runner
Implement query candidates, observation models, explicit cost/delay/permission checks, and EVSI estimates.

**Acceptance:** evaluator can independently replay candidate query choices; adversarial cases above pass; EVSI beats random/uncertainty on held-out clusters after cost, or the system reports no demonstrated acquisition advantage.

### Step 6 — Independent qualification
Run a sealed held-out evaluation with preregistered statistics and independent evaluator custody.

**Acceptance:** exact-head source and data digests match; Rust tests and executable runners pass; baseline protocol and statistical report are reproducible; all negative controls are preserved. Skipped CI, authored tests, or source-level inspection are not passes.

## 11. Claim boundary and risks

The present #13242 branch is a synthetic, diagnostic-only foundation. Its 5 retrieval methods, shared intervention schedule, candidate/evaluator stream split, and initial cluster reducer are valuable infrastructure. Its direct root/family grouping does not authenticate identity or establish transitive lineage; the benchmark has no external relevance labels; the exact scan is not a same-cost competitor; and the statistical reducer has not been executed or independently qualified.

Do not claim that MACE already beats SOTA, has learned event-driven temporal dynamics, proves real-world causality, guarantees truthful sources, or is production-ready.

The contribution becomes compelling only after MACE demonstrates, on independently held-out evidence and strong reproduced baselines, that it can reduce all-in cost while preserving decisions, expose fragile support, identify valuable missing observations, and refuse to act when verification fails.

## 12. Primary references

- [AML Cycle 2 official results](https://agentmemoryleaderboard.ai/results/cycle-2/) — published 2026-11-19.
- [AML Cycle 2 rules](https://agentmemories.ai/rules).
- [Microsoft Research: Memora — a harmonic memory representation](https://www.microsoft.com/en-us/research/publication/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/), ICML 2026.
- [AMA-Bench / AMA-Agent](https://proceedings.mlr.press/v306/zhao26bs.html), ICML 2026.
- [LongMemEval-V2](https://xiaowu0162.github.io/longmemeval-v2/).
- [AgentRunbook-C V2 research update](https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/).
- [WorldMemArena](https://arxiv.org/abs/2605.29341).
- [StateMemBench](https://arxiv.org/abs/2608.19652).
- [FAMA / From Recall to Forgetting](https://aclanthology.org/2026.findings-acl.1337/).
- [TimlMemBench](https://aclanthology.org/2026.findings-emnlp.507/).
- [MACE external baseline reproduction issue #13261](https://github.com/Luminous-Dynamics/symthaea/issues/13261).
- [MACE transitive lineage issue #13264](https://github.com/Luminous-Dynamics/symthaea/issues/13264).
- [MACE statistical qualification issue #13265](https://github.com/Luminous-Dynamics/symthaea/issues/13265).
