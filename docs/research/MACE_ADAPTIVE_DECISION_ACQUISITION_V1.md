# MACE Adaptive Decision Acquisition V1

**Snapshot:** 2026-11-26  
**Status:** Research contract and benchmark proposal. This branch adds a machine-readable policy schema and an evaluation protocol; it does not implement a sequential policy executor or claim measured gains.

## 1. The next paradigm shift

Decision-value retrieval V1 evaluates one candidate acquisition operation against a consumer-owned decision. Real evidence work is often adaptive:

1. ask a question of a source or tool;
2. inspect the observation and its provenance;
3. decide whether the observation resolved the uncertainty;
4. choose a different query, an independent source, or stop;
5. select an action only after the required evidence obligations are satisfied.

A fixed top-k retriever cannot express this process. Nor can a one-shot EVSI score capture the value of a second query whose relevance depends on the first result.

MACE's next target is an **adaptive evidence-acquisition policy**: a bounded policy DAG whose next query is conditioned on the complete verified observation history, current source-dependency graph, DecisionNeed and remaining resource budget. Retrieval becomes a sequence of explicit information-acquisition decisions, not a bag of passages.

This is distinct from an autonomous agent that freely calls tools. The policy is consumer-scoped, bounded, replayable and subject to independent receipt verification. It recommends queries and terminal decisions; it does not authorize external action.

## 2. Formal model

Let:

- (S) be the consumer-defined world-state set;
- (H_t) be the verified history at step (t), including query IDs, observation receipts, source/derivation digests, root-family dependencies, valid time and observed time;
- (q_t) be an eligible evidence-acquisition operation;
- (o_t) be its observation;
- (a_T) be the terminal decision;
- (c(q_t)) be its all-in acquisition cost.

An adaptive policy maps a history to either another query or a terminal action:

[
pi(H_t) in {q_t,; operatorname{terminal}(a_t)}.
]

For calibrated expected-value evaluation, the next observation model is conditional on the entire verified history:

[
P(o_t mid s, q_t, H_t),
]

not merely (P(o_tmid s,q_t)). The joint likelihood must represent dependence between queries, shared source roots, common upstream datasets, correlated sensors, mirror/archive paths, and prior observations. Multiplying marginal likelihoods while pretending correlated sources are independent is a benchmark failure.

The policy's net expected utility is:

[
V(pi)=
mathbb{E}[U(a_T,S)]
-mathbb{E}left[sum_{t<T}c(q_t)ight]
-C_{	ext{risk}}(pi),
]

where (C_{	ext{risk}}) includes separately declared privacy, rights, safety and operational costs. Do not combine unlike costs without a consumer-owned conversion rule. Report the components separately as well as any justified utility conversion.

The relevant decision is the improvement over a no-additional-retrieval policy under the same DecisionNeed, action constraints and resource units. EVSI is meaningful only when every conditional observation model and utility profile is independently calibrated and receipt-bound.

### Probability-free mode

When probabilities are unavailable or not defensible, the policy must not manufacture them. Let (Omega(s,pi)) be the conservative set of observation histories compatible with state (s) and policy (pi). A terminal action selected after history (h) must be permitted for every state still compatible with (h). A robust policy is judged against its worst compatible outcome:

[
operatorname{WorstRegret}(pi)=
max_{sin S,,hinOmega(s,pi)}
left[
U^*(s)-U(a_pi(h),s)+C_pi(h)
ight].
]

Compare this policy against the strongest no-query and fixed-policy baselines using the same utility, action constraints, support sets and total budgets. This is a robust decision criterion, not an approximate Bayesian score. Keep its results separate from EVSI.

## 3. Why the history-conditioned model matters

Three common failures make naive sequential retrieval look better than it is:

**Duplicate-source multiplication.** A query returns a report and the next query returns three summaries of the same report. Marginally each may appear informative; jointly they may add almost no independent evidence. The history model must carry source-root and common-cause lineage. New evidence can add detail without adding independent corroboration.

**Contradiction collapse.** A second source disagrees with the first. A policy must preserve both observations, identify the disagreement class and either seek a discriminating source or abstain. It must not average conflicting publishers into a false consensus.

**Stale branch reuse.** An earlier branch was selected because a memory was current. A correction, quarantine, revocation or rights event arrives before the next query or final action. The current event sequence and projection-invalidation receipt must be rechecked before continuing. A cached plan cannot outlive its evidence dependencies.

The bitemporal ledger and derived-projection invalidation work provide the basis for these lifecycle checks. The adaptive policy adds branching over the changing qualified evidence state.

## 4. Policy contract and schema

The new JSON Schema is:

`data/schemas/mace_adaptive_decision_policy_v1.schema.json`

It defines:
- a consumer-owned DecisionNeed and utility-profile digest;
- an evidence-dependency-graph digest;
- separate EVSI and probability-free robust modes;
- a bounded entry node, horizon, query count, policy-node count and budgets;
- query nodes with action-template and resource-receipt commitments;
- history-conditioned model commitments;
- observation branches and an explicit fallback edge;
- terminal actions, including abstain and defer.

JSON Schema validates field shape, primitive bounds and mode-specific fields. It cannot prove graph reachability, acyclicity, unique node IDs, branch completeness, horizon compliance, query-budget compliance, safe terminal-action admissibility, or cryptographic validity. A separate fail-closed graph validator must enforce those properties.

Two illustrative policy fixtures are published:

- [EVSI adaptive policy example](../../data/fixtures/mace_adaptive_decision_policy_evsi_v1.example.json)
- [Probability-free robust adaptive policy example](../../data/fixtures/mace_adaptive_decision_policy_robust_v1.example.json)

Both demonstrate a two-query path: first verify a primary requirement, then seek an independent cross-check only when the first observation warrants it. Contradiction or denial leads to abstention; unknown/unavailable results defer. The SHA-256-shaped values are deterministic placeholders for schema examples only. They are not valid signatures, trust profiles, calibrated models, independently issued resource receipts or authority to execute the policy.


Required graph invariants:
1. exactly one entry node exists and every referenced node exists;
2. every node is reachable from the entry and the graph is acyclic;
3. every path terminates within the declared horizon;
4. every query node has explicit branches for observed outcomes and a fallback for unknown, unavailable, timed-out or malformed results;
5. no path exceeds total query, tool-call, latency, token, byte or acquisition-cost budgets;
6. all terminal actions are permitted under every state still possible on that path;
7. every observation edge is bound to a verified receipt, exact source/evidence lineage and the history digest used by its conditional model;
8. a correction, quarantine, revocation or rights-erasure event invalidates any branch whose dependencies are no longer current;
9. the executor cannot reinterpret a failed, timed-out or unknown query as a successful empty observation;
10. execution is idempotent by policy, run, node, query and observation identity.

## 5. Adaptive acquisition lifecycle

A qualified execution should produce one receipt per transition:

- **Policy admission:** exact policy digest, DecisionNeed, utility profile, budgets, trust profiles and policy-validator version.
- **Query authorization:** selected node, action-template digest, remaining budget and current evidence-history digest.
- **Resource attestation:** independently measured tool count, latency, context tokens, inspected bytes and cost; signed expiry included in the verifier binding.
- **Observation attestation:** raw observation bytes, source identity, acquisition time, valid time, source revision, rights/use scope, source-origin receipt and derivation graph.
- **History transition:** deterministic before/after state digests, dependency-root changes, invalidated projections and next node.
- **Terminal decision:** selected action or abstention, complete evidence-history digest, unresolved obligations and cost/risk summary.

An unknown execution result blocks a new mutation until reconciliation. An unavailable verifier or expired receipt does not become a positive observation. A terminal action is still subject to its own independent authorization boundary.

## 6. Benchmark design

The benchmark should be a new MACE track, not a replacement for external memory benchmarks.

### Baselines

Run the exact same tasks and budgets against:
1. no additional retrieval;
2. exact-match and BM25;
3. dense retrieval;
4. BM25 + dense reciprocal-rank fusion;
5. graph/causal retrieval;
6. fixed top-k;
7. full context at matched total tokens;
8. one-shot entropy/information-gain selection;
9. single-query EVSI and robust minimax from Decision-Value Retrieval V1;
10. adaptive EVSI;
11. adaptive robust minimax regret;
12. an oracle adaptive policy using hidden synthetic ground truth, reported only as an upper bound.

All baselines must share the same action space, utility profile, source corpus, evidence rights, model, token budget, latency budget and terminal safety gate. Do not compare a high-budget adaptive policy against a low-budget fixed baseline.

### Task families

- **Conditional follow-up:** the first observation determines which second query has value.
- **Correlated source fan-out:** query results share hidden root sources, mirrors, datasets, models or sensor dependencies.
- **Independent corroboration:** distinguish a genuinely independent acquisition path from another representation of the same source.
- **Contradictory evidence:** preserve conflicting claims and choose between discriminating acquisition and abstention.
- **Unknown and unavailable outcomes:** timeouts, malformed results and inaccessible sources follow explicit fallback branches.
- **State drift:** user preferences, object states, project facts or operational conditions change between acquisition steps.
- **Lifecycle invalidation:** correction, quarantine, revalidation, revocation and rights erasure arrive between query and terminal decision.
- **Resource-constrained branching:** query choices trade latency, bytes, tokens, tool calls, cost and risk.
- **Adversarial branch injection:** replay old receipts, alter a branch target, rebind a policy digest, substitute a source revision or exploit a stale projection.
- **Stopping:** all candidate continuations have non-positive net value, or all remaining paths exceed resource budgets.
- **Safe abstention:** every feasible query leaves at least one state in which the terminal action is prohibited.
- **Long-horizon agentic tasks:** memory acquired in one session changes later environment actions and outcomes.

### Required metrics

| Metric | What it measures |
|---|---|
| Paired terminal decision regret | Utility loss against the task's declared decision oracle |
| Net utility after acquisition | Decision benefit minus all-in retrieval cost |
| Worst-case regret | Robust outcome under declared support sets |
| Unsafe action rate vs supported coverage | Selective risk, including abstentions |
| Correlated-evidence overcount | Cases where duplicate roots falsely increase support/value |
| Contradiction preservation | Whether disagreement remains explicit through the whole policy |
| Stale-branch use | A path uses evidence invalidated before terminal selection |
| Budget compliance | Any path exceeding query, tool, latency, token, byte or cost limits |
| Stop efficiency | Whether the policy stops when all eligible marginal values are non-positive |
| Receipt and replay integrity | Correct rejection of altered history, branch, model, source or receipt |
| False invalidation | Valid unrelated evidence lost through over-broad invalidation |
| Calibration / shift | Conditional calibration, coverage and performance under held-out shifts |

### Proposed targets (not achieved results)

- At least **10% lower paired terminal regret** than the strongest matched non-adaptive baseline, with the 95% cluster-adjusted confidence interval excluding zero improvement.
- At least **15% lower all-in acquisition cost** at non-inferior regret, or a strictly better regret-cost Pareto frontier.
- **Zero** accepted correlated-root double counts in exhaustive synthetic controls.
- **Zero** stale-branch terminal decisions in deterministic lifecycle fixtures.
- **Zero** hard action-constraint violations in exhaustive fixtures.
- **100%** deterministic policy-graph and receipt replay on frozen fixtures.
- **≤1% unsafe terminal actions at ≥80% supported coverage** on the predeclared in-scope test population, with all task families reported.
- Nominal **95%** calibration intervals must attain declared coverage overall and on critical subgroups; report sample counts and intervals.
- No strategy promotion from predicted value alone; promotion still requires independent outcomes and held-out counterfactual qualification.

These thresholds are ambitious research gates, not current performance. A result is not SOTA-leading unless the same official harness or a clearly matched, independently reproduced protocol supports that statement.

## 7. External benchmark alignment

Run external suites unchanged as complementary evaluation:
- [Agent Memory Leaderboard Cycle 2](https://agentmemoryleaderboard.ai/results/cycle-2/) for textual, coding and multimodal parity. The official page reports separate tracks and divisions; do not average them.
- [AMA-Bench](https://proceedings.mlr.press/v306/zhao26bs.html) for recall, causal inference, state updating and state abstraction over agent-environment trajectories.
- [MemoryArena](https://proceedings.mlr.press/v306/he26am.html) for interdependent multi-session Memory-Agent-Environment loops.
- [Memora / FAMA](https://aclanthology.org/2026.findings-acl.1337/) for long-horizon remembering, reasoning, recommending and obsolete-memory penalties.
- [UtilMem](https://aclanthology.org/2026.emnlp-main.1639/) for integration of distributed implicit/noisy evidence and resilience to similar distractors.
- [MemUse](https://aclanthology.org/2026.emnlp-main.804/) for naturally integrating relevant memory into conversational responses rather than answering explicit memory questions only.
- [COLLABMEM](https://aclanthology.org/2026.emnlp-main.1194/) for role-aware sharing, access control, routing, provenance and conflict resolution across collaborating agents.
- [TimlMemBench](https://aclanthology.org/2026.findings-emnlp.507/) for governed activation, revocation, supersession, scope and relevance; use both canonical-pool and raw-native evaluation.
- [StreamMemBench](https://aclanthology.org/2026.findings-emnlp.992/) for streaming evidence use, feedback incorporation and follow-up reuse.
- [DeepControl](https://aclanthology.org/2026.findings-emnlp.52/) as an algorithmic comparator for adaptive retrieval continuation and granularity control. Its reported +9.4/+8.6 average-point gains over Search-R1 on Qwen2.5-7B/3B are paper-specific, not directly comparable MACE results.
- [MEMORA-Bench](https://aclanthology.org/2026.findings-emnlp.1211/) for embodied memory and prospective planning from egocentric video.
- [MEMOBench](https://aclanthology.org/2026.findings-emnlp.333/) for process-level storage, update and compression during robotic manipulation.

### AML parity ladder for MACE (targets, not results)

The latest official Cycle 2 results published on November 19, 2026 report these track leaders:

| Track | Open-source leader | Commercial leader | Stage 1: open-source leader +5 points | Stretch: commercial leader +1 point |
|---|---:|---:|---:|---:|
| Textual | Resurface 61.22 | MemoraX 69.26 | ≥66.22 | ≥70.26 |
| Multimodal | Khipu 43.26 | MemoraX 53.68 | ≥48.26 | ≥54.68 |
| Coding | InsightMem 63.33% | MemoraX 68.00% | ≥68.33% | ≥69.00% |

For Coding, the margins are percentage points. Keep each track's official denominator, grader, model, API limits, and cost reporting unchanged. The stretch goal is deliberately beyond the current commercial leader, but must be reached in a fresh official evaluation rather than inferred from the paper's internal benchmark.

For the textual capability sub-scores, MACE should first target parity with the commercial leader on Streaming (63.11), Temporal & Event Sequencing (67.05), Personalization & Care (54.70), and Epistemic Safety & Privacy (81.98). A stretch gate is at least +1 point over each of those values under the official harness. Also track governance, explicit fact recall, relational reasoning and in-context workflow execution; improving one capability must not conceal regression in another.

MACE has no published AML score yet. These are a staged target ladder, not a claim of parity or superiority. If the official Cycle 3 schedule is not yet published, use the public Cycle 2 harness for reproducible development runs and wait for an official announcement before promising a Cycle 3 submission.

These suites measure different capabilities. Publish one scorecard per official benchmark and a separate adaptive-decision-assurance scorecard. Do not average their scores or claim that a high recall score establishes decision quality, provenance integrity or action safety.

## 8. Research hypotheses and ablations

The key falsifiable hypotheses are:

- H1: adaptive acquisition improves decision utility over one-shot retrieval when the value of a query depends on a previous observation.
- H2: explicit source-dependency conditioning prevents overconfidence and improves decisions when sources share hidden roots.
- H3: receipt-bound state transitions prevent stale evidence from being used after lifecycle changes.
- H4: robust support-set policies remain safer than uncalibrated probabilistic policies under unknown or shifted distributions, at a measurable coverage/cost trade-off.
- H5: an explicit stop option reduces unnecessary retrieval without increasing terminal regret.

Ablate the history digest, source-root graph, conditional model, resource attestation, fallback branch, invalidation event, safe terminal action and stop option one at a time. Report both quality gains and valid-evidence loss.

## 9. Execution plan

1. Freeze policy DAG schema and independent structural validator.
2. Build a tiny exact synthetic environment with 2-8 states, 2-6 actions, 2-5 queries and bounded horizons; verify it with an independent exhaustive enumerator.
3. Add conditional observations and shared source-root latent variables.
4. Add correction/revocation/rights events between policy steps.
5. Compare single-query, fixed retrieval and adaptive policies under identical budgets.
6. Run the external benchmark suites without modifying their scoring rules.
7. Only after exact fixtures pass, implement the Rust executor and connect real resource, calibration and observation issuers.

The initial benchmark should be small enough to exhaustively enumerate all policy paths and state/outcome combinations. This gives MACE a trustworthy reference before optimizing runtime or adding learned policies.

## 10. Independent finite-world reference evaluator

The crate now exports `adaptive_policy_reference`, an independent reference evaluator for small finite acquisition worlds. It is deliberately separate from `decision_value_retrieval`: it does not call the production evaluator, reuse its EVSI likelihood calculation, or accept candidate-reported scores.

### World representation

A scenario contains:
- a finite sorted state set and consumer-owned action utility/admissibility tables;
- latent worlds, each mapping every candidate query to one deterministic observation;
- an optional exact probability mass per possible world for EVSI;
- conservative possible/not-possible support for robust evaluation;
- query source-root IDs and per-query tool, latency, token, byte and utility-cost units;
- finite horizon and per-path budgets.

The joint-world table is the important independence boundary. Two queries over mirrored pages can have different query IDs but the same root IDs and correlated observation assignments across worlds. Their dependence is represented in the complete joint outcome table; the evaluator does not multiply marginal query likelihoods under a false conditional-independence assumption. Stochastic observations can be represented by multiple latent worlds with the same decision state and different observation assignments. The evaluator does not infer or prove causal dependence from root IDs on its own: the scenario builder must provide a defensible joint-world table and independently grounded root-family labels.

### Expected-value objective

For a terminal action chosen in each leaf, the evaluator computes the exact fixed-point numerator:

[
J_{EV}(pi)=sum_{w in W} p_wleft[U(a_pi(h_w),s_w)-C_pi(w)ight],
]

where (h_w) is the observation history for world (w), and (C_pi(w)) is the accumulated query cost on that world's path. The denominator is exactly 1,000,000; values are compared before rounding. Probability mass on possible worlds must sum to 1,000,000. Impossible worlds must have zero or absent mass.

### Probability-free robust objective

Robust mode ignores all probability fields and evaluates the consumer-declared conservative possible-world support. If that support omits a real state or outcome, the minimax guarantee does not apply:

[
J_R(pi)=max_{w in W_{possible}}
left[
max_{a:,allowed(a,s_w)} U(a,s_w)
-U(a_pi(h_w),s_w)+C_pi(w)
ight].
]

It selects the policy with minimum worst-case regret-plus-cost. Every terminal action must be admissible in every state still compatible with that observation history. No probability estimate is synthesized.

### Search and limits

The reference uses exact recursive backward induction over each reachable history, trying every eligible query and every consumer-permitted terminal action. Query outcomes partition the active joint worlds; each branch is solved independently because resource budgets are per path and the objective separates over disjoint observation histories. Equal scores prefer fewer policy nodes and then a canonical lexical policy representation.

The search rejects scenarios exceeding the fixed small-world limits (8 states, 64 worlds, 6 actions, 8 queries, horizon 4) or the search-history cap. A returned policy exceeding the declared node cap is explicitly rejected as `PolicyTooLarge`; the reference does not silently substitute a different, potentially suboptimal smaller policy. The reference is not designed for production-scale search and has no learned proposal mechanism.

### Deterministic adversarial fixtures

The evaluator's portable input contract is [mace_adaptive_reference_scenario_v1.schema.json](../../data/schemas/mace_adaptive_reference_scenario_v1.schema.json), with a worked [finite joint-world example](../../data/fixtures/mace_adaptive_reference_scenario_v1.example.json). The schema validates JSON shape and primitive bounds; the Rust validator additionally enforces cross-table key coverage, uniqueness and probability-mass invariants. The example contains no production trust material or signatures.

The Rust module currently authors 15 deterministic test functions covering:
- conditional routing to different follow-up queries;
- exact EVSI and robust evaluation over joint worlds;
- duplicate source roots with mirrored observations;
- revocation between the initial observation and terminal choice;
- action prohibition across unresolved possible worlds;
- per-path query, cost and resource ceilings;
- malformed expected-value probability mass;
- query enumeration-order invariance;
- bounded search and policy-size rejection;
- zero-probability/unsupported-world handling.

These tests are authored only. They have not been compiled or run; hosted CI is not reported as passing. The reference can be used as an independent expected-result oracle for a later optimized executor and for cross-language implementations.
For the published four-world example, an independent brute-force policy enumeration yields the conditional tree `route → A: check-a; B: check-b`. Its exact expected-value numerators correspond to expected utility 10, expected query cost 1 and expected net utility 9 units; the probability-free worst-case regret-plus-cost is 1 utility unit. These are synthetic fixture values used to pin implementation correctness, not external benchmark scores.




## 11. Limits

This proposal does not prove source truth, full ancestry completeness, independence of declared operators, calibrated likelihood quality, safe real-world action or secure erasure. Those remain separate qualification obligations. The schema validates shape, not trust. A valid policy graph is not a valid evidence graph, and a valid evidence graph is not an action permit.
