# PITB research track: two-level proof-graph scheduling (2026-11-01)

**Status:** proof-graph state schema, structural verifier and synthetic shared-helper fixture added. No live Lean graph capture, agent routing, proof-graph benchmark run or measured frontier is claimed.

## Competitive research update

Three current results change the target for proof engineering:

1. **CoCo-Prover** (Yao et al., 7 October 2026) treats code proving as task-level metareasoning. It combines an AND/OR proof hypergraph within each declaration with a lemma-dependency DAG across declarations; a symbolic scheduler selects dependency-ready goals and an agentic router purchases bounded actions from specialists. Across five Lean program-verification benchmarks it reports the best solve rate on each benchmark, up to 100% on two, and up to 30.9% lower monetary cost than the strongest evaluated baseline/model combination. Source: https://arxiv.org/abs/2610.09681.

2. **EasyProver / Aegis** (Ganguly et al., 28 October 2026) reports 669/672 PutnamBench problems, six IMO 2026 formalizations, 597/600 NTP4VC conditions, and all 189 VERINA tasks by proof or refutation. The authors introduce Aegis because several public proof sets are approaching saturation. Aegis derives verification obligations from real Rust implementations, using translation through Aeneas and differential validation; the release contains 2,045 Rust-Lean implementation pairs and 11,533 candidate specifications. On a stratified sample of 2,000 statements, EasyProver proves 1,054 (52.7%) within 30 minutes per statement. Source: https://arxiv.org/abs/2610.38174.

3. **Smarter Spending** reports +7.9% cost-quality AUC and 28.9% lower budget at matched accuracy in a simulation over 64 seeds on an 85-problem PutnamBench subset. Its authors identify both a gap to oracle routing and a narrow action space as limitations. Source: https://matharena.ai/optimizing_agents/.

These results do not form one directly comparable leaderboard. CoCo-Prover is a program-verification orchestration system, EasyProver/Aegis includes implementation-derived code obligations, and Smarter Spending evaluates a small PutnamBench subset in simulation. PITB should reproduce the exact task slice, model, environment, budgets, verifier and accounting before claiming a head-to-head.

CoCo-Prover's own limitations matter for the next design: it evaluates only two repository-level benchmarks (NTP4VC and Vero), fixes implementations/specifications as inputs, supports Lean 4 only, depends on LLM semantic judgment for routing/lemma planning/recovery, and reports that its sequential control can be slower in wall-clock time than coding-agent baselines. PITB should target the remaining gap: auditable proof-state graphs, selective reuse and rollback, multi-resource and parallel scheduling, and reproducible semantic/trust evidence.

## Paradigm shift: a proof system should manage a graph of verified progress

The unit of search should no longer be a monolithic proof attempt or a single open goal. It should be a persistent graph of proof obligations and verified state transitions.

### Layer 1: AND/OR hypergraph inside each declaration

- An AND node is a normalized Lean goal state. All of its child subgoals must close.
- An OR node is an alternative proof step, tactic or strategy for the same goal.
- An applied step records the exact parent goal-state digest, the checker that applied the transition, a transition receipt, and the resulting subgoal IDs.
- A terminal proof step carries an exact proof digest and kernel-check receipt.
- A goal is closed only when at least one applied hyperedge has all children closed and the goal carries a kernel-verified closure proof.
- Candidate, rejected and failed steps remain evidence but never count as proof progress.

### Layer 2: dependency DAG across declarations

- Target theorems and proposed helper lemmas are nodes.
- A dependency edge means that a declaration's proof or plan consumes another declaration.
- Helpers move through open, provisional, stable and discarded lifecycle states.
- A stable lemma requires a kernel proof receipt and stable prerequisites.
- A provisional helper is still being proved; it can itself enter the ready frontier, but dependent goals are not ready until the helper is stable and kernel-verified.
- Discarding a helper invalidates its transitive dependents; unrelated proof work is retained.
- A stable declaration cannot depend on a provisional or discarded lemma.

This is the difference between "retry the whole proof" and "retract exactly the work whose assumptions became invalid."

### Symbolic scheduler plus semantic router

At each decision epoch:
1. Recompute the dependency-ready frontier using graph structure only.
2. Remove closed/discarded goals and goals whose active prerequisites are unavailable.
3. Batch related ready goals when shared context or a helper proposal can amortize work.
4. Choose one bounded specialist action and reasoning configuration.
5. Record the action and all resource consumption.
6. Write back only typed, state-bound transitions and checker evidence.
7. Recompute readiness and rollback closure.

The scheduler is symbolic and should not require a model call. The router handles semantic choices and can dispatch:
- deterministic automation (Aesop, simplification, arithmetic and registered tactics);
- proof writer (bounded tactic/proof generation);
- lemma planner (typed helper proposals with explicit consumers);
- recovery (repair or replace an unproductive route).

The controller can grow this set later with premise retrieval, PITB typed repair, proof-insight transfer and R5 snapshot branches. A new specialist is only an option if it has a typed input/output contract, bounded budget, exact state identity and independent validation.

## PITB's differentiators

PITB should combine the graph architecture with evidence that is already being developed:

- ContextEvidenceV2 records Lean's ordered LocalContext and LocalInstances registry.
- R5 live snapshot receipts bind environment, goal metadata, target, ordered local context and the full MetavarContext digest.
- Support-antichain analysis identifies alternative minimal context resources instead of one brittle support set.
- Typed repair records bounded operations, source state and target-preserving post-state.
- Adaptive allocation prices horizon-bounded proof success against remaining model, tool, token, wall, Lean CPU and monetary budgets.
- Benchmark integrity keeps proof validity, target validity and semantic approval separate.

The graph becomes the control plane that unifies these pieces. The existing adaptive allocator is a one-step action selector; it is not yet a graph scheduler or an implementation of CoCo-Prover.

## New contract implemented

The current branch adds:

- data/benchmarks/proof-insight-transfer-v0/proof-graph/proof-graph-state-v0.schema.json
- data/benchmarks/proof-insight-transfer-v0/proof-graph/fixtures/proof-graph-state.synthetic.json
- crates/bridges/symthaea-proof-insights/src/proof_graph.rs
- crates/bridges/symthaea-proof-insights/src/bin/verify_proof_graph_state.rs
- a focused workflow schema check and Rust self-test

The deterministic goal-batching policy is `pitb.shared-stable-helper-batch.v0`: it derives the ready frontier from the declaration DAG, groups goals by shared stable-helper dependencies (or by declaration-local context where there is no shared helper), chooses the largest group with a lexical tie-break, and truncates to `max_selected_goals`. It emits an explicit `stop` when the ready frontier is empty. Both policy ID and cap are part of the frozen scheduler decision.

The verifier checks:
- unique declaration, goal and step IDs;
- all lemma and subgoal references resolve;
- cross-declaration dependencies are acyclic;
- a discarded helper has no active transitive dependents;
- stable declarations have kernel proof receipts, no sorryAx, stable prerequisites, and only closed goals;
- provisional helpers block dependent goals until promotion to stable, while their own open proof goals remain schedulable;
- proof-step goal digests match the exact parent goal state;
- applied transitions carry checker identity and transition receipts;
- terminal steps carry kernel-verified proof receipts;
- AND/OR goal dependencies are acyclic and closed goals have a completed hyperedge;
- the ready frontier is independently recomputed and matches the declared frontier;
- the selected goal batch exactly matches the deterministic shared-helper policy and registered cap;
- an empty frontier forces a no-model-call stop rather than invented work;
- deterministic automation carries no specialist model call; and
- router, specialist and Lean costs reconcile under every resource cap.

The synthetic fixture is deliberately a no-run architecture example. Its two target roots share one stable helper and are scheduled as one ready group. Receipt bytes are hash-shaped test references; the verifier currently checks graph structure and digest shape, not the bytes behind every transition/proof receipt. No live Lean proof graph has been captured or executed.

## Benchmark plan

Keep mathematical proving, open-hole repository proving and implementation-level code verification as different cohorts.

| Track | Cohort | Primary question |
|---|---|---|
| Transfer | Frozen research-math family split | Do proof strategies transfer to unseen mathematical families? |
| Real Lean proof holes | SorryDB_2601, all 1,000 frozen tasks | Can the system close real holes in their original repositories? |
| Program verification | CoCo-Prover task/benchmark revisions: CLEVER, VERINA, AlgoVeri, NTP4VC, Vero | Does graph scheduling improve cost and coverage on interdependent code obligations? |
| Implementation-derived verification | Aegis frozen task/specification snapshot | Does it generalize to fixed-width arithmetic, loops, mutation, casts and explicit failure in real Rust implementations? |
| Repository proof compression | LeanLean's 64 repositories | Can graph dependency analysis remove proof/code overhead while preserving protected declarations? |

For CoCo-Prover, reproduce its exact benchmark subsets and model/harness where possible; use its reported 30.9% maximum cost reduction as an external reference, not a same-cohort target. For Aegis, reproduce the published stratified 2,000-statement sample and 30-minute budget before scaling. For SorryDB, keep Aleph's 57.2% Pass@1 and Gemini's 30.3% agentic Pass@1 as named external reference rows, but don't compare those scores to code-verification task rates.

### Causal ablation arms

| Arm | Scheduler | Action set | Graph capability |
|---|---|---|---|
| G0 | Fixed-step per-goal loop | One proof writer | No persistent graph |
| G1 | Symbolic dependency-ready frontier | One proof writer | Per-declaration AND/OR graph |
| G2 | Symbolic frontier + bounded specialist routing | Automation, proof writer, recovery | Intra-declaration graph |
| G3 | G2 + reusable helper lemma planning | Adds helper planner | Cross-declaration DAG |
| G4 | G3 + provisional/stable lifecycle and selective rollback | Same specialist set | Dependency-aware invalidation |
| G5 | G4 + PITB retrieval/support/typed repair/insight transfer | Full registered action set | Evidence-backed graph and adaptive resource pricing |
| G6 | G5 + R5 post-elaboration snapshot branch execution | Full action set | Same graph with reduced restore/setup cost |

Hold the base model, model versions, context policy, Lean toolchain, target set, verifier, prompts, total budget and random seeds fixed for paired comparisons. Compare one-task-at-a-time sequential control and asynchronous bounded parallelism as a separate factor. A graph scheduler must not gain an unfair compute budget by parallelizing.

### Required metrics and stretch targets

Targets are research goals, not results:

| Endpoint | Qualification target |
|---|---:|
| Verified roots at equal monetary budget | Non-inferior to strongest matched baseline; stretch +5 percentage points |
| Cost per independently verified root | ≥35% lower than matched fixed-step baseline; stretch beyond CoCo-Prover's reported 30.9% best-case saving |
| Cost-quality AUC | ≥10% improvement over matched baseline |
| p95 wall-clock time | ≥20% lower than sequential graph orchestration without increasing total resource spend |
| Shared-helper economics | ≥20% lower total cost on pre-registered multi-consumer cases than proving each root independently |
| Rollback precision | 100% of transitive dependents invalidated; 0 unrelated stable subgraphs invalidated |
| Graph integrity | 0 cycles, dangling edges, stale goal-state joins or false closed-goal claims in mutation suites |
| Proof trust | 100% credited roots pass clean independent kernel replay; 0 accepted sorryAx or unapproved axiom cases |
| Calibration | Publish action/goal-stratified ECE, Brier score and nominal 95% interval coverage on authenticated outcomes |
| Resource accounting | 0 budget violations; router, specialist, Lean and parallel branch costs reconcile |

The 35% cost target is deliberately above the strongest cited CoCo-Prover result, but it is only meaningful on matched tasks/harnesses. The p95 wall-clock goal addresses a reported CoCo-Prover weakness: better monetary efficiency does not automatically imply lower latency.

## Key experiments

1. **Shared-helper crossover:** vary the number of root consumers (1, 2, 4, 8, 16). Measure break-even helper cost, lemma reuse, total proofs, tokens, dollars and latency.
2. **Selective rollback:** inject a false helper conjecture at different depths; measure how many dependent nodes are invalidated and how much unrelated verified work survives.
3. **Goal-batch routing:** compare independent calls, semantic clusters and dependency-aware batches under identical total token/model/tool budgets.
4. **Specialist routing:** ablate deterministic automation, proof writing, helper planning and recovery; record each action's cost and verified outcomes.
5. **Adaptive scheduler:** compare fixed routing, fixed-cost policy, current LCB allocator and a calibrated graph-aware router.
6. **Snapshot scheduling:** compare R4 cold rebuild to R5 snapshot branch execution while reporting wall time, total Lean CPU, memory, cache invalidation and proof yield.
7. **Cross-cohort transfer:** train strategy/router policy on research families, validate on SorryDB/CoCo-style tasks, then freeze it before Aegis evaluation. Do not assume transfer; measure it.

## Known limitations and next milestones

- The graph contract is structural. The referenced Lean transition/proof receipt bytes are not yet individually opened and authenticated by this verifier.
- No Lean adapter currently exports a full live graph from LocalContext, LocalInstances, metavariables and elaborator transitions.
- The scheduler currently verifies a declared frontier and selection; it does not itself call Lean, route agents or run asynchronous workers.
- Cost models and success estimates need authenticated trajectory data and empirical calibration.
- Helper-lemma semantic quality still needs type checking, dependency-acyclicity checks, random/property tests and independent review; structural correctness alone is not enough.

Next:
1. Compile and run the graph verifier's synthetic negative controls in the isolated PITB Cargo workflow.
2. Implement a Lean capture/transition adapter that writes graph nodes and exact transition receipts.
3. Add byte-authenticating checkers for goal-transition and terminal proof receipts.
4. Implement a deterministic frontier scheduler and a bounded router interface with four specialist contracts.
5. Capture a small live Lean task with at least two roots and one shared helper, then prove and replay it cleanly.
6. Reproduce G0–G4 on a pinned CoCo-Prover/Aegis-compatible smoke cohort before scaling to the full datasets.
