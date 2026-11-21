# Mycelial Replay Assurance and Causal Provenance

**Date:** 2026-11-21  
**Status:** Research/implementation contract; acceptance targets are proposed, not achieved.  
**Scope:** CognitiveTopology event replay, branch comparison, evidence ancestry, and the Mycelial Cognitive Instrument.

## Executive decision

Do not use *deterministic replay*, *fork*, *lineage*, or *counterfactual* as unqualified capability labels.

An event log can deterministically reconstruct a graph while failing to reproduce model/tool execution, safely continue after an irreversible action, or describe a world in which a discarded action never happened. The instrument therefore needs a **Replay Assurance Vector (RAV)**: independently assessed claims with explicit evidence and scope.

The central research contribution should be a bounded, typed, renderer-independent cognitive-evidence contract that connects:

`owner observation -> accepted event -> causal/evidentiary lineage -> semantic projection -> qualified branch comparison -> read-only morphology`

The renderer remains downstream and non-authoritative.

## 1. External research and comparator findings

### Replay and fork safety

Nakajima's August 2026 paper, *Replay Is a Family of Assertions: Fork-Safety in Event-Sourced Agent Runtimes* (arXiv:2611.06924), distinguishes byte, trace, projection, path, decision, execution, and environment assertions. It shows why a deterministic fold over a fixed log is weaker than schedule-independent execution or safe external continuation. It also identifies incomplete model/tool requests, recorded oracle results, and irreversible external effects as fork boundaries.

Reference: https://arxiv.org/abs/2611.06924

The May 2026 ActiveGraph paper, *The Log is the Agent* (arXiv:2605.21997), is a direct architectural comparator for append-only graph events, deterministic projections, replay, branch forks, structural diffs, and causal lineage.

Reference: https://arxiv.org/abs/2605.21997  
Implementation reference: https://github.com/yoheinakajima/activegraph

**Differentiation boundary:** event sourcing, replay, forking, and graph diffs are not novel by themselves. Symthaea's opportunity is strict Rust contracts for bounded cognitive evidence, provenance-class separation, fail-closed application, explicit status semantics, renderer invariance, and independently qualified replay claims. These are engineering objectives, not proof of superiority over ActiveGraph.

### Dynamic graph comprehension

The 2026 survey *Survey of the intersection of mental maps and visualization* reviews 98 studies and emphasizes both structural comprehension and consistency maintenance. It notes the tension between preserving a user's mental map and allowing important graph changes to become legible.

Reference: https://doi.org/10.1186/s42492-026-00225-1

The 2026 IEEE TVCG paper *Generating Coherent Visualization Sequences for Multivariate Data by Causal Graph Traversal* reports user-study evidence that causal traversal can improve understanding of causal relationships compared with correlation-based or randomized sequences.

Reference: https://doi.org/10.1109/TVCG.2026.3656952

**Implication:** the interface should support causal-path-first navigation, not merely animate the complete graph. Mental-map stability and causal-path readability are separate outcomes.

### Biological morphology

Palmer et al.'s 2025 *Physical Review E* paper, published September 2025 and indexed by PNNL in February 2026, extends BMX with 3D filament mechanics, branching, anastomosis, and nutrient transport. It reports comparisons of generated networks against real fungal structures.

Reference: https://doi.org/10.1103/xbjp-zqj1  
PNNL record: https://www.pnnl.gov/publications/three-dimensional-modeling-hyphal-fusion-branching-and-nutrient-transport-filamentous

**Differentiation boundary:** graph-level fusion candidates and cycle-rank calculations are not a biological growth model. Symthaea should compare against biological morphology only when it has matched image/data-derived measurements, mechanical rules, and transport or other functional dynamics.

### Graph layout evaluation

The 2024 systematic review of graph-layout evaluation surveyed 206 papers and identified substantial variation in datasets, graph characteristics, metrics, and reporting.

Reference: https://doi.org/10.1111/cgf.15073

Use matched datasets and task-specific metrics. Do not infer SOTA from a force-layout microbenchmark or compare results from unmatched graph sizes, hardware, and workloads.

## 2. Replay Assurance Vector (RAV)

Every replay receipt should report each assertion independently. A higher item in this list does not automatically imply a lower or adjacent item.

| Assertion | Question | Evidence required |
|---|---|---|
| Byte identity | Are the serialized source artifacts identical? | Raw artifact digest and canonicalization/version details |
| Trace identity | Is the accepted ordered semantic event history identical? | Ordered event IDs, event digests, session and prefix-chain digest |
| Projection equivalence | Does the same base and accepted log reconstruct equivalent semantic state? | Independent replay results and canonical snapshot digests |
| Path identity | Were the same model/tool/action steps taken? | Typed request and outcome events, with schedule/order evidence |
| Decision equivalence | Did the task-level decision remain equivalent? | Domain-specific decision record and an explicit equivalence relation |
| Execution replay | Can the computation be re-executed without new live model/tool calls? | Recorded request/result receipts, replay interpreter, zero-live-call verification |
| Environment equivalence | Does the external world satisfy the claimed branch assumptions? | Effect footprints, reconciliation evidence, and an explicit environment premise |

### Current source-level scope

The public research branch currently has:

- a versioned, session-bound event type;
- strict contiguous sequencing and monotonic timestamps;
- exact duplicate idempotence and conflicting duplicate rejection;
- atomic event-batch application;
- one optional `caused_by_event_id` pointer;
- a prefix-hash-chained accepted event-log digest;
- a separate semantic snapshot digest;
- graph-projection replay and in-memory prefix forking;
- a read-only morphology projection and a synthetic replay fixture.

The replay CLI now performs an independent second projection fold and reports agreement for the accepted-log digest and semantic snapshot digest. Its scope is deliberately limited to the same binary and process. The receipt does **not** establish byte identity, model/tool path identity, decision equivalence, execution replay, external continuation, or counterfactual-world validity.

The reducer is in-memory. It is not yet a durable event store, distributed log, effect journal, or external-world rollback mechanism. The single causal-parent field does not represent multi-input reasoning completely.

## 3. Multi-parent provenance: the next schema evolution

Real cognitive operations have fan-in: a retrieval result may depend on a query event, multiple candidate observations, a ranking operation, and a downstream consumer. A single parent pointer can preserve only one of those ancestry links.

Introduce a versioned event schema with a bounded set of typed lineage links. Keep **causation** distinct from **evidentiary support**:

- `CausedBy`: an upstream event is a causal prerequisite of this event.
- `DerivedFrom`: this event's artifact or value was computed from a specific upstream artifact.
- `ObservedAfter`: temporal ordering only; not a causal claim.
- `SupportedBy`: a source supports a semantic claim; this does not imply it caused the computation.
- `ContradictedBy`: a source conflicts with a semantic claim; it must not erase the source or the earlier claim.
- `Supersedes`: a later state or record replaces an earlier operational state while retaining history.

Schema and validation requirements:

1. Preserve v1 deserialization and define an explicit migration to v2; do not silently reinterpret existing event identity.
2. Bound lineage links per event (initial experiment: 32); validate identifiers before allocation.
3. Every causal parent must resolve to the same session's base evidence or an already accepted earlier event.
4. Reject self-links, duplicate links, forward references, and cycles. Strict earlier-sequence references make cycle prevention checkable.
5. Preserve parent-link type in event hashing and receipt output.
6. Keep semantic `Supports` / `Contradicts` graph edges separate from the causal event DAG.
7. Provide ancestry closure and shortest/selected causal paths with stable ID ordering.
8. Never infer a causal parent from geometry, visual proximity, activation, or a similarity score.

The resulting provenance graph should answer: which inputs were available, which were actually consumed, which event derived the output, and which evidence merely supports or contradicts the resulting claim.

## 4. External effects and fork boundaries

Graph projection events and real-world side effects are different domains. Add a separate effect contract before claiming safe continuation:

`EffectRequest -> EffectReceipt -> Reconciliation -> ForkPermission`

A receipt should bind operation ID, idempotency key, effect class, declared external footprint, request digest, result digest, producer, and lifecycle state.

Required states include:

- `Requested`
- `Committed`
- `Failed`
- `Unknown`
- `Compensated`
- `NotRun`

Rules:

- Replay consumes recorded results; it must never silently make a new live call.
- A committed effect remains part of the real-world history even when a child graph drops the event suffix.
- An `Unknown` outcome blocks a new mutation until reconciliation.
- Forking before an effect request can be projection-safe.
- Forking after a committed effect can be projection-safe but is not automatically counterfactual-world-safe.
- Continuation requires explicit effect policy, idempotency/reconciliation evidence, and permission for the declared footprint.
- A counterfactual claim requires an isolated simulator or a declared and validated world model. A graph-only fork is a graph-only alternative, not an alternative reality.

## 5. Causal branch comparison

Extend structural diffs with a separate lineage explanation:

- identify the common accepted prefix and each branch's divergent event tail;
- show all changed node, edge, and episode identities;
- connect each changed item to the event that last changed it;
- show the typed causal/evidentiary ancestors for each branch;
- report missing, ambiguous, or unrecorded ancestry explicitly;
- distinguish branch-local candidate, rejected, stale, resolved, and observed states;
- present evidence coverage and unresolved dependencies without ranking which branch is true.

A branch comparator may report *which branch has more complete recorded evidence under a declared policy*. It must not decide that a branch is true merely because it has more edges, more sources, higher activation, or a more attractive morphology.

## 6. Network morphology: from cycle count to functional resilience

The current `beta_1 = E - V + C` calculation is a valid cycle-rank invariant. It is not a full measure of useful redundancy.

Add independent structural and functional measurements:

- node and edge connectivity; minimum cuts and edge-disjoint paths;
- algebraic connectivity (λ₂ of the graph Laplacian) where defined;
- global/local efficiency and path-length distributions;
- single-filament, single-fusion, and correlated-region failure;
- random versus targeted removal and recovery/reconnection cost;
- candidate-fusion precision/recall against a frozen geometric ground truth;
- topology change, path preservation, and evidence-ID preservation after fusion;
- bounded search cost and false-acceptance rate.

Anastomosis must remain a proposed geometric/topological operation requiring explicit acceptance. It must not create semantic relationships. Measure whether a loop restores reachability or alternative routes; do not assume every additional cycle improves resilience.

Later biological comparison should use BMX-compatible image-derived structural metrics, including branch-length and angle distributions, tip density, loop/cycle statistics, component morphology, and—only when implemented—transport behavior. Biological fidelity remains unclaimed until matched data and a validated model exist.

## 7. Benchmark programme

All targets below are proposed acceptance gates, not measured results.

### RAV replay/fork conformance

| Metric | Initial gate |
|---|---:|
| Projection digest agreement for a fixed accepted log | 100% |
| Invalid event/batch mutations | 0 |
| Conflicting duplicate acceptance | 0 |
| Unrecorded live calls during replay | 0 |
| Duplicate committed side effects during fork tests | 0 |
| Unknown effect outcomes that permit new mutations | 0 |
| Unsafe external-continuation claims | 0 |
| Provenance/lineage references that fail closed on invalid input | 100% |

Test cuts immediately before, inside, and after: model requests, tool requests, recorded results, committed effects, failed effects, and unknown outcomes. Include schedule permutations and interrupted/incomplete request boundaries.

### Provenance and causal explanation

| Metric | Initial gate |
|---|---:|
| Valid causal-parent closure recovered | 100% |
| Invalid/dangling/forward causal parents accepted | 0 |
| Ground-truth causal-link precision and recall | 100% on synthetic conformance corpus |
| Semantic support incorrectly presented as causation | 0 |
| Branch-changed items with resolvable last-writer evidence | 100% |
| Unresolved lineage silently omitted | 0 |

Test fan-in, shared ancestors, branch-exclusive ancestors, contradictory observations, duplicated delivery, stale sources, and missing evidence.

### Network resilience and geometric correctness

| Metric | Initial gate |
|---|---:|
| Cycle-rank correctness | 100% against independent graph oracle |
| Fusion proposal determinism | 100% for frozen inputs |
| Unaccepted candidate converted into a link | 0 |
| Non-finite / out-of-bound geometry accepted | 0 |
| Semantic digest changed by morphology-only operation | 0 |
| Failure cases with complete structural report | 100% |

### Layout and instrument comprehension

Measure edge crossings, node overlap, normalized stress, angular resolution, unchanged-node displacement, path tracking, causal-path finding time, provenance lookup, and graph-state classification. Report each separately.

The confirmatory human-factors study should retain text-only and conventional node-link controls, include a static evidence view and the Mycelial view, counterbalance task order, preregister primary metrics, and measure false attribution. Aesthetic preference, engagement, and time spent are not comprehension.

Initial comprehension gates remain:

- at least 90% macro-F1 for state classification;
- at least 95% candidate-versus-resolved discrimination;
- at least 90% provenance comprehension;
- no increase in unsupported beliefs about internal activity, intent, or emotion;
- no material reduced-motion comprehension regression;
- no information conveyed through colour alone.

### Performance and scale

Use fixed, versioned inputs at 32/128/256 semantic nodes and 128/512/1,024 edges (respecting the current schema bounds), plus event logs at 32/256/1,024/4,096 events where the test operations remain valid.

Report cold and warmed P50/P95/P99 for validation, replay, fork, diff, ancestry closure, layout, morphology construction, and rendering. Include allocations/peak memory and event-to-visible latency. Do not compare the synthetic CPU search benchmark to GPU rendering or biological simulation.

## 8. Industry/SOTA comparison discipline

| Comparator | What it already establishes | Symthaea's testable differentiation |
|---|---|---|
| ActiveGraph | Event-sourced reactive graph, replay, fork, structural diff and lineage architecture | Bounded Rust evidence schema, fail-closed provenance, semantic-vs-causal separation, replay/fork property receipts |
| Replay-safety research (arXiv:2611.06924) | Property-relative fork/replay contracts and failure cases | A small executable Rust conformance suite embedded in a cognitive graph / evidence instrument |
| BMX fungal modeling | 3D hyphal mechanics, branching, fusion and nutrient transport | Evidence-preserving projection plus separate morphology experiments; biological modeling only after data validation |
| OGDF / Graphviz and graph-layout literature | Mature force, stress and multilevel layout methods | Temporal, causal-task and provenance-comprehension evaluations on exactly matched fixtures |
| Bevy / Three.js / Babylon.js / PlayCanvas | Production rendering systems with different batching and backend trade-offs | Renderer-neutral input and evidence identity; matched workload and device receipts |
| Conventional status panels and graph views | Strong and necessary baselines for human task performance | Demonstrate improved provenance and causal understanding without increased false attribution |

Do not claim “better than SOTA” until the same graph/event corpora, device class, viewport, user tasks, and statistical method have been used. Publish raw results and negative results.

## 9. Implementation sequence

1. **Replay receipt (current increment):** the CLI repeats the graph projection fold and reports accepted-log and semantic-digest agreement, explicitly scoped to the same binary/process. This does not prove execution or environment equivalence.
2. **RAV contract types:** typed assertion names, status, scope, input digests, verifier identity, and reasons. Serialize a versioned machine-readable receipt.
3. **Multi-parent lineage schema v2:** migration, bounded typed parent links, fan-in/fan-out fixture, ancestry closure and causal path export.
4. **Branch explanation:** common-prefix verification, divergent tails, changed-object last-writer links and ancestry coverage.
5. **Effect request/receipt contract:** lifecycle and external-footprint tracking; unknown-outcome reconciliation; fork permission policies.
6. **Functional mycelial network experiments:** failure/flow metrics and independent graph-oracle tests.
7. **Matched layout/render benchmark suite:** stable and dynamic graph tasks; same input and provenance across renderers.
8. **Preregistered human study:** evaluate comprehension and false attribution before claiming the instrument improves understanding.

## 10. Non-goals and claim boundaries

- No claim of consciousness, subjective experience, intention, or emotion from rendered morphology.
- No biological fidelity claim from a fungal visual analogy.
- No execution-replay or external-world-counterfactual claim from graph projection replay.
- No automatic promotion of synthetic fixture evidence to runtime provenance.
- No causal attribution from temporal adjacency, semantic support, activation, or visual proximity alone.
- No branch ranking by visual aesthetics or unsupported confidence.
- No weakening of validation, provenance, or effect boundaries to improve performance.

The desired outcome is not an attractive avatar for a cognitive system. It is a **provenance-first counterfactual laboratory**: replayable, inspectable, bounded, falsifiable, and honest about the boundary between recorded evidence and the world outside the log.
