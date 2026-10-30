# Broca Incremental Alternative-Support Engine

**Status:** prototype design and implementation branch; tests authored but not executed. No benchmark measurements are attached.

## 1. Motivation

The alternative-support graph separates propositions from justifications:

- Dependencies within a justification are conjunctive (AND).
- Justifications for the same proposition are alternatives (OR).
- A proposition is supported at a query time when at least one complete, eligible justification supports it.
- A dependent proposition references the proposition, not one proof instance.

The first implementation deliberately recomputes all support intervals after a mutation. This is a correctness-first reference. The next step is to reduce recomputation without changing its semantics.

## 2. Incremental point-support algorithm

The new `IncrementalSupportSession` evaluates one fixed valid-time date. This is a deliberate first boundary: interval aggregation remains in `SupportGraph::support_snapshot`, and the session is checked against it at a selected date.

The session maintains:

- A topological order of proposition dependencies.
- A reverse source-event-to-output-proposition index.
- A reverse proposition-to-dependent-proposition index.
- A rule-version-to-output-proposition index.
- A count of active justifications for each proposition.
- The set of active justification IDs and supported proposition IDs.
- Counters for propositions and justifications reevaluated per update.

### Initial evaluation

1. Validate the graph and reject cyclic proposition dependencies.
2. Build all reverse indexes and a deterministic topological order.
3. Visit propositions in topological order.
4. Evaluate every justification for the proposition at the session's fixed `valid_at` date.
5. A justification is active only when its own interval contains the date, its rule version is not retired, every source event is unretracted and temporally active, and every proposition premise is supported.
6. The proposition is supported when its active-justification count is greater than zero.

### Deletion and rule retirement

For a source retraction, the session starts with propositions whose justifications directly reference that source. For rule-version retirement, it starts with propositions produced by that version.

It then performs one deterministic topological sweep:

- Recompute each scheduled proposition's active justification set and count.
- If the proposition's supported/un-supported Boolean state is unchanged, do not schedule its dependents. Removing one proof while an alternative remains therefore does not trigger downstream recomputation.
- If the proposition changes support state, schedule its dependent propositions.
- A dependent is visited after all of its possible premises because the dependency graph is acyclic and the order is topological.

The current prototype uses clone-validate-swap for mutations, preserving the reference graph's in-memory atomicity boundary. The clone cost is intentionally not hidden; future benchmarks must report it separately from reevaluation work.

## 3. Correctness argument and limits

For an acyclic proposition dependency graph, every proposition can be evaluated after all proposition premises in topological order. Its support is a pure Boolean function of:
- the validity and retirement state of its own justification;
- the validity of its source-event premises; and
- the supported state of each proposition premise.

A source retraction or rule retirement can directly change only propositions whose justifications reference that source or rule version. A downstream proposition can change only if at least one proposition premise changes its supported Boolean state. Therefore, recursively scheduling dependents only after such a Boolean change is sufficient to reach the same point-in-time least supported state as a full recomputation.

This argument is limited to the current finite acyclic, positive dependency model. It does not cover:
- recursive or negated rules;
- non-monotone aggregates;
- authority-policy precedence or bitemporal transaction-time filtering;
- simultaneous concurrent writers;
- durable transactions or crash recovery;
- interval-union updates for arbitrary query dates;
- proof provenance minimization.

Do not extend the algorithm to recursive Datalog or negation without explicitly defining fixed-point and deletion semantics.

## 4. Differential test strategy

The integration tests compare the incremental session with two references:

1. **Full interval recomputation:** `SupportGraph::support_snapshot`, evaluated at the session's fixed date. Compare proposition support, active justification IDs, per-proposition active support counts, and ledger digest after each mutation.
2. **Independent date oracle:** a deliberately simple Boolean fixed-point evaluator. It scans all justification inputs and determines support at the date without using the incremental session's reverse indexes or the interval-union implementation.

Generated test traces use deterministic seeds, acyclic proposition edges, multiple alternative justifications, source interval variation, source retractions and rule-version retirement.

A separate work-counter test checks two important efficiency properties:
- Retracting one of two independent supports recomputes only the directly affected proposition; its still-supported dependents are not revisited.
- Retracting an unreferenced source performs zero proposition and justification reevaluations.

These are structural work-count assertions, not wall-clock performance results.

## 5. Algorithmic comparisons

Use one common hypergraph generator and one update trace for all algorithms. Do not compare results from different rule languages or graph distributions as if they were like-for-like.

| Baseline | Deletion behavior | Applicability / caveat |
|---|---|---|
| Full recomputation | Recompute all propositions and justifications | Semantic reference; expected high update work |
| Naive descendant invalidation | Invalidate all descendants of the deleted proof | Simple baseline; can over-invalidate alternatives |
| DRed | Over-delete affected materialisations then rederive | General recursive Datalog baseline; potentially expensive with many alternate proofs |
| Backward/Forward | Revalidate possible alternative derivations before forward propagation | Datalog-specific semantics; implement a compatible subset or use a documented reference |
| Counting | Maintain derivation/support counts | Efficient for suitable rule fragments; classical variants have recursion/negation restrictions |
| Broca session | Maintain point support counts and skip dependents when proposition truth does not change | Current prototype is acyclic positive rules, fixed valid-time date, clone-on-write |

The literature establishes that B/F can outperform DRed by orders of magnitude on some workloads, while other work compares Counting, DRed variants and FBF on different rule fragments. This is a reason to implement the baselines carefully, not a claim that Broca is already faster.

References:
- Motik et al., *Incremental Update of Datalog Materialisation: the Backward/Forward Algorithm*, AAAI 2015: https://ojs.aaai.org/index.php/AAAI/article/view/9409
- Hu, Motik & Horrocks, *Optimised Maintenance of Datalog Materialisations*, AAAI 2018: https://ojs.aaai.org/index.php/AAAI/article/view/11554
- *Maintenance of Datalog Materialisations Revisited*: https://www.sciencedirect.com/science/article/pii/S0004370218307495
- *Incremental Maintenance of DatalogMTL Materialisations*, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/39025

## 6. Benchmark matrix

Generate workloads across these axes:

| Axis | Levels |
|---|---|
| Propositions | 10^2, 10^3, 10^4, 10^5 |
| Justifications per proposition | 1, 2, 4, 16, 64 |
| Dependency fan-in | 1, 2, 4, 8, 16 |
| Downstream fan-out | 1, 4, 16, 64, 256 |
| Graph shape | chain, tree, diamond, layered DAG, power-law fanout, dense acyclic |
| Deletion type | leaf source, shared source, rule retirement, last support, non-last support, irrelevant source |
| Deletion fraction | 0.01%, 0.1%, 1%, 10%, 50% |
| Temporal shape | common interval, disjoint segments, nested intervals, adjacent intervals |
| Alternative proof overlap | disjoint supports, shared leaves, partial overlap, identical support sets |

Report:
- Exact semantic parity with the full reference.
- False invalidation and missed invalidation counts.
- Active justification and proposition support-count accuracy.
- Proposition recomputations and justification evaluations per update.
- p50/p95/p99 update latency and initial-build latency.
- Peak memory and bytes per proposition/justification/index edge.
- Clone/validation cost separately from incremental reevaluation.
- Crossover points where full recomputation is faster.
- Sensitivity to high alternative-proof density and large downstream fanout.

Do not report only average speedup. Publish per-workload curves and the update-size distribution.

## 7. Proposed promotion gates

These are internal targets, not official thresholds or measured results:

1. 100% support-state parity with both reference evaluators on deterministic and generated fixtures.
2. Zero missed invalidations and zero false invalidations.
3. Exact active-justification sets and support counts after every update.
4. Event-order-independent semantic state and deterministic digests.
5. A measured reduction in justification evaluations on non-last-support deletion and irrelevant-source workloads.
6. No worse end-to-end task success or evidence safety after integration.
7. Crash/replay parity before any persistent or production use.

A structural work-count win is not a performance win until wall-clock, allocation and persistence costs are measured on a representative workload.

## 8. Evidence and claims discipline

A September 2026 study of agent-memory consolidation reports that a proposed consolidation-quality score did not predict transfer accuracy on its two-benchmark sample, despite other promising internal measurements. That is an important warning: internal benchmark quality must be validated against independent task outcomes. See *The Epistemics of Agent Memory* (2026): https://arxiv.org/abs/2609.33013.

Therefore:
- Keep intrinsic support correctness, algorithmic efficiency and downstream agent utility as separate claims.
- Validate whether changes in intrinsic scores predict changes in held-out task performance.
- Report failures, counterexamples, negative controls, and confidence intervals.
- Never interpret a deterministic synthetic suite as a generalization result or an external leaderboard score.

## 9. Next steps

1. Compile and execute the exact PR head and differential tests.
2. Fix any counterexample before adding optimization.
3. Add an explicit source-event and rule update event log for reproducible update traces.
4. Add an independent baseline implementation for the supported positive, acyclic rule fragment.
5. Run workload sweeps and publish crossover curves.
6. Extend to transactional persistence and bitemporal support only after point-in-time parity is established.
