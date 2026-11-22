# MACE: Interaction-Aware Memory Portfolio Allocation v0.1

**Snapshot:** 2026-11-22  
**Status:** Research implementation and authored fixtures. No Rust build/test pass or external performance result is claimed.

## 1. Research question

An additive allocator assumes portfolio utility is the sum of per-item values. That fails when memories are redundant, complementary, substitutable, conflicting, or useful only together. The target is the downstream utility of the **whole portfolio** under the same no-memory baseline, task, model, retrieval policy and resource budget.

For two items, the second-order interaction is:

`delta(i,j) = U({i,j}) - U({i}) - U({j}) + U(empty)`

Positive values represent complementarity; negative values represent redundancy or interference. This is a pairwise approximation, not a general Shapley, submodular or globally complete utility model.

## 2. Implemented candidate path

The new candidate module is `consolidation/src/portfolio/interaction.rs`, exported by `consolidation::portfolio`. It defines:
- `PairwiseInteractionEstimateV1`: canonical option digests, signed interaction and absolute error bound.
- `PortfolioInteractionProposalV1`: option-set, model, baseline-policy, task-protocol, scope, bitemporal snapshot, budget, dataset, task-family and source-root split digests, plus a complete pair matrix.
- `PortfolioInteractionCalibrationReceiptV1`: proposal/split identity, sample counts, independent seeds, calibration and held-out MAE, paired cluster-bootstrap lower bound, and held-out multi-event residual.
- `VerifiedPortfolioInteractionSetV1`: opaque admission result.
- `InteractionAwarePortfolioPlanV1`: base resource accounting, selected interaction terms and total conservative value.

The selector consumes representation plans re-derived from verified future-use predictions and source-availability profiles. It does not accept raw LLM-generated utility values as authority.

## 3. Qualification and fail-closed rules

The v0.1 contract requires exact model/baseline/task/scope/as-known/budget identity; disjoint training, calibration and held-out dataset digests; disjoint task-family and upstream-root attestations; at least 200 training, 64 calibration and 64 held-out pair samples; at least five independent seeds; calibration and held-out MAE below their respective base-rate comparators; and a positive paired cluster-bootstrap 95% lower bound for held-out improvement.

Pair matrices must be complete and canonical. Missing interactions are rejected, not assumed to be zero. Every pair is reduced by its absolute error bound before optimization.

For portfolios with at least three distinct events, the receipt additionally requires at least 32 held-out multi-event samples and a higher-order residual within a preregistered tolerance. If the model cannot explain the joint effect beyond pair terms, the verifier rejects the model. Selected terms are rechecked against the exact verified proposal; rehashing a mutated plan is insufficient. The verified interaction set retains the exact verified option records. Admission rechecks every selected option digest, event ID, representation-plan digest, disposition, net value and resource-accounting field against those records; it also rechecks the selected plan's scope, `(valid_at, known_at)`, budget and authority-scope digest against the verified proposal. The authority-scope digest is bound into the proposal and calibration-receipt identity. Rehashing a self-consistent plan after changing these fields is not sufficient for admission.

A digest is not a signature. Fixture verifiers do not authenticate production issuers or independently prove calibration and split lineage.

## 4. Bounded optimizer

The pairwise selector uses deterministic branch-and-bound, selects at most one representation per event, applies the same five resource/risk budgets as the additive portfolio and fails closed on search exhaustion. It is bounded to 90 representation options (4,005 distinct-event pairs), at most 4,096 supplied pairs, and the existing portfolio search-node limit. The additive exact optimizer and greedy baseline remain unchanged.

## 5. Evaluator integration fixture

The evaluator fixture builds three candidate options through receipt-gated raw Session-A ingestion, calibrated future-use prediction, checked source availability, and re-derived `StoreLocator` representation plans. The hidden whole-portfolio utility remains in the evaluator test module. In the fixture, the additive selector prefers the strongest singleton plus one alternative; the hidden evaluator assigns additional joint value to B+C, and the interaction-aware selector should choose B+C.

This is an authored synthetic integration fixture, not a measured performance result or an executed test.

## 6. Benchmark protocol and targets

Use the same model, no-memory baseline, memory/retrieval pipeline, task episodes, tool permissions, task/source split and token/call/latency/cost budgets. Compare no-memory, greedy, exact additive, pairwise interaction-aware and bounded subset oracle (small portfolios only).

Include exact duplicates, semantic paraphrases with different digests, redundant and complementary facts, mutually exclusive procedures, locator/snapshot overlap, source-root common causes, stale/retracted memories, high-value decoys and three-way-or-higher interactions.

Report whole-portfolio utility, regret to bounded oracle, paired lift versus additive, redundant token/call spend, synergy captured, false-redundancy and false-complementarity rates, held-out MAE/interval coverage, higher-order residual, stale activation, provenance violations, p50/p95/p99 latency, search nodes and worst-family regret.

Aspirational qualification targets:
- exact agreement with bounded enumeration on frozen small fixtures;
- zero double valuation of byte-identical source evidence;
- at most 5% regret versus bounded oracle on held-out small portfolios;
- positive paired held-out utility lift versus additive with cluster-bootstrap 95% lower bound above zero;
- at least five seeds and source-/task-family-disjoint held-out clusters;
- reject rather than overclaim when higher-order residual exceeds the preregistered margin.

Freeze thresholds before opening the held-out set. These are goals, not achieved results.

## 7. Industry positioning

Official Agent Memory Challenge Cycle 2 results (published November 19, 2026) report open-method leaders Resurface 61.22/100 (Textual), InsightMem 63.33% resolved (Coding), and Khipu 43.26/100 (Multimodal). Commercial leaders are MemoraX 69.26/100, 68.00% resolved and 53.68/100, respectively. Track metrics differ, so none is a direct MACE score comparison.

For the next eligible Add/Search cycle, a proposed stretch target is five absolute points above the open-method leaders: 66.22/100 textual, 68.33% coding resolution and 48.26/100 multimodal. These are aspirational targets, not a prediction of future SOTA.

OpenAI's Agents SDK separates durable sandbox memory from conversational session history and uses progressive disclosure. Anthropic Managed Agents separates session log, harness and sandbox. TimlMemBench evaluates raw-session ingestion and governed activation. Luminous should preserve raw bitemporal evidence in Mycelix, create versioned evidence-bound memory views in Symthaea, keep provider adapters replaceable, and let Symtropy own isolated execution receipts. Portfolio utility must be measured on downstream tasks, not inferred from memory size or retrieval similarity.

## 8. Claim boundary and next steps

This is a bounded pairwise research optimizer. It does not prove that real utility is pairwise, that the verifier is production-authentic, or that MACE beats external SOTA. Higher-order residual checks reduce one failure mode but cannot prove a universal pairwise model.

Next: add a larger evaluator-owned generator for semantic aliases and higher-order worlds; estimate interaction terms from held-out task episodes; add sampled Shapley only after pairwise calibration stabilizes; integrate Mycelix signed source-root/portfolio receipts (#13187); move evaluator custody outside the candidate repo (#11816); and deploy hosted Add/Search before claiming AML participation (#13188).

No Rust compile, test, rustfmt or CI pass is claimed.
