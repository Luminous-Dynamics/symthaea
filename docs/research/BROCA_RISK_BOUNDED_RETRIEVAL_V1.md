# Broca risk-bounded adaptive retrieval v1

Status: design/preregistration plus a standalone statistical reference crate. The crate implements a held-out binomial risk-gate calculation; an adaptive retrieval policy is not implemented, and no performance or task-quality improvement is claimed.

Related work:
- [Issue #8477: approximate HDC retrieval against exact MemoryBridge](https://github.com/Luminous-Dynamics/symthaea/issues/8477)
- [Issue #7778: nested-width progressive retrieval and adaptive precision](https://github.com/Luminous-Dynamics/symthaea/issues/7778) — complementary representation/precision ladder
- [Issue #6531: adaptive HDC resolution ladder](https://github.com/Luminous-Dynamics/symthaea/issues/6531) — compatible-representation constraints
- [PR #8494: paired exact-vs-approximate retrieval benchmark](https://github.com/Luminous-Dynamics/symthaea/pull/8494)
- [PR #7502 / issue #7580: adjacent selective-risk and context-stratified coverage tooling](https://github.com/Luminous-Dynamics/symthaea/pull/7502) — prior art inside this repository; the memory-specific curve here is not a novelty claim
- [Store ANN evaluation contract](../../crates/domains/symthaea-hdc-store/docs/ANN_EVALUATION.md)
- [Store read-only ANN contract](../../crates/domains/symthaea-hdc-store/docs/READ_ONLY_ANN.md)
- [Standalone Rust risk-gate reference](../../research/broca-risk-bounded/README.md)
- [Exact binomial confidence limits (NIST)](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm)
- [Selective Conformal Risk Control](https://arxiv.org/abs/2512.12844)
- [SCoRE: Conformal Selective Prediction with General Risk Control](https://arxiv.org/abs/2603.24704)
- [Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control (2026)](https://www.mdpi.com/2571-905X/9/5/109)

## Thesis: retrieval should spend compute where the risk is

A fixed approximate search has one global quality/latency trade-off. A candidate-count fallback is useful for empty or undersized candidate sets, but candidate count alone cannot show whether the true nearest neighbors were omitted. A large candidate set can still be wrong; a small one can still contain the exact top-K.

The proposed shift is from **fixed approximate retrieval** to **risk-bounded adaptive retrieval**:

1. Run a cheap approximate search.
2. Estimate the risk that the result violates an explicit quality contract, using query-time signals calibrated on separate exact-ground-truth queries.
3. Return the approximate result only when the operating point has sufficient held-out evidence; otherwise expand the search if a supported broader-search tier exists, or fall back to the exact scan.
4. Record the chosen route, estimated risk, realized cost, index/version identity, and quality diagnostics in an auditable decision record.
5. Treat unsupported query classes, drift, missing diagnostics, and failed integrity checks as reasons to fail closed to exact search—not reasons to lower the quality bar.

This makes latency a controlled resource allocation problem rather than an unconditional promise that ANN is always faster. It also keeps approximate similarity separate from the claim that a result is globally top-K.

**Boundary with existing work:** issue #7778 investigates progressively increasing HDC dimensionality/precision to preserve candidates; this document investigates when to trust, escalate, or reject an approximate retrieval result. These axes may compose, but they must first be evaluated independently, then as a joint policy against the same exact high-width oracle. Do not count the two proposals as independent wins if they save the same work or compound the same miss risk.

## Non-goals and safety boundary

- Do not change the current exact-default MemoryBridge behavior as part of the initial research stage.
- Do not call an ANN candidate count, score margin, or calibration output a correctness certificate.
- Do not train or tune a routing policy on the same queries used for its final evaluation.
- Do not treat random out-of-distribution vectors as semantic negative examples unless an explicit answerability/abstention label exists.
- Do not pool vector-retrieval metrics with downstream agent-memory scores into a single headline number.
- Do not ship a learned/adaptive policy until its provenance, failure modes, calibration population, and fallback behavior are observable and tested.

## Candidate policy design

### Stage A — exact-ground-truth replay

Use the source-bound benchmark from PR #8494 as an instrumentation baseline, then extend the workload with representative memory traces. For each query, run the exact search as the evaluation oracle and capture the approximate result and all features that would have been available *before* choosing a routing action.

Candidate routing features to evaluate—not assume useful—include:

- whether the index is present and validated, plus index/configuration identity;
- candidate count and candidate fraction;
- LSH bucket occupancy and collisions across independently seeded hash views, if such views are exposed;
- top-1/top-K score gaps and result-score distribution;
- candidate diversity and overlap between independent approximate views;
- store size, live/tombstone population, index age, and recent update/rebuild status;
- query and memory-type strata that are available at runtime without leaking benchmark labels.

An exact-vs-approximate comparison can label whether Recall@K crossed a chosen threshold for offline training and evaluation. Exact outputs must remain hidden from the policy during its test-time decision.

### Stage B — fixed, interpretable cascade

Start with a policy that can be explained and replayed:

- **Exact-only:** reference and safe baseline.
- **Approximate-then-exact:** try the existing approximate search, then return the exact result when integrity checks fail or the frozen risk gate rejects the approximate result.
- **Progressive:** where the index/API supports additional probe budgets, escalate through preregistered budgets before the final exact fallback. If the current index exposes no supported expansion operation, do not simulate one by relabeling repeated identical queries as more search.
- **Always-approximate:** ablation only, not a default safety policy.

The current `ApproximateSearchOptions` exposes candidate-count and empty-result fallback controls; it does not itself expose a confidence certificate or general progressive-probe schedule. The implementation must add real, testable search tiers before claiming progressive retrieval exists.

### Stage C — calibration and selective routing

Fit a simple, interpretable risk model on one partition, tune routing thresholds on a second partition, and evaluate once on a locked test partition. A simple monotone model or a small calibrated classifier is preferable to an opaque model until it proves a material advantage.

Define an unsafe accepted result before seeing the final test outcomes. Example starting definition for a non-critical workload:

`unsafe(q) = Recall@K(q) < 0.80`

This is a proposed engineering definition, not a universal safety standard. Also report strict exact-ID Recall@K, tie-aware recall, and task-specific evidence misses. For high-impact or privacy-scoped memory, the unsafe event should include failure to retrieve a known required fact or cross-scope leakage; a good mean recall does not offset such errors.

The policy may choose ANN only when the calibrated risk gate satisfies the preregistered contract. An unsupported stratum, stale calibration, drift, missing feature, or integrity fault routes to exact. This is a statistical operating policy under a defined evaluation population, not a mathematical guarantee for arbitrary future data. Monitor drift and revalidate whenever the index, representation, corpus regime, or routing model changes.

## Preregistered comparison matrix

Run the same stable query IDs and the same store contents through each arm. Separate index build time from query time, and distinguish warm-cache from cold-start and steady-state measurements.

| Arm | Purpose |
|---|---|
| Exact HdcStore scan | Ground truth, latency floor/ceiling reference for the current storage layout |
| Current LSH default fallback | Existing approximate behavior, including current candidate-count / empty-set fallback |
| LSH without fallback | Ablation that exposes quality loss and prevents fallback from masking ANN weakness |
| Risk-gated ANN → exact | Main hypothesis: accept only queries whose calibrated unsafe-result risk meets the contract |
| Progressive ANN → exact | Evaluate only after real expansion/probe tiers exist |
| Nested-width HDC ladder (#7778) | Separate precision-routing axis; compare to maximum-width exact reference |
| Joint risk gate × precision ladder | Optional composition arm, only after each axis passes independently |
| FAISS `IndexBinaryFlat` | Optimized exact binary-Hamming baseline |
| FAISS `IndexBinaryHNSW` and/or `IndexBinaryIVF` | Established approximate binary index baselines |
| pgvector bit-Hamming HNSW | Database/index operational baseline |
| Strong task-level memory baseline | Compare memory utility, not just binary-neighbor recall |

Official references:
- FAISS binary indexes: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes
- FAISS binary-hashing benchmark methodology: https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark
- pgvector Hamming operator/index support: https://github.com/pgvector/pgvector
- MemoryAgentBench: https://arxiv.org/abs/2507.05257
- Agent Memory Leaderboard / Cycle 2 guide: https://agentmemoryleaderboard.ai/api-guide

For binary ANN comparisons, match the vector bytes, dimensionality, Hamming metric, query vectors, top-K, data population, and hardware. Include adapter overhead separately. Do not compare different bit widths or different corpora as if they were matched. For agent memory, compare under the same task set, answer model, prompt/context budget, API contract, and versioned evaluation protocol.

## Initial gates: targets, not results

These are proposed go/no-go criteria to preregister before the locked evaluation. Adjust only before collecting the final test set, with the reason recorded.

### Retrieval quality and routing risk

- At least 1,000 held-out queries for aggregate quality; increase sample size when evaluating rare failure modes or multiple independent strata.
- Mean strict exact-ID Recall@K at least 0.98 for the target production-like query distribution; also report the existing store-level 0.95 threshold as a comparable baseline rather than silently replacing it.
- Worst-query Recall@K at least 0.80, with per-query output retained.
- Mean tie-aware Recall@K at least 0.99, shown only as a supplement to strict recall.
- For the primary target stratum, the one-sided 95% upper confidence bound on the unsafe-accepted-result rate should be at most 1%. Report numerator, denominator, and the exact interval method. This gate needs enough accepted test examples; zero observed failures in a small set is not sufficient evidence. The standalone reference in [research/broca-risk-bounded](../../research/broca-risk-bounded/README.md) computes a one-sided Clopper-Pearson bound and requires a configured minimum accepted sample count; under its binomial assumptions, 299 independent accepted cases with zero unsafe outcomes are needed to put the 95% upper bound at or below 1%. It also provides a descriptive risk–coverage curve (ties enter together; missing outcomes remain visible) and a fixed-sequence threshold certification routine that stops at the first risk-budget failure. The candidate threshold sequence must be preregistered independently of evaluation labels. This design is aligned with recent finite-sample selective-risk calibration work, including the 2026 fixed-sequence confidence-interval calibration framework; it is not a claim of a novel risk-control theorem.
- Report every error class separately: low-recall results, high-confidence false accepts, exact fallback, no candidates, stale index, OOD/unanswerable query, and scope/privacy failure.
- Scope isolation, deletion/forgetting and index-integrity tests have a zero-tolerance gate: any cross-scope retrieval, resurrected deleted memory, or unvalidated index use fails the candidate policy regardless of average recall.

### Latency, cost and task utility

- Target at least 2× p95 query-latency reduction over exact HdcStore scanning at 10,000+ live vectors, while satisfying the quality and routing-risk gates above.
- Report p50, p95 and p99 latency, full latency distributions, candidate/examined vectors, exact-fallback rate, cold-open/index-build cost, steady-state throughput, peak RSS and bytes per live vector.
- Measure write/update/delete/compaction/rebuild cost and behavior after interruptions; fast search is not a win if it makes memory mutation or recovery unacceptable.
- On held-out agent-memory tasks, require task success to be non-inferior to exact retrieval by a preregistered margin (initial candidate: no more than a 1 percentage-point absolute loss), and separately report false memories, missed required evidence, selective forgetting, and privacy failures.
- Report CPU model, OS/kernel, Rust toolchain, compiler flags, storage, power mode, exact commit, source cleanliness, seed, commands, raw per-query data, and build/run exit codes.
- Run multiple independent process-level repetitions. Query bootstrap intervals alone do not quantify machine/run variation; report both query-sampling uncertainty and run-to-run variation.

The 2× latency, 0.98 recall, 1% unsafe-acceptance and one-percentage-point task non-inferiority numbers are proposed targets. They have not been measured and are not claims about published SOTA.

The Rust reference uses a simple binomial confidence bound, not a conformal or sequential risk-control guarantee. Its interpretation assumes a frozen policy and label, plus independent or exchangeable evaluation units. Repeated measurements of the same query do not count as independent observations. Correlated query lineages should be grouped or sampled at lineage level. Distribution shift, selectively missing labels, and ongoing monitoring require additional methods and prospective audits. Compare selective conformal risk-control approaches such as SCRC and SCoRE before considering deployment claims; do not describe the binomial reference as distribution-free or robust to arbitrary drift.

## Workloads required before promotion

Synthetic clustered BinaryHV data is useful for controlling neighborhood difficulty, but it is not sufficient for selecting a memory policy. Build versioned, privacy-safe replay sets that contain:

1. clean known-memory retrieval and multi-hop/compositional queries;
2. semantic distractors, near-duplicate memories, contradictory facts, and similarly scored but wrong evidence;
3. temporal edits, superseded facts, deletion/selective-forgetting operations and stale index states;
4. low-evidence and unanswerable questions with an explicit abstention target;
5. user/session/tenant boundaries tested with adversarial cross-scope probes;
6. insert/update/delete churn, compaction, restart, recovery and index rebuild;
7. query distributions that differ from calibration (language, topic, density, memory age, corpus size) to measure drift and selective fallback.

Keep train/calibration/tuning/test data disjoint by conversation, session or memory lineage—not merely by randomly splitting individual queries from the same conversation. The exact-search oracle is for offline scoring, never an input feature on the final test query.

## Decision rule and launch sequence

1. **Baseline:** run PR #8494 and preserve the complete raw run receipt. If the local environment cannot compile/run, record that limitation rather than substituting source inspection for measurements.
2. **Instrumentation:** reuse `AnnValidationReport` / `AnnValidationSuite` for strict recall, candidate fraction and fallback metrics. Add only the extra observability required for latency, tie handling, per-class analysis and routing-risk evaluation. The standalone risk-gate crate can produce development risk–coverage curves and fixed-sequence certification reports once its exact-head Rust tests pass; it does not train or calibrate the routing model.
3. **Offline policy replay:** train/tune on disjoint partitions; evaluate frozen rules on a locked test set. Include always-exact and always-approximate endpoints so the cascade's trade-off is visible.
4. **Adversarial and lifecycle checks:** prove that deletion, scope boundaries, index validation and exact fallback are preserved through normal operation and recovery.
5. **Shadow mode:** compute the proposed route and its audit receipt but return exact results; compare predicted risk with realized misses without affecting product behavior.
6. **Opt-in canary:** only after offline gates pass, allow policy selection on non-critical, explicitly opted-in workloads. Keep exact fallback available and compare downstream utility.
7. **Promotion:** require matched open-source baselines, task-level non-inferiority, reproducible receipts and a rollback path. A speedup alone is not enough.

## Opportunity for a real paradigm shift

The longer-term opportunity is **compute-on-demand memory**: use cheap search when the query is easy, spend more probing effort when neighborhood evidence is ambiguous, and pay for exactness when the decision is consequential or uncertainty is unsupported. The retrieval system should expose the cost-quality frontier and the reason for every escalation, rather than forcing all queries through one fixed approximation level.

The differentiator is not simply “HDC is faster.” It is an auditable, storage-aware memory substrate that makes its uncertainty operationally visible, preserves exact reference behavior, and chooses its compute budget against a measured risk contract.

## Submission opportunity (separate from algorithmic claims)

The Agent Memory Leaderboard Cycle 2 guide currently lists a material deadline of October 31, 2026 at 23:59 UTC+8 (17:59 Johannesburg), with evaluation ending November 4. Participants must deploy stable Add/Search APIs and submit a version/materials package; a repository-only benchmark is not a qualifying endpoint. Treat this as a time-bounded deployment decision, not as evidence that a leaderboard win is guaranteed. Track endpoint feasibility, scope isolation, reproducible version binding, and smoke-test readiness independently from the retrieval research.


## Cross-domain selective-risk contract and consolidation

A runner-independent comparison against the CRISPR selective-risk kernel in [PR #7502](https://github.com/Luminous-Dynamics/symthaea/pull/7502) found a concrete interface mismatch: the memory curve exposed a zero-coverage origin, whereas CRISPR T10 emitted only observed confidence tiers. I have now updated the CRISPR branch to represent a zero-coverage origin explicitly and use optional fields for rates undefined at zero accepted cases, with a new regression test. That change is source-authored but not compiled or executed; behavioral parity is not yet established. The domains still define different failure events (retrieval unsafe outcome versus classification error), which must not be treated as interchangeable safety labels.

The new [cross-domain fixture](../../research/broca-risk-bounded/fixtures/SELECTIVE_RISK_CROSS_DOMAIN_V1.json) pins threshold inclusivity, tie atomicity, coverage denominators, unknown-label bounds and null rates at zero accepted cases. Its JSON syntax and four expected operating points were independently reconstructed and matched in the authoring environment. Neither Rust implementation was run; this is not source parity or a performance result.

The follow-on [issue #8562 — unify selective-risk semantics](https://github.com/Luminous-Dynamics/symthaea/issues/8562) requires each domain adapter to normalize its **own preregistered error event** to a common metric contract, pass the shared fixture, and preserve domain-specific calibration, context partitioning, sampling units, data rights and authority. Extract a shared kernel only after exact-head Rust tests show parity; do not force both domains into one unsafe-event definition.
