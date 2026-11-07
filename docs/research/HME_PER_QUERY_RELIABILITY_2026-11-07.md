# HME per-query reliability and selective exactness research protocol

**Date:** 2026-11-07  
**Status:** Research protocol; no implementation, reproduction, or HME performance result is claimed  
**Parent:** #10915 — HDC-native Holographic Memory Engine  
**Related:** #10917, #10918, #10945, #10973, #10974, #10978, #11000

## Executive thesis

HME should expose a query-level guarantee contract rather than a single undifferentiated "search quality" number.

The proposed service contract has two legitimate execution families:

1. **Deterministic exact:** full exact scan or qualified progressive exact scan. Every eligible item is fully scored or excluded by an independently admissible upper bound. The final result agrees with the canonical exact oracle.
2. **Probabilistic approximate:** a named, versioned index profile with an explicit per-query target (tau, delta), a declared probability space, assumptions, index generation, update history, and independent quality evidence.

A third output is mandatory when neither contract is met: **Incomplete / Uncertified**. A deadline, candidate count, empirical average, or confidence estimate never upgrades this state to exact.

The intended differentiator is *selective exactness*: spend approximate compute where it is qualified and useful, preserve an exact route for callers that require it, and never silently downgrade a requested guarantee.

## 1. Literature and guarantee boundaries

### 1.1 PQR-Search

Loh and Lee, "Per-Query Recall Guarantees for Approximate Nearest Neighbor Search in Dynamic Vector Databases," arXiv:2611.07579, submitted 2026-11-05.

The paper states a guarantee of the form:

P[Recall(q) < tau] <= delta

for every fixed query q. The probability is over a random orthogonal transformation within the index. The stated independence condition is that queries and updates do not depend on this transformation. The paper stores 1-bit rotated residual codes relative to IVF centroids and uses exact quantiles of the residual-error distribution for pruning. It reports experiments on 20 datasets up to 100 million vectors, including a streaming-update audit.

The paper reports that HNSW tuned to average Recall@k 0.99 still left 3–29% of tested queries below 0.99 in compared settings. Where HNSW met the target, PQR-Search reported 2.5x its single-query latency and 1.3x lower batch throughput. PQR-Search also reports substantial speedups over FAISS exhaustive search in its static experiments. These are the authors' reported results, not HME measurements and not universal ratios.

**Interpretation:** PQR is a serious comparator for a probabilistic HME profile. It is not a deterministic exact top-k result. Its independence condition is a real operational boundary. A fixed query selected before the random transform is distinct from an adaptive query selected after observing responses from that index. This protocol treats adaptive interaction as a separate threat model and makes no PQR-style guarantee claim for it without a new proof.

### 1.2 Decision-aware quantization

Xiao and Cao, "When Does Low-Bit Quantization Preserve the Decisions of Vector Search?", arXiv:2609.09854, submitted 2026-09-09.

The paper separates exact decision margins from quantization residual tails and accounts for covariance between residuals that share a query or graph node. It also studies whether approximate replay preserves a frozen Vamana selection trace.

Its stated scope matters: frozen candidate sets and traces do not prove candidate-generation coverage or end-to-end recall. HME must report ranking/pruning-decision flips and exact-oracle top-k recall separately.

### 1.3 Block-max dynamic pruning

BlockMaxWAND and variable-sized block-max WAND are established information-retrieval prior art. HME's contribution, if demonstrated, is not block pruning itself. It would be a representation-aware adaptation using HDC block geometry, exact score semantics, page locality, and conservative per-candidate bounds.

Relevant HME owners are #10945 and #11000. This protocol owns the per-query quality and benchmark contract, not a second pruning implementation.

## 2. Guarantee taxonomy

Every result and benchmark row must name exactly which claim is being made.

| Class | Claim | Probability source | Required evidence |
|---|---|---|---|
| ExactFull | Canonical full exact scan result | None | 100% oracle agreement under the declared score contract |
| ExactProgressive | Same exact result with safe pruning | None | Candidate accounting, admissible outward bounds, 100% oracle agreement |
| ProbabilisticPerQuery | P[Recall(q) < tau] <= delta for each fixed q | Explicitly named index randomness | Proof, assumptions, implementation conformance and update validity |
| EmpiricalSelective | A held-out workload has measured failure risk below a stated upper confidence limit | Sampling/calibration process | Frozen holdout, independent sampling and a valid finite-sample procedure |
| IncompleteOrUncertified | The required completion or guarantee was not established | None | Explicit stop reason and work accounting |

Important non-equivalences:

- Mean Recall@k does not imply a per-query bound.
- A per-query probability bound does not imply deterministic exactness.
- An empirical held-out risk estimate does not imply a distribution-free guarantee for arbitrary future queries.
- A low score-estimation error does not imply candidate coverage.
- A checksum proves byte identity relative to a digest; it does not prove a mathematical bound is conservative.
- "Exact rerank" of an incomplete candidate set does not make the end-to-end result exact.

## 3. Proposed execution contract

The query request should choose a guarantee, not just a latency budget.

### ExactOnly

- Use ExactFull or a qualified ExactProgressive profile.
- Every eligible item must be fully scored or independently pruned by an admissible bound.
- If deterministic work limits expire before proof, return Incomplete / BudgetExhausted.
- Never return an approximate result under an exact request.

### ApproximateAllowed(tau, delta, profile)

- Run only a named, versioned approximate profile whose stated guarantee matches the request.
- Include the exact oracle definition, metric, eligible corpus generation, query digest, index digest, transform commitment, update watermark and assumption record.
- If the profile cannot support the requested target, return Uncertified or use an explicitly permitted exact fallback.
- Report actual candidate coverage and recall against exact ground truth in the benchmark.
- Do not promote a held-out empirical confidence estimate into a theorem.

### BestEffort

- May return provisional neighbors during execution.
- Provisional results remain Incomplete until a defined completion condition is met.
- If a budget is exhausted, expose the stop reason and resource use.
- An incomplete exact search must not silently become ApproximateComplete.

### Selective exactness hypothesis

For each query, the planner can choose among:

1. exact progressive scan;
2. a probabilistic index profile that satisfies the caller's declared (tau, delta);
3. an exact fallback if permitted and budget remains;
4. explicit Incomplete / Uncertified.

The planner may use estimated query difficulty to choose execution order or recommend a profile. It may not infer a guarantee from query difficulty alone. The guarantee must come from the selected algorithm's proof or an independently valid empirical procedure.

## 4. Per-query reliability benchmark

### 4.1 Baselines

Use one canonical exact ground-truth oracle, identical IDs, metric, normalization, tie policy, eligible corpus and query set.

Compare where implementations and licenses permit:

- exact brute-force / FAISS Flat;
- HNSW;
- IVF and IVF-PQ;
- DiskANN / Vamana;
- RaBitQ and Lucene BBQ;
- PQR-Search official implementation and configuration;
- existing Symthaea HdcStoreReader exact and LSH paths for the supported BinaryHV family;
- HME ExactFull;
- HME ExactProgressive fixed-order and adaptive-order;
- each explicitly approved HME approximate profile.

Do not compare different similarity spaces as if Recall@k were directly equivalent. BinaryHV and ContinuousHV require separate ground truth and parameter sweeps.

### 4.2 Query strata

Freeze the query list before generating or revealing the PQR-style index transform. Partition the evaluation into:

1. in-distribution queries;
2. held-out domain / encoder shift;
3. sparse-support and localized-coordinate queries;
4. periodic and phase-structured queries;
5. near-duplicate and high-interference queries;
6. low-margin kth-boundary queries;
7. norm-extreme and heavy-tail queries within the admitted input domain;
8. adversarial queries constructed independently of the transform;
9. adaptive-feedback queries selected after observing index responses.

Stratum 9 is a distinct threat-model audit. Do not fold it into the theorem-supporting non-adaptive panel. It specifically tests whether an exposed service leaks enough index-dependent information for a client to choose subsequent queries as a function of the transform.

### 4.3 Dynamic update trace

Replay a frozen insert/delete/update sequence. Record and test:

- initial index generation;
- every update batch and watermark;
- every query's position relative to the transform and update events;
- recall and per-query failure after each declared audit point;
- whether a retune, rebuild, centroid change or recalibration occurred;
- stale metadata, tombstones and candidate resurrection;
- query/update dependence on revealed index state.

Use the PQR paper's public streaming runbook when its data and license are available. Preserve a separate HME-native HDC update trace. Never claim to reproduce the official runbook if its exact protocol was not run.

### 4.4 Metrics

For every query, retain:

- Recall@1, @5, @10 and @100 where applicable;
- exact top-k ID agreement and tie-aware ordering;
- per-query candidate coverage before exact rerank;
- whether the query fell below each requested tau;
- exact-score margin to the kth boundary;
- bytes read, payload bytes avoided and candidate evaluations;
- query latency and batch throughput;
- profile / index / corpus / query-list / oracle digests.

Report the complete per-query recall distribution, minimum, p01, p05, p50, p95 and p99, and the fraction of queries below each tau. Also report the confidence interval for the failure fraction by stratum. Averages are supplemental, never the only quality statistic.

Performance must include p50/p95/p99, single-query latency, batch throughput, resident/index memory, build time, update cost, compaction cost, bytes read, and cache-resident versus cold-storage profiles. Report equal-memory and equal-quality frontiers separately.

## 5. Statistical audit design

For a probabilistic guarantee whose probability is over an independently sampled transform, repeat independent transform seeds against each frozen query in the audit panel. Count a failure when Recall(q) < tau.

For a simple one-sided binomial audit with zero observed failures in n independent trials, the upper confidence limit is:

p_upper = 1 - alpha^(1/n)

Therefore, the minimum zero-failure trial counts are approximately:

| Target failure probability delta | One-sided confidence | Independent transforms per fixed query |
|---:|---:|---:|
| 0.01 | 95% | 299 |
| 0.01 | 99% | 459 |
| 0.001 | 95% | 2,995 |
| 0.001 | 99% | 4,603 |

These counts are audit-planning values, not proof of the PQR theorem. With observed failures, calculate an exact binomial upper limit rather than applying the zero-failure shortcut. Account for multiple query strata and multiple simultaneous claims, for example with a predeclared family-wise error allocation. Record the seed-generation process and ensure transform trials are actually independent.

Do not pool all queries into one binomial count and then claim a per-query guarantee: query heterogeneity is exactly what the evaluation is intended to expose. A theorem can establish a bound for every fixed query under its assumptions; empirical audits are falsification tests and implementation checks.

## 6. Numeric and exactness controls

- ExactContinuousHV uses the canonical full-dimensional score definition owned by #10918.
- ExactBinaryHV uses the admitted valid-bit mask and canonical Hamming semantics.
- NaN, infinity, invalid dimension, zero-norm and tie behavior must match the oracle contract.
- Outward-rounded bounds are mandatory before a bound can prune.
- Upper-bound equality with the kth threshold continues scoring unless the stable-ID tie rule proves the candidate cannot win.
- Candidate-order hints and quantized sketches may change order only; they do not authorize exact pruning.
- For adaptive block order, separate the pruning interval accumulator from the canonical exact reduction when evaluation order would change floating-point results.
- Independently reconcile eligible count = fully scored + safely pruned + unprocessed for exact queries; approximate queries also reconcile candidate-policy omissions and candidate coverage.

## 7. Promotion targets

These are proposed project targets, not measured results or external standards.

| Profile | Promotion target |
|---|---|
| ExactFull / ExactProgressive | 100% exact top-k ID/order agreement; canonical score agreement; zero false pruning |
| ProbabilisticPerQuery standard tier | tau >= 0.99 and delta <= 0.01 for each fixed query under a reviewed proof and explicit assumptions |
| ProbabilisticPerQuery stretch tier | tau >= 0.999 and delta <= 0.001, only if proof, memory and latency remain practical |
| EmpiricalSelective | Predeclared held-out risk target and valid one-sided upper confidence limit, explicitly not a worst-case theorem |
| Dynamic update support | Maintain the exact stated guarantee after each supported update without unreported retuning |
| IncompleteOrUncertified | Zero promotion to any completed guarantee class |

For exact-performance promotion, use equal outputs and report payload-byte reduction, metadata overhead, end-to-end throughput and p95/p99 latency. The existing HME aspiration of at least 1.5x throughput at equal quality remains a target only after a valid common harness is operational. Never compare a probabilistic result against an exact result solely by latency.

## 8. New research hypotheses

### H1 — Query-tail-aware budget allocation

A fixed approximation budget is inefficient when query difficulty varies substantially. Investigate a versioned controller that estimates difficulty from admissible bounds, score margins and workload metadata, then allocates more work to hard queries. It must not turn an estimate into a guarantee.

### H2 — Risk-to-exact escalation

When an approximate profile cannot satisfy the requested (tau, delta), test whether HME can escalate to progressive exact retrieval. If the exact route cannot finish within budget, return Incomplete / Uncertified. Measure how often escalation succeeds and its bytes/latency cost.

### H3 — Transform-adaptivity exposure

Test whether query outputs, timing, candidate IDs, scores or update acknowledgements create an adaptive channel that violates the transform-independence premise. Evaluate opaque transform commitments, delayed seed disclosure for audits, information minimization and query-rate controls. These are mitigations to investigate, not proofs of independence.

### H4 — Joint retrieval and decision certificates

Measure candidate coverage, pairwise ranking flips, pruning-action flips and final top-k recall as separate observables. Determine whether query-conditioned margin profiles predict when an approximate run should escalate, while retaining exact upper bounds as the only authority for deterministic pruning.

## 9. Nonclaims

This protocol does not claim that HME implements PQR-Search, has a per-query probability theorem, beats PQR-Search, achieves exact retrieval faster than PQR, or has passed a runtime benchmark. It does not claim a statistical audit proves a universal guarantee. It does not treat a checksum, a synthetic fixture, a quantization error average or a frozen-trace result as proof of end-to-end retrieval correctness.

The immediate deliverable is a reproducible evaluation contract and a falsifiable set of hypotheses. A later implementation must be reviewed against the theorem, the exact oracle, the dynamic update protocol and the independent audit harness.
