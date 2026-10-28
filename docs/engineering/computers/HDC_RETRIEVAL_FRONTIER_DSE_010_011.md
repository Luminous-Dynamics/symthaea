# HDC Retrieval Frontier — DSE-010/011 Findings

**Date:** 2026-10-28  
**Status:** Independent Python/NumPy reference experiments and protocol proposal. Not Rust qualification, not production corpus results, not an SOTA claim.

## Executive result

Retrieval should not be treated as “nearest vector + threshold.” A defensible retrieval decision must account for (1) corpus-wide selection, (2) whether the query contains enough information to identify one record, and (3) whether the application asks for semantic relevance or exact identity.

## DSE-010 — Open-set calibration pilot

The independent NumPy script built 16,384-bit records by majority-bundling 3, 7 or 15 role-bound facts, with 256, 1,024 and 4,096 records per corpus. Two fixed seeds were used. For each of 18 seed/corpus/arity cells, 32 clean known cues and 32 independent absent-value cues calibrated a threshold fixed at one discrete score quantum above the maximum calibration-unknown score. It was then frozen and evaluated on 32 independent unknown cues and 32 held-out known IDs at five corruption levels.

| Measurement | Result | Interpretation |
|---|---:|---|
| Robustness rows | 90 | 18 corpus cells × 5 corruption levels |
| Held-out known queries | 2,880 | 32 queries × 5 levels × 18 cells |
| Held-out known Hit@1 | 2,880/2,880 (100%) | Deliberately easy independent-fact synthetic workload |
| Held-out independent unknowns accepted | 23/576 (4.0%) | 95% Wilson interval approximately 2.7–5.9% |
| Calibration unknowns accepted | 0/576 | Zero by construction; not a population guarantee |
| Byte-identical absent-value twins accepted by score-only threshold | 2,880/2,880 | Same observed bytes as corrupted known cues, so score-only decisions must match |
| Independent scalar-oracle fixture checks | 180 | Packed distances cross-checked on small fixtures |

The 4.0% held-out unknown acceptance is not a stable deployment estimate: each calibration/test cell contains only 32 examples. It illustrates why zero errors in a small calibration sample is not a guarantee. Under exchangeability, setting a cutoff strictly above the maximum of 32 calibration-unknown scores gives a new-score exceedance probability on the order of 1/33 (~3.0%). The observed aggregate is compatible with that scale.

The all-100% known Hit@1 result is itself a warning that independent facts make an insufficiently challenging semantic benchmark. Include shared constituents, overlapping records, stale versions, hard negatives and distribution shift.

## DSE-011 — Context sufficiency under overlap

The second pilot builds 1,024 records from 7 or 15 role-bound facts; records are grouped in clusters of eight and the first S fact slots are identical within each group. Queries reveal M = 1, 3 or 7 fact cues and are tested with 0% or 10% constituent corruption. The comparison is bundled-prototype search versus a role/slot-aware component scorer.

| K | Shared slots S | Query cues M | Information state | Bundled Hit@1 | Component Hit@1 | Bundled mean target rank | Component mean target rank |
|---:|---:|---:|---|---:|---:|---:|---:|
| 7 | 0 | 1 | Unique cue | 100% | 100% | 1.00 | 1.00 |
| 7 | 1 | 1 | Shared cue only | 3.1% | 15.6% | 4.78 | 3.78 |
| 7 | 3 | 3 | All cues shared | 6.3% | 18.8% | 4.62 | 4.44 |
| 7 | 3 | 7 | Includes unique cues | 100% | 100% | 1.00 | 1.00 |
| 15 | 3 | 3 | All cues shared | 12.5% | 12.5% | 4.28 | 4.22 |
| 15 | 7 | 7 | All cues shared | 15.6% | 12.5% | 4.97 | 4.41 |

These are small pilot proportions over two seeds, with 32 query targets per aggregate cell. The exact Hit@1 rates fluctuate; the structural finding is more robust: when all supplied cues are shared across an eight-record group, the target rank stays near the middle of the group (expected rank 4.5 for a uniformly selected target). An algorithm cannot infer the intended identity from a distinguishing feature that was not observed. In the tested cells where M > S, both methods reached 100% Hit@1, including 10% cue corruption.

For K=15 and S=7, this pilot only tests M up to 7; it does **not** demonstrate the recovery point. Extend to M=9, 11 and 15.

The slot-aware component index did not fix missing information. It also stores K fact vectors per record and scans each requested slot, increasing memory traffic and query work. It must be compared on a quality/memory/latency frontier, not presumed superior.

## Paradigm shifts

1. **Check information sufficiency before optimizing.** Return an ambiguity set and the additional evidence needed rather than silently guessing.
2. **Calibrate corpus maxima.** Fit thresholds to each query's maximum score over the whole frozen corpus, not a pairwise score in isolation.
3. **Risk versus coverage.** Track false identity acceptance, false semantic rejection, accepted coverage, abstention and target rank separately.
4. **Context acquisition as an action.** When ambiguous, ask for or retrieve the next most informative independent cue, then measure the number and cost of clarification steps.
5. **Representation frontier.** Compare bundled prototypes, component facts and hybrid indexes at equal task-level quality, including their complete memory footprint and query work.
6. **Evidence receipts.** Bind task mode, query/corpus digests, candidate/version, policy/calibration artifact, evidence checks and source revision to the decision. A digest proves byte identity—not semantic truth or authorization.

## Risk calibration and qualification targets

A useful methodological baseline is selective/conformal risk control, not arbitrary threshold tuning alone. See [Selective Conformal Risk Control (2025 preprint)](https://arxiv.org/abs/2512.12844) and [Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control (2026)](https://www.mdpi.com/2571-905x/9/5/109). These are methodological references, not direct HDC retrieval benchmarks; the assumptions must be checked.

- Under a maximum-score order-statistic rule, the threshold is set just above the maximum of m exchangeable calibration-negative scores. A new exchangeable score exceeds that maximum with probability at most 1/(m+1), so m=32 supports only about 3% marginal exceedance. The same simple rule needs m=9,999 for a 10^-4 target.
- Separately, zero errors in about 30,000 independent representative holdout cases gives an approximate 95% upper bound near 10^-4 by the rule of three. Calibration guarantees and confidence bounds are different claims.
- Neither statement guarantees conditional risk for every identity, adversarial group or shifted future distribution.

| Axis | Initial benchmark gate |
|---|---|
| Exact top-1 | Zero mismatches against an independent Hamming oracle, including ties |
| Semantic quality | Hit@1/Hit@k, target rank and margin stratified by overlap, arity, noise, corpus size and seed |
| Open-set | Frozen threshold; per-negative-profile false accepts/rejects; risk-versus-coverage with intervals |
| Ambiguity | Never identity-confirm same-byte twins without independent identity evidence; measure clarification cost |
| Calibration | Sweep 32, 128, 512, 2,048 and 10,000+ negatives where compute permits; preserve untouched test data |
| Baseline | FAISS IndexBinaryFlat on identical packed bytes/queries/tie rules; approximate methods on separate recall/latency frontier |
| Performance/energy | p50/p95/p99, throughput, setup/build, index memory; directly measured joules/query and joules/success |

FAISS supports exhaustive binary Hamming search with popcount and multiple indexed alternatives: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes. MLPerf Tiny provides useful fixed-workload and energy-measurement discipline, without implying Symthaea has an official score: https://mlcommons.org/working-groups/benchmarks/tiny/.

## Evidence boundary

The DSE-010/011 results are small synthetic NumPy pilots. No Rust compilation, FAISS performance, production false-accept rate, energy measurement, SOTA win or real-world error guarantee is claimed. See issue #8665 for next steps.
