# HME group-level admissible envelopes for exact retrieval

**Date:** 2026-11-08  
**Status:** Research protocol and synthetic arithmetic fixtures; no runtime implementation or performance result  
**Owner:** [#11059 — HME-GROUP-BOUND-001](https://github.com/Luminous-Dynamics/symthaea/issues/11059)  
**Composes:** #10917, #10918, #10945, #10973, #10978, #11000, #11003

## 1. Research question

Can HME build a compact, immutable synopsis for a group of ContinuousHV records that proves *every* record in the group is below the current exact top-k threshold, allowing an entire page/group to be skipped before its individual payloads are fetched?

This is distinct from:
- #10978, which certifies a compressed reconstruction for each vector;
- #11000, which changes the order in which blocks of one candidate are evaluated;
- #10945, which owns the general candidate-level progressive exact scan.

This protocol tests a group/page-level certificate. It is not a claim that bounding boxes, vector approximation files, metric trees or branch-and-bound are novel.

## 2. Prior art and likely limitation

VA-files and partition/tree bounds are established high-dimensional similarity-search techniques. Weber, Schek & Blott (VLDB 1998) showed that partitioning/tree approaches may approach linear scan complexity as dimensionality increases and introduced the VA-file as a way to make sequential filtering efficient. Ball trees, VP-trees, cover trees and R/X-trees are also established bounds-based indexing approaches.

The relevant prior-art lesson is not "use a bounding box." It is that a theoretically valid bound may be too loose in high dimensions to justify its metadata, branching, cache and build costs.

The HME hypothesis is narrower: representation-aware HDC block grouping, norm-aware cosine envelopes and page-aligned synopsis storage may make group skipping beneficial for some corpus structures/dimensions without giving up canonical exact scores.

Reference:
- Weber, Schek & Blott, *A Quantitative Analysis and Performance Study for Similarity-Search Methods in High-Dimensional Spaces*, VLDB 1998: https://www.vldb.org/dblp/db/conf/vldb/WeberSB98.html
- Filtered Vector Search tutorial/survey, PVLDB 2025: https://doi.org/10.14778/3750601.3750700

## 3. Group synopsis

For each immutable group G, store per-coordinate extrema:

L_j = min(x_j for x in G)  
H_j = max(x_j for x in G)

Also store an interval containing the canonical candidate norm-squared for every member:

N_min(G) <= norm_sq(x) <= N_max(G)

The synopsis must bind:
- segment and group generation;
- representation, dimension and metric;
- #10918 score-contract ID;
- ordered member-ID range or a membership digest;
- L/H coordinate extrema;
- N_min/N_max;
- builder/verifier version;
- original payload generation/digest.

The synopsis is valid only for the pinned immutable member set. A segment update, group membership change or payload replacement invalidates it.

## 4. Real-arithmetic group cosine bound

For query q, the coordinate box is a superset of every x in G. Therefore:

U_dot(q,G) = sum_j max(q_j * L_j, q_j * H_j)

is an upper bound on q dot x for every x in the group. The max handles negative query coordinates correctly.

Let Q2 be the query norm-squared under the score profile and X2 be each candidate's norm-squared, bounded by [N_min,N_max]. For a nonnegative upper dot bound, the largest cosine occurs with the smallest denominator; for a negative upper dot bound, it occurs with the largest denominator.

At the real-arithmetic level:

if U_dot >= 0:
    U_cos = U_dot / sqrt(Q2 * N_min)
else:
    U_cos = U_dot / sqrt(Q2 * N_max)

If U_dot is positive and N_min is zero/uncertain, no useful finite bound follows; use the metric's universal maximum (1) or refine/score the members. A zero-vector member also follows #10918's declared zero-denominator behavior and must not be silently treated as a positive-norm point.

The runtime implementation must *not* directly evaluate these equations in ordinary f32/f64 and assume the result is conservative. It must produce an outward-rounded interval that encloses the canonical f32 score contract, including:
- the eight-coordinate reduction grouping;
- the ordered tail;
- product and sum rounding;
- norm accumulation and multiplication;
- sqrt and division;
- clamping and the near-zero denominator branch.

Prune the group only when the conservative score upper bound is strictly below the current exact kth score. Equality or interval overlap means continue.

## 5. Fine-grained refinement

A single global box may be too loose. Compare:

1. whole-group coordinate extrema;
2. fixed-size block extrema (e.g. 64/256/1024 coordinate blocks);
3. hierarchical blocks with lazy refinement;
4. whole-group box plus verified per-group norm-squared interval;
5. whole-group box plus #10978's per-vector residual certificate for survivors.

A predictor may choose which synopsis block to inspect first, but only the verified outward-rounded bound may authorize a prune.

## 6. Eligibility composition

A group synopsis cannot bypass #11003:
- eligibility policy/source-ledger generation is pinned first;
- query valid-time, known-time, scope, authorization, revocation, quarantine, forgetting and retention are resolved before payload access;
- if the group mixes eligible and excluded records, either build a separate verified synopsis for the eligible subset or refine to item-level metadata before payload access;
- excluded payload reads remain zero;
- a stale synopsis cannot resurrect excluded IDs or content.

Do not expose excluded membership, titles, scores or reason details to unauthorized principals merely because a group-level envelope was consulted.

## 7. Synthetic arithmetic controls

The machine-readable fixture is:
[hme_group_bound_envelopes_v0.json](https://github.com/Luminous-Dynamics/symthaea/blob/research/hlb-memory-engine-20261107/docs/research/fixtures/hme_group_bound_envelopes_v0.json)

It includes:
- a positive-cosine group that can be rejected in real arithmetic;
- a negative-dot group showing why the largest norm denominator is required;
- a zero lower-norm bound that disables useful positive pruning;
- a threshold-equality case that must continue.

These examples are arithmetic illustrations, not floating-point proofs or runtime tests.

## 8. Evaluation matrix

Dimensions: 2K–256K where admitted.  
Group sizes: 8, 16, 32, 64, 128, 256, 512 and physical-page-aligned groups.  
Corpora: 64–64K and larger only on a named host.

Data/query strata:
- random normalized;
- clustered, anisotropic and high-correlation;
- sparse/localized and periodic;
- near-duplicate / low kth-margin;
- adversarial coordinate-extrema;
- boxes whose extrema come from different group members;
- groups containing zero/near-zero norm items;
- held-out corpus/query distribution shift;
- eligibility-mixed groups and stale membership metadata.

Compare:
A. canonical full exact scan;
B. global coordinate boxes;
C. blockwise/hierarchical boxes;
D. boxes + norm intervals;
E. composition with #10978 residual certificates;
F. group-order hints without pruning (isolates scheduling);
G. group bounds with fixed order.

Report:
- exact top-k IDs/order/scores;
- false group-prune count;
- group bound slack relative to the true best member;
- candidates covered by a group prune;
- synopsis bytes per group and per item;
- build/verify time and metadata CPU;
- payload pages/bytes avoided vs synopsis bytes read;
- p50/p95/p99, throughput and concurrency fairness;
- cache residency, peak RSS, page faults and physical I/O only where measured;
- insert/update/rebuild/compaction/recovery cost.

## 9. Promotion criteria

Project targets, not industry standards:
- 100% agreement with the canonical exact oracle;
- zero false group prunes;
- zero excluded payload reads;
- every group and candidate reconciled against the pinned manifest/eligibility generation;
- demonstrated end-to-end Pareto improvement after synopsis storage, build, verification and query-time metadata cost;
- p95 and p99 reported separately;
- if boxes saturate or metadata dominates at high dimensions, retain exact scan and document the negative result.

## 10. Nonclaims

No group synopsis has been implemented. No speedup is measured. The arithmetic formula is not a finite-precision proof. The established prior art is explicitly credited. A production pruning implementation requires an independent interval-bound proof and adversarial differential tests before any ExactComplete result can rely on it.
