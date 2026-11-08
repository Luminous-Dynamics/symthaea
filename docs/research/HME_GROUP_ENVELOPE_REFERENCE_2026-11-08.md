# HME group-envelope reference implementation

**Date:** 2026-11-08  
**Owner:** #11059 — HME-GROUP-BOUND-001  
**Status:** in-memory Rust reference prototype; no persistence, page-I/O or performance claim

## 1. Objective

Move the proof unit from one compressed candidate to an immutable group of candidates. If a conservative interval encloses the canonical score of every eligible member in a group, and its upper endpoint is strictly below the current exact kth threshold, the entire group's original vector payloads can remain unread.

This is established branch-and-bound / VA-file-style bounding-box retrieval, not a claim of a new search primitive. The HME question is whether query-conditioned coordinate envelopes and HDC-specific data layout offer a useful Pareto point at 16K–256K dimensions. High-dimensional boxes can become uninformative; negative results are expected.

## 2. Source implementation

Core score interval:

- `symthaea-core/src/hdc/continuous_hv_interval.rs`
- `ContinuousHV::similarity_interval_full_resolution_for_component_bounds(lower, upper)`

Given coordinate ranges `x_j in [L_j,H_j]`, it propagates outward binary32 intervals through the #10918 canonical group-of-eight/tail cosine score, norm product, sqrt, near-zero denominator, divide and clamp. If any interval arithmetic is ambiguous, it returns [-1,1]. Inputs require finite query and candidate bounds in [-1,1] with lower <= upper.

Memory-layer group builder:

- `symthaea-memory/src/hme/group_envelope.rs`
- `ContinuousHVGroupEnvelope::from_original_members(manifest_generation, eligibility_generation, members)`
- `score_interval_for_snapshot(...)`
- `can_prune_for_top_k(...)`

The builder scans every original member and computes coordinate minima/maxima itself; callers cannot inject extrema. It rejects empty/duplicate IDs, mixed dimensions and invalid profile values. Members are sorted by stable ID. The BLAKE3 content identity binds:
- domain and score-contract ID;
- manifest and eligibility generations;
- dimension and sorted member-ID set;
- each member ID and every original f32 bit pattern;
- computed coordinate minima and maxima.

A separate member-set digest allows the query to compare the synopsis to the authoritative eligible-member set. The query path requires expected manifest generation, eligibility generation, member-set digest and content digest. A mismatch fails closed.

The prune condition is strictly:

`group_score_interval.upper < exact_kth_threshold`.

Equality remains unresolved. This function returns a Boolean group-prune decision, not an ExactComplete receipt; corpus-wide accounting must still show every eligible member covered by an exact result, an admissible group/record bound, or an explicit incomplete state.

## 3. Eligibility and generation safety

A group envelope must be built for the exact eligible member subset, not merely the physical segment membership. A principal/valid-time/known-time/revocation/forgetting filter can change the eligible subset. The caller must:
1. apply #11003 policy eligibility before payload work;
2. partition eligible members by compatible scope/temporal/rights generation;
3. build an envelope for that exact member subset;
4. compare the member-set digest and eligibility generation with the pinned snapshot before using the interval.

A mixed-eligibility group cannot be pruned under one member-set identity unless the authorized eligible subset has its own verified envelope. A checksum/digest binds bytes but does not prove coverage or authorization semantics.

## 4. Tests authored

The Rust source includes tests for:
- every member's canonical score being enclosed by the group interval;
- strict threshold equality (not pruned);
- manifest and eligibility generation mismatch;
- member-set mismatch;
- deterministic content digest independent of input ordering;
- duplicate IDs, dimension mismatch and non-finite member rejection.

Tests are authored but not run in this environment because `cargo` and `rustc` are unavailable. The core numeric kernel still requires compilation, test execution, cross-platform qualification and the floating-point assumptions documented in the canonical interval protocol.

## 5. Required comparisons

Compare on frozen payload and query manifests:

A. full original-payload exact scan;  
B. #10945 progressive per-candidate exact pruning;  
C. #11078 per-candidate certified residual intervals;  
D. one global coordinate box per group;  
E. blockwise coordinate boxes and coarse-to-fine interval refinement;  
F. group boxes plus per-record residual intervals;  
G. group ordering hints without group pruning (scheduler-only control);  
H. HdcStoreReader exact mmap scan for BinaryHV as a separate representation baseline.

Sweep:
- dimensions 2K, 4K, 8K, 16K, 32K, 64K, 128K and 256K where admitted;
- group sizes 8, 16, 32, 64, 128, 256 and 512, plus actual storage-page aligned groups;
- random normalized, clustered/anisotropic, sparse/localized, periodic, near-duplicate, low-margin, high-correlation, anti-correlation and adversarial-extrema groups;
- uniform and selectively filtered eligible subsets;
- cold/warm cache and single/batched/concurrent queries.

Report:
- exact top-k ID/order/score agreement and false-group-prune count;
- eligible records covered and group/member counts accounted;
- interval width and bound slack versus each group's true best score;
- groups/pages pruned, payload bytes/pages not read, synopsis metadata bytes read;
- synopsis build/verification bytes and time, peak RSS, allocations and cache misses;
- p50/p95/p99 latency, throughput, batch fairness;
- ingest/update/delete/compaction/recovery and rebuild costs.

Do not infer physical disk I/O from virtual page-residency arithmetic. Use OS/device telemetry for actual I/O claims.

## 6. Promotion gates

1. 100% exact canonical top-k agreement and zero false group prunes.
2. Every eligible member accounted for in each completed exact query.
3. Group membership and eligibility generation match the authoritative pinned snapshot.
4. No payload access for excluded records.
5. Stale/corrupt/mismatched envelope falls back to individual exact work or UnavailableOrInvalid.
6. p95/p99 and total bytes improve on at least one named workload after synopsis build, verification, metadata and update costs.
7. If coordinate extrema saturate in high dimensions or metadata cost dominates, retain exact scan and publish the negative result.

At least 50% cold-payload bytes avoided after total overhead remains a stretch target; it is not an observed result.
