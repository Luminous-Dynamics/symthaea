# HME global interval selection and selective exact refinement

**Date:** 2026-11-08  
**Status:** Algorithmic research protocol with synthetic theorem fixture; no runtime implementation or benchmark is claimed  
**Parent:** #10915 — HDC-native Holographic Memory Engine  
**Related owners:** #10917, #10918, #10945, #10973, #10978, #11003, #11022

## 1. Motivation: refinement-centered retrieval

Yang et al., "Recast: Refinement-Centered Approximate Nearest Neighbor Search Optimized for Storage," arXiv:2611.02765 (2026-11-02), present a storage-oriented refinement-first ANN architecture:

- compact IVF/RecastQ encodings and interval metadata reside in memory;
- query processing calculates estimated distances and uncertainty intervals;
- QUIC identifies the minimum candidate set whose refinement can still matter;
- raw vectors are fetched from storage only for that set and receive exact distance evaluation.

The paper reports 1.56–2.01x QPS over its strongest evaluated baselines at 0.99 recall under matched memory budgets, 1.22–6.75x less external storage than evaluated systems, and batched update improvements of 3.10–37.27x over SPFresh/OdinANN. These are the authors' reported system results and are not HME results.

**Important guarantee qualifications:**

1. Recast's deterministic distance-residual inequality gives a distribution-free but potentially loose uncertainty bound.
2. Its tighter interval uses Assumption 5.2 (directional isotropy): normalized encoding residuals are modeled as uniformly distributed on the unit sphere. The paper explicitly says this is an analytical model and not necessarily imposed by the encoding.
3. Its probabilistic interval coverage is asymptotic in dimension and empirically tested at aggregate level; the paper reports approximately 68.16–68.71% coverage at z=1 and 95.43–95.85% at z=2.
4. QUIC's deterministic retention lemma is conditional on every interval containing its true distance.
5. Its Theorem 4.3 adds simplifying assumptions (conditional independent standard-Gaussian normalized errors, common uncertainty scale and evenly spaced upper bounds) and lower-bounds expected refinement recall within the already-probed IVF list union. It is not a deterministic corpus-wide exact-recall theorem.
6. Candidate selection inside probed lists does not by itself prove that unprobed IVF lists contain no global top-k items.

HME should adopt refinement-first storage layout and interval-driven selection while keeping deterministic certificate mode and probabilistic uncertainty mode separate.

Paper: https://arxiv.org/abs/2611.02765

## 2. New algorithm hypothesis: global interval select-and-refine

For an eligible corpus of N records, suppose every item i has a certified interval for the *canonical exact similarity*:

s_i ∈ [L_i, U_i]

where larger scores are better.

For 1 <= k <= N, define:

theta_k = kthLargest(L_1, ..., L_N)

and the refinement set:

C = { i | U_i >= theta_k }.

Fully score every item in C from the original payload using the #10918 canonical full-resolution score contract. Return the exact top-k among those exact scores, using the canonical stable-ID tie policy.

### Correctness argument

By definition of theta_k, at least k eligible items have L_i >= theta_k. Since each interval contains its canonical exact score, those k items have exact score s_i >= L_i >= theta_k.

For any excluded item j, U_j < theta_k. Since s_j <= U_j, its exact score is strictly below theta_k. It cannot outrank the k items whose exact scores are at least theta_k.

Therefore every exact top-k item is in C. Exact-scoring C and sorting under the canonical order returns the same top-k as the full exact scan.

This proof is deterministic **only if** each interval is genuinely conservative for the canonical score, every eligible record is covered, and all arithmetic used to derive and compare intervals preserves that conservatism.

Use strict exclusion U_i < theta_k. Equality remains in the refinement set unless the canonical stable-ID tie rule independently proves that the item cannot win.

### Smaller-is-better metrics

For a distance interval d_i ∈ [L_i,U_i], define theta_k as the kth-smallest upper endpoint and retain:

C = { i | L_i <= theta_k }.

This is the distance-minimization analogue of QUIC. Metric polarity must be typed; never reuse a similarity comparison with reversed inequality by convention alone.

## 3. Coverage boundary: no hidden ANN omission

Global interval select-and-refine is exact only over the set for which all potentially relevant records are covered.

For HME ExactOnly:
- scan the compact interval metadata for every eligible item in the pinned manifest, or;
- prove an admissible aggregate bound for every unopened segment/partition and include the unresolved segment if that aggregate bound can still cross theta_k.

An IVF/PQR/HNSW/LSH candidate list cannot silently define C. If any unprobed partition has no admissible bound, ExactComplete is unavailable. The planner may use an ANN index to order metadata or payload work, but cannot omit an uncovered eligible item.

Metadata-first rights/temporal eligibility (#11003) happens before payload access and before the relevant eligible-corpus threshold is established. Pin the manifest and policy generation. Excluded items must not influence caller-visible retrieval, and stale index entries cannot resurrect them.

## 4. Deriving deterministic ContinuousHV similarity intervals

Let a compact reconstruction be x_hat and a verified residual certificate establish:

||x - x_hat||_2 <= epsilon_x.

For a finite query q, define the real-arithmetic dot interval:

d_lo = q·x_hat - ||q||_2 epsilon_x  
d_hi = q·x_hat + ||q||_2 epsilon_x.

By reverse triangle inequality, the real-arithmetic candidate norm lies in:

n_lo = max(0, ||x_hat||_2 - epsilon_x)  
n_hi = ||x_hat||_2 + epsilon_x.

If ||q|| > 0 and n_lo > 0, enclose the cosine ratio by taking the min and max of all four endpoint quotients:

{ d_lo/(||q|| n_lo), d_lo/(||q|| n_hi),
  d_hi/(||q|| n_lo), d_hi/(||q|| n_hi) }.

Clamp the resulting real-arithmetic bounds to [-1,1] only after the enclosure has been established. If the query norm is zero, n_lo is zero, an input is non-finite/out-of-profile, or the enclosure is ambiguous, use [-1,1] (no pruning) or fail explicitly.

These formulae first enclose the mathematical real-valued cosine. The actual HME interval must also enclose the #10918 canonical f32 score algorithm, including its defined reduction order, multiplication/addition rounding, denominator and clamp behavior. A real-arithmetic inequality evaluated using ordinary f32/f64 is not an outward-rounded floating-point proof. Implement a separately qualified interval arithmetic path / conservative round-off envelope before claiming exactness.

Residual certificates require independent validation at build/admission time. A digest binds certificate bytes to payload/codebook/generation; it does not prove ||x-x_hat|| <= epsilon.

## 5. Probabilistic interval mode is separate

Recast's spherical-residual model is a useful approximate baseline. Under its directional-isotropy assumption, it derives an asymptotic uncertainty scale:

tau(q,x) = (2 r ||q_c|| ||e||) / sqrt(D)

and approximate interval coverage 2 Phi(z)-1.

HME may evaluate this as ApproximateComplete with an explicitly named model and empirical calibration, but:
- it cannot authorize deterministic ExactComplete pruning;
- the assumption must be tested by representation, query population, corpus generation and update regime;
- intervals that fail an independent coverage audit must be widened or the profile revoked;
- query-level failure events may be correlated across candidates; aggregate per-item interval coverage does not automatically imply top-k recall coverage;
- full-corpus candidate coverage remains a separate requirement.

The deterministic residual norm certificate is slower/tighter only to the degree supported by its verified bounds. The probabilistic interval can be a separate faster path when approximation is explicitly permitted.

## 6. Synthetic fixture and falsifiable experiment

Fixture: [hme_interval_select_refine_v0.json](fixtures/hme_interval_select_refine_v0.json).

It contains a seven-record synthetic similarity example with k=2. The fixture sets exact scores and intervals whose validity is assumed for testing the selection logic. It demonstrates:
- a kth-largest lower-bound threshold;
- safe exclusion where upper < threshold;
- retention of boundary equality;
- exact reranking of all retained items;
- stable score / ID ordering;
- incomplete coverage and invalid interval controls that must prevent ExactComplete.

This fixture validates the interval-selection theorem's finite example; it does not create or validate production residual certificates.

## 7. Required comparisons

A. Full exact scan of all original payloads.  
B. Progressive exact scan (#10945).  
C. Global deterministic interval select-and-refine.  
D. Hierarchical segment interval selection then per-record refinement.  
E. Recast-style probabilistic interval selection, only under ApproximateAllowed.  
F. PQR-Search candidate retrieval plus its own probability contract.  
G. Approximate ANN candidate order used only as a scheduler while all HME eligible records remain covered.

Report:
- exact top-k IDs/order/scores and false-prune count;
- interval width / lower-to-upper gap by score margin;
- refinement set size and full payload bytes/pages read;
- metadata/certificate/index bytes and codebook cost;
- fraction of candidates and segment payloads refined;
- query latency p50/p95/p99 and batch throughput;
- build, update, delete, compaction, recovery and cold-open costs;
- interval coverage by query and corpus stratum for probabilistic profiles;
- OOD and generation mismatch behavior;
- total fast-tier, cold-tier and replicated bytes.

## 8. Promotion gates

Project targets, not external standards or measured results:

1. Deterministic mode: 100% canonical exact top-k agreement and zero false pruning.
2. Every eligible ID is accounted for by a record interval or a valid aggregate unopened-segment interval.
3. Equality at the kth threshold is retained unless stable-ID dominance is proven.
4. Missing/stale/corrupt interval or payload metadata forces full exact fallback or UnavailableOrInvalid.
5. The production interval encloses the canonical f32 score, not merely a real-valued ideal cosine.
6. Probabilistic intervals never authorize ExactComplete.
7. Full payload bytes avoided must exceed compact metadata, certificates, codebooks, cache and branch costs on the named workload.
8. No p95/p99/throughput promotion without end-to-end measurements including refinement reads and updates.

## 9. Nonclaims

No Recast reproduction, HME interval selector, certified ContinuousHV quantizer, f32 outward interval proof, runtime performance result or SOTA win is claimed by this protocol. The fixture does not prove any production interval is valid. The probabilistic RecastQ interval and deterministic HME residual certificate are different guarantee classes.


### Additional generated selector-property check (2026-11-08)

Added [hme_interval_select_refine_sweep_v0.json](fixtures/hme_interval_select_refine_sweep_v0.json).

A deterministic xorshift32 generator (seed 0x5eed1234) produced 50,000 small integer-valued cases:
- N from 1 to 32;
- k uniformly chosen from 1 to N;
- exact scores in [-5,5];
- lower/upper interval slack independently chosen from [0,5], so every generated interval contains its exact integer score;
- stable tie ordering by score descending, ID ascending.

The global interval selector produced zero false-prune disagreements with full exact top-k across these 50,000 generated cases. Threshold equality occurred frequently and was retained.

This is a small-domain property sweep of the selection rule. It does not validate a production residual encoder, floating-point outward rounding, canonical f32 score containment, metadata freshness or whole-corpus eligibility coverage. It is not a formal proof and not a runtime benchmark.
