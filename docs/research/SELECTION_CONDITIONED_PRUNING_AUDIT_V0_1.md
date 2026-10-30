# Selection-Conditioned Pruning Audit v0.1

**Status:** design and Rust source authored; not compiled, executed, or qualified.  
**Tracking:** [#9098](https://github.com/Luminous-Dynamics/symthaea/issues/9098).  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/pruning_audit.rs`.

## 1. Why another evaluator layer is needed

A multi-fidelity optimizer uses cheap measurements to decide which candidates deserve expensive evaluation. The usual diagnostics—global Spearman correlation, prediction error, or confidence-interval coverage—do not directly measure the harm caused by a specific pruning decision.

This is not a hypothetical concern. Phan and Luong's 2026 MF-NAS study on NAS-Bench-201 reports that a zero-cost proxy with Spearman correlation around 0.8 did not necessarily locate the best architectures; other lower-correlation proxies sometimes returned better candidates. Their analysis also shows that proxy-guided local search can pass high-performing regions and later converge to weaker proxy optima. The operational target should therefore be selection quality and candidate survival, not correlation alone.

Reference: [Phan and Luong, Robust and Efficient Multi-Fidelity Neural Architecture Search with Zero-Cost Proxy-Guided Local Search (2026)](https://doi.org/10.1145/3774940).

## 2. Paradigm shift: make pruning falsifiable

Each pruning step asserts that the discarded population is sufficiently unlikely to contain candidates that would beat the selected cohort at the next fidelity.

The proposed control loop is:

1. Score the current cohort at fidelity (F_i).
2. Produce the proposed selected and pruned sets using the frozen search policy.
3. Before observing any (F_{i+1}) audit outcomes, sample a uniform subset of the pruned set without replacement using a committed seed.
4. Evaluate both the promoted cohort and the random audit sentinels at exactly the same target fidelity (F_{i+1}).
5. Let the selection frontier be the lowest target-fidelity score among the promoted cohort (maximize-is-better convention).
6. Count audited sentinels whose target-fidelity score is strictly above that frontier.
7. Invert the finite-population hypergeometric distribution to obtain a one-sided upper confidence bound on the number of false-pruned candidates in the complete pruned population.
8. Continue pruning only when the pre-registered minimum sample size is met and the upper bound is within the risk budget. Otherwise disable pruning at that rung.

The Rust API now binds the sample to a private receipt containing the seed, the sorted complete pruned population, the sampled IDs, and domain-separated SHA-256 digests. A persisted receipt can be reconstituted only by replaying the declared seed against the complete population and matching the exact sampled-ID sequence. The audit rejects duplicate, missing, unexpected, or out-of-sample outcomes; it requires exactly one target-fidelity result for every committed sentinel. The report contains only aggregate statistics and the receipt digests, not sentinel identities or raw scores.

The sample generator is deterministic and uses a seeded SplitMix64 stream with rejection-based bounded draws. It is reproducible, not a cryptographic random-number generator. The evaluator must choose/commit the seed independently before candidate outcomes are visible and protect raw audit outcomes from the search policy. The integrating service must preserve this information boundary; the Rust module alone is not process isolation or cryptographic access control.

## 3. Finite-population risk bound

Let:

- (N) be the number of candidates in the pruned population;
- (n) be the number of uniformly sampled audit sentinels;
- (x) be the number of sampled sentinels that beat the selected frontier;
- (K) be the unknown number of candidates in the entire pruned population that would beat the frontier.

Under simple random sampling without replacement,

[
X sim \operatorname{Hypergeometric}(N,K,n).
]

For confidence level (1-\alpha), compute

[
K_U = \max\left\{K \in [x,N]:
\Pr(X \leq x \mid N,K,n) \geq \alpha\right\}.
]

Then the conservative upper bound on the false-pruned fraction is

[
p_U = K_U/N.
]

This is a discrete, conservative one-sided bound for the stated finite-population sampling design. It is preferable here to applying a binomial/Wilson interval mechanically to a sample drawn without replacement. The inversion is computed in log space to reduce underflow.

The gate continues pruning only when both conditions hold:

- (n \geq n_{\min}), the pre-registered minimum number of audit sentinels;
- (p_U \leq \tau), the pre-registered maximum false-pruned fraction.

Otherwise the disposition is fail-closed: disable pruning for that rung. The caller must implement the fallback (for example, retain the full cohort or apply a separately preregistered safe policy) and charge its actual compute cost.

This bound concerns the *fraction of the discarded population* that beats the observed selection frontier. It does not certify retention of the global optimum, directly bound the probability that at least one optimum was discarded, or guarantee safety under workload shift. Top-1/top-K survival and held-out challenge performance remain separate endpoints.

## 4. Cost accounting

Audit evaluations are not free. For a resumable cumulative fidelity ladder, the audit cost for (n) sentinels promoted from (F_i) to (F_{i+1}) is

[
C_{audit}=n(C_{i+1}-C_i).
]

The helper rejects invalid ladders, terminal rungs, zero audit counts, and integer overflow. The difference formula is valid only when the higher-fidelity evaluation resumes the same cumulative computation. If promotion restarts training, changes hardware, or evaluates a distinct workload, charge the measured actual audit cost instead.

Comparisons must use the same total budget, including:
- all low- and higher-fidelity evaluations;
- audit sentinel evaluations;
- failed runs, timeouts, OOMs, and retries;
- fallback evaluations after a gate failure.

Do not compare an audit-enabled method against a baseline that is charged only for successful promotions.

## 5. Initial Rust primitive

The first code increment provides:

- canonicalized candidate populations and a private sample receipt with seed, population digest and sample digest;
- receipt verification by deterministic replay, preventing caller-supplied cherry-picked sample IDs;
- deterministic, seeded uniform sentinel sampling without replacement;
- validation of duplicate/empty IDs, selected/pruned overlap, out-of-population sentinels, mixed target fidelities, non-finite scores, and invalid risk limits;
- selection-frontier comparison at the same target fidelity;
- exact finite-population hypergeometric upper-bound inversion;
- aggregate-only audit report with three dispositions;
- exact one-outcome-per-sentinel enforcement, preventing silent omission of failed or missing audit results;
- explicit cumulative-cost audit accounting.

Fifteen Rust unit tests are authored for receipt replay, deterministic sampling, uniqueness, census behavior, monotone bounds, false-prune detection, insufficient-sample fail-closed behavior, mixed-fidelity rejection, invalid IDs/scores/configurations, and cost overflow. They have not been compiled or executed.

Authored unit tests cover reproducible sampling, uniqueness, census behavior, monotone bounds, false-prune detection, insufficient-sample fail-closed behavior, mixed-fidelity rejection, invalid IDs/scores/configurations, and cost overflow.

## 6. Qualification experiment

The primitive should be integrated into a paired multi-fidelity benchmark before any performance claim. Compare at equal *total* cost:

1. Full-fidelity random search.
2. Successive halving / Hyperband with the implementation and settings pinned.
3. MF-NAS or a documented faithful reimplementation where the artifact is available.
4. A pinned TPE or Bayesian multi-fidelity baseline.
5. The existing static promotion policy.
6. The same policy with randomized pruning audits and fail-closed fallback.

For each landscape, seed and budget, report:
- best full-fidelity objective and simple regret versus cumulative cost;
- top-1 and top-10 oracle retention where the synthetic oracle is known;
- empirical false-pruned fraction and the finite-population upper bound;
- audit cost as a share of total cost;
- sample-size shortfalls and no-pruning fallback rate;
- invalid/failed evaluations and retries;
- per-landscape and pooled paired effects with uncertainty intervals.

Required negative controls:
- rank-preserving proxy;
- strong rank reversal;
- high global correlation but deliberately poor top-tail recall;
- heteroscedastic and under-reported uncertainty;
- shuffled candidate labels;
- non-uniform or post-outcome sentinel selection;
- target-fidelity mismatch between selected and audited cohorts;
- audit outcome leakage into search-policy inputs;
- dropped failed evaluations or uncharged retries.

The random sample must be selected and frozen before its target-fidelity outcomes are read. The benchmark should hash the candidate population, selected IDs, pruned IDs, sample seed, sampled IDs, fidelity metadata and cost receipt. The evaluator may read raw audit records; the search policy should receive only the aggregate gate disposition.

## 7. External benchmark and industry comparison

The 2025/2026 analytical multi-fidelity benchmark framework by Mainini et al. is a useful external mathematical suite because it spans high dimensionality, multimodality, discontinuities and noise. It should be a separate qualification target; this pruning-audit primitive does not reproduce that suite.

Reference: [Mainini et al., Analytical benchmark problems and methodological framework for the assessment and comparison of multifidelity optimization methods](https://doi.org/10.1007/s11831-025-10392-8).

NAS-Bench-201 remains an immediate public validation target for queryable architecture outcomes. It is also the space used by the cited MF-NAS study, enabling a direct pinned comparison if licenses, artifact versions and exact evaluation APIs are verified. At least one additional held-out search space is required before claiming transfer.

## 8. Claim ceiling

The current code is a research primitive, not a trained architecture searcher and not a benchmark result. The tests are authored but must still compile and pass at an exact commit. No claim is made that the audit gate is more accurate, faster, cheaper or better than Hyperband, MF-NAS, TPE, Bayesian optimization or any production NAS system.

The intended advance is methodological: replace unexamined pruning assumptions with randomized, cost-accounted, selection-conditioned evidence and an explicit fallback when the evidence is insufficient.
