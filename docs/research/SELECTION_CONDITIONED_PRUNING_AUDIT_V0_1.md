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

Under simple random sampling without replacement:

```text
X ~ Hypergeometric(N, K, n)
alpha = 1 - confidence_level
K_U = largest K in [x, N] with P(X <= x | N, K, n) >= alpha
p_U = K_U / N
```

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

## 5. Audit economics: a useful safety gate can be too expensive

A first-order cost calculation already exposes a serious risk: a statistically conservative audit at every rung may cost more than the work it is intended to protect.

Consider the existing illustrative ladder with 256 candidates and promotion schedule 256 -> 64 -> 16 -> 4. The cumulative costs are 1, 4, 16 and 64 units, so incremental promotion costs are 3, 12 and 48 units. At each rung, suppose we audit enough of the discarded population to bound its false-pruned fraction to at most 5% with 95% confidence, assuming the most favorable observation of zero audited candidates beating the selected frontier.

| Pruning step | Discarded population | Minimum audit sample | Audit fraction | Audit cost | Normal promotion cost | Audit / promotion |
|---|---:|---:|---:|---:|---:|---:|
| 256 -> 64 | 192 | 49 | 25.5% | 147 | 192 | 76.6% |
| 64 -> 16 | 48 | 30 | 62.5% | 360 | 192 | 187.5% |
| 16 -> 4 | 12 | 12 | 100% | 576 | 192 | 300.0% |

The calculation is a finite-population inversion of the same 95% one-sided hypergeometric bound used by the primitive. It is an arithmetic scenario, not an experimental result. It also assumes every audit succeeds with zero observed false-prunes; observed misses can require a larger sample or force pruning to stop. A failed audit may additionally trigger fallback evaluation costs.

Auditing every rung under this 5% criterion adds 1,083 units to the original 832-unit schedule, taking it to 1,915 units before any fallback work. That is more than the 832-unit full-fidelity-random reference budget. Therefore the audit cannot simply be bolted onto the current scheduler and assumed to improve efficiency.

**New qualification requirement:** every audit-enabled policy must be evaluated under one fixed total budget that includes sentinel runs and fallback runs. The controller must choose where audits have positive value, possibly auditing only early/high-leverage rungs or using a looser pre-registered risk limit at low-impact rungs. A risk threshold must never be relaxed after seeing results.

For any budget, compare three quantities together:
- the upper confidence bound on false-pruned prevalence;
- the incremental audit cost and fraction of the full budget it consumes;
- the resulting full-fidelity regret/quality under the same total cost as baselines.

The right research target is not “minimize false pruning at any cost.” It is “minimize final regret under a fixed resource budget while keeping pruning risk within a pre-registered ceiling.” A future budget allocator may use decision-theoretic value-of-information estimates, but must be benchmarked against static audit schedules and must charge every look/decision. Sequential or adaptive audits require an anytime-valid or pre-registered alpha-spending procedure; repeatedly checking ordinary fixed-sample bounds invalidates their nominal coverage.

## 6. Fixed-budget audit allocation

The naive alternative—impose the same false-prune tolerance at every rung—can spend more compute auditing than searching. The new module `src/budgeted_audit.rs` therefore introduces a *pre-outcome* fixed-budget allocator.

For each stage (j), the input contains the discarded population size (N_j), the actual marginal audit cost (c_j), and a pre-registered integer importance weight (w_j>0). For any proposed sample count (n_j), the planner computes (U_j(n_j)), the one-sided finite-population upper bound on the false-pruned count assuming zero observed misses. It solves this bounded integer program exactly:

```text
minimize    sum_j w_j * U_j(n_j)
subject to  sum_j n_j * c_j <= B
            0 <= n_j <= N_j, integer
```

This is a multiple-choice knapsack solved by dynamic programming, not a greedy marginal-value heuristic. It returns a deterministic schedule sorted by stage ID, total and unspent budget, each sample count/cost, each optimistic risk bound, and the weighted objective. Stage weights are pre-registered engineering-priority multipliers, not expected regret, probabilities, or values learned from held-out outcomes.

Crucially, the allocator has no access to audit outcomes. Its schedule can therefore be frozen before any sentinel is measured. The zero-miss bounds are an optimistic design objective only. After sentinel evaluation, the ordinary `audit_pruning` gate must use the *observed* number of misses and its pre-registered risk limit. A failed/missing sentinel or an over-limit bound disables pruning; the caller must run and charge its fallback. Do not infer a realized safety guarantee from the planned schedule.

The planner rejects invalid stage identities/sizes/costs/weights, invalid confidence levels, budgets above 50,000 cost units, and inputs whose conservative transition estimate exceeds 25 million DP cells. These explicit limits keep the exact solver bounded; a future sparse or Pareto-front implementation can extend scale after separate verification.

### Example schedule under a 1,083-unit sentinel budget

For the three discarded populations (192, 48, 12), per-sentinel costs (3, 12, 48), equal risk weights, and 95% confidence, an independent SciPy-backed reference calculation gives this exact-DP schedule:

| Stage | Sample count | Audit cost | Zero-miss upper bound |
|---|---:|---:|---:|
| Early | 121 | 363 | 2 / 192 |
| Middle | 30 | 360 | 2 / 48 |
| Late | 7 | 336 | 2 / 12 |
| **Total** | **158** | **1,059** | **Weighted sum = 6** |

The schedule leaves 24 units unspent because they cannot improve the integer upper-bound objective. Unlike the 5%-per-stage schedule (which consumes 1,083 units and audits the final 12/12 candidates), this equal-weight allocation optimizes the sum of upper-bound counts. It does **not** meet a 5% cap at every stage: the late-stage bound is 16.7%. This is intentional evidence that a global budget objective and per-stage hard safety constraints are different optimization problems. If a use case requires a per-stage maximum risk, add that as an explicit constraint and allow the planner to report infeasibility rather than silently relaxing it.

The allocator's objective is not a claim that the selected schedule minimizes downstream model regret. Benchmark it against uniform allocation, early-only allocation, per-stage threshold allocation and no audit under the same total resource budget. The primary experiment must include actual regret/top-K retention as well as risk-bound and audit-cost diagnostics.

### Allocation-policy schedule comparison

The independent SciPy-backed reference mirror compares the exact DP against two simple pre-outcome schedules on the same three-stage example. The reported objective is the unweighted sum of zero-miss upper-bound counts across the three discarded populations; audit costs are charged at 3, 12 and 48 units per sentinel.

| Audit budget | Exact DP risk bound / cost | Uniform count per stage: risk / cost | Early-only: risk / cost |
|---:|---:|---:|---:|
| 166 | 45 / 162 | 192 / 126 | 68 / 165 |
| 300 | 30 / 291 | 129 / 252 | 64 / 300 |
| 500 | 20 / 486 | 82 / 483 | 61 / 498 |
| 832 | 11 / 795 | 50 / 756 | 60 / 576 |

"Uniform" means the largest equal integer sample count at all three stages that fits the budget; "early-only" spends on the first stage until its population is exhausted. These are deterministic mathematical comparisons for this chosen cost/risk objective. They do not show that the DP finds better architectures, that the zero-miss assumption will hold, or that the selected audit schedule is optimal for realized regret. They do demonstrate why cost and stage populations should be solved jointly rather than applying one arbitrary sample fraction everywhere.



## 7. Initial Rust primitive

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

Thirty Rust unit tests are authored (17 for the pruning-audit primitive and 13 for the budgeted allocator) for receipt replay, deterministic sampling, uniqueness, census behavior, pinned independent hypergeometric reference values, monotone bounds, false-prune detection, insufficient-sample fail-closed behavior, mixed-fidelity rejection, invalid IDs/scores/configurations, and cost overflow. They have not been compiled or executed.

A one-off independent mathematical check used a Python log-domain mirror of the inversion and compared its upper-bound decisions with SciPy's hypergeometric CDF: 8,900 cases passed (all configurations for N <= 35 at 95% confidence plus 500 randomized cases up to N = 100,000 at 90%, 95% and 99%). This is independent formula-level validation only; it is not execution of the Rust implementation, is not part of CI, and should be converted into pinned reference fixtures/tests before qualification.

Independent mathematical validation remains separate from Rust execution: the hypergeometric inversion should also be checked against an independent high-precision/reference implementation over randomized larger populations, especially near confidence-bound transitions.

## 8. Qualification experiment

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


### Latest low-budget NAS baseline: ZAPS (September 2026)

[ZAPS: Zero-Cost Active Proxy Search for Neural Architecture Search](https://arxiv.org/abs/2609.14184) is a direct new baseline, not yet implemented in this branch. The September 12, 2026 arXiv preprint reports NAS-Bench-201 P@100 of 52.3% on CIFAR-10 and 65.8% on CIFAR-100 at 200 distinct architecture evaluations, with means over 200 seeds. It combines anti-redundant proxy selection (ProxyFit), hybrid K-means initialization, bootstrapped online proxy re-selection, topology features, and a proxy-augmented XGBoost ensemble with UCB acquisition.

The important metric shift is **P@100**: the proportion of the benchmark's actual top-100 architectures that the method truly evaluates. This is more decision-relevant than full-space Spearman correlation or predicted top-100, and makes the score comparable to rankers and non-ranking search methods. The paper reports that its top-100 capture differs materially from global proxy correlations and that removing structured initialization reduces P@100 by 15.6–22.6 percentage points at (B=200). These are the paper's reported findings, not Symthaea results.

The paper's offline ProxyFit calculation uses full-space proxy-to-accuracy correlation but also reports a cross-dataset transfer study. Our benchmark must distinguish an exact protocol reproduction (including its stated offline calibration) from a strict no-oracle variant whose proxy subset is calibrated only on development tasks and frozen before test tasks. Never let the active search policy inspect unqueried table labels. Include ZAPS alongside random search, local search, REA, BANANAS, TPE, fixed proxy baselines, and the audit-enabled variants. Pin source version, proxy artifacts, table digest, metric/split and baseline deviations before comparing.

Reference: [Touayouch, Najem and Benjelloun, arXiv:2609.14184](https://arxiv.org/abs/2609.14184), submitted 12 September 2026.

## 9. External benchmark and industry comparison

The 2025/2026 analytical multi-fidelity benchmark framework by Mainini et al. is a useful external mathematical suite because it spans high dimensionality, multimodality, discontinuities and noise. It should be a separate qualification target; this pruning-audit primitive does not reproduce that suite.

Reference: [Mainini et al., Analytical benchmark problems and methodological framework for the assessment and comparison of multifidelity optimization methods](https://doi.org/10.1007/s11831-025-10392-8).

NAS-Bench-201 remains an immediate public validation target for queryable architecture outcomes. It is also the space used by the cited MF-NAS study, enabling a direct pinned comparison if licenses, artifact versions and exact evaluation APIs are verified. At least one additional held-out search space is required before claiming transfer. For NAS results, make actual P@100/P@10/P@1 capture, best-found regret, and the distinct full-fidelity query budget the primary reporting set; predicted ranking correlation remains diagnostic only.

## 10. Claim ceiling

The current code is a research primitive, not a trained architecture searcher and not a benchmark result. The tests are authored but must still compile and pass at an exact commit. No claim is made that the audit gate is more accurate, faster, cheaper or better than Hyperband, MF-NAS, TPE, Bayesian optimization or any production NAS system.

The intended advance is methodological: replace unexamined pruning assumptions with randomized, cost-accounted, selection-conditioned evidence and an explicit fallback when the evidence is insufficient.
