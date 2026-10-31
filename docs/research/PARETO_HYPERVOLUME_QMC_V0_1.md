# Randomized QMC Hypervolume Estimator v0.1

**Status:** Randomized shifted-Halton estimator and ten Rust test declarations are committed on `research/pareto-hypervolume-qmc-v0.1`; compilation, formatting, execution and benchmark replay remain outstanding.

**Stack:** based on the hardware objective registry / Pareto reference work. Exact small-cohort oracle: [#9391](https://github.com/Luminous-Dynamics/symthaea/pull/9391). Multi-objective research issue: [#9389](https://github.com/Luminous-Dynamics/symthaea/issues/9389).

## 1. Purpose and claim boundary

The exact reference module computes two-objective hypervolume and finite-posterior Pareto regret for at most 16 candidates. It is valuable as a correctness oracle, but cannot be the volume kernel for 5–10 hardware objectives or large candidate sets.

This module estimates dominated hypervolume for 2..=16 normalized maximize-is-better objectives using randomized quasi-Monte Carlo. It supports arbitrary candidate counts, but runtime is proportional to:

`replicates × samples_per_replicate × candidate_count × objective_count`

It is a numerical volume estimator only. It does not yet select candidate portfolios, compute posterior Pareto regret, update a calibrated posterior, optimize actions, or enforce pruning risk. Those remain planner layers.

## 2. Estimator

For reference vector r, define the integration box B = product over dimensions [r_j, 1]. For a random point x uniform in B, define I(x)=1 when at least one candidate dominates x in every objective, otherwise zero.

`HV(P;r) = Volume(B) × E[I(x)]`

A Halton sequence is generated using the first 16 prime bases. Each replicate receives an independent Cranley–Patterson random shift drawn from a deterministic SplitMix64 stream seeded by the frozen run seed. The points are wrapped modulo one and mapped into B. The replicate hypervolume estimate is the box volume times the fraction of dominated samples.

The final estimate is the mean across independently shifted replicates. The reported standard error is the sample standard deviation across replicate estimates divided by the square root of the replicate count. The reported 95% interval is a normal approximation clipped to [0, Volume(B)].

This standard error is empirical randomized-QMC uncertainty, not a distribution-free confidence bound. The discontinuous dominance indicator can converge slowly in high dimension. A result must always include sample count, replicate count, seed, point-set digest, reference, and sampling-design digest.

## 3. Deterministic evidence bindings

- The point-set SHA-256 binds the canonicalized exact candidate vectors and reference point. Candidate input order does not affect this digest.
- The sampling-design SHA-256 binds reference, objective dimension, sample count, replicate count and seed.
- The same seed/config gives common random numbers for comparing nested or alternative point sets. With the same design, adding candidates cannot reduce the estimated hypervolume because every sample already dominated remains dominated.
- A different seed changes the sampling-design digest and should be treated as a distinct estimator run.
- The full estimate is reproducible from the points, reference and config. The sampling digest is not a substitute for retaining the source point vectors and config.

## 4. Supported configuration

- Objectives: 2 through 16, all normalized into [0,1], maximize-is-better.
- Reference: finite coordinates in [0,1); the integration box must have positive volume.
- Samples per replicate: 128 through 1,000,000.
- Independent scramble replicates: 8 through 256.
- Default: 4,096 samples × 16 scrambles, deterministic seed.

Points that fail to dominate the reference in any coordinate contribute zero. Non-finite/out-of-range values and dimension mismatches fail closed. Empty eligible point sets return exact zero volume and zero estimator standard error.

## 5. Reference checks

The independent Python mirror, using the same Halton bases, random shifts, SplitMix64 recurrence and sampling formula, produced:

| Fixture | Exact volume | QMC estimate | Between-scramble SE |
|---|---:|---:|---:|
| 3-D single rectangle [0.5,0.4,0.3] | 0.06000 | 0.05975 | 0.00025 |
| 2-D union [(0.8,0.3),(0.4,0.7)] | 0.40000 | 0.40016 | 0.00010 |

The exact 2-D result is computed by the parent oracle, which gives 0.40. The QMC estimator is compared against that value in a Rust test. Python results are an independent mathematical mirror, not a Rust build or test execution.

## 6. Test inventory

Ten Rust test declarations currently cover:
1. 3-D single-box volume;
2. 2-D result against the exact oracle;
3. order-invariant input vectors and digest;
4. monotonicity under candidate addition with common sampling design;
5. duplicate candidate invariance;
6. zero-volume points below the reference;
7. deterministic replay under a fixed seed;
8. seed-specific sampling-design digest;
9. invalid dimension/sample count/reference;
10. non-finite and wrong-dimension candidate vectors.

The environment currently has no cargo, rustc or rustfmt, so these tests remain uncompiled and unexecuted.

## 7. Next layers

1. Build a posterior portfolio policy that proposes a bounded collection of Pareto recommendation sets (e.g. NSGA-II, qEHVI/HVKG, preference slices).
2. Evaluate posterior expected portfolio hypervolume and regret over joint objective particles with common randomized-QMC designs.
3. Use identical sample designs for the prior and outcome-conditioned posterior to reduce variance in the VOI difference (common random numbers); independently replicate the design to quantify estimator noise.
4. Adapt sample count by confidence-width / ranking-stability requirements, not by an unrecorded fixed choice.
5. Keep Monte Carlo estimator error separate from posterior uncertainty, measurement noise, and task-transfer error.
6. Cross-check 2-D exact fixtures, 3-D analytic fixtures and small finite-posterior portfolio regret before adding any operational action selection.
7. Benchmark against qEHVI/qNEHVI, HVKG, LaMOO+qEHVI and FlexiBO under equal total measurement costs. BoTorch documents competitive decoupled evaluations and cost-aware HVKG; that is a strong systems baseline, not a claim that its Python implementation is a direct comparator to this Rust kernel.

## 8. Claim ceiling

This is a randomized numerical estimator with a reproducible error diagnostic. It is not exact hypervolume for arbitrary dimensions, not a rigorous simultaneous confidence interval, and not a scalable NAS algorithm by itself. No real benchmark result or better-than-SOTA claim is made.
