# CMAQ-057 — Predictable adaptation benchmark contract v0.1

Status: preregistration draft. No benchmark result is implied.

## Primary question

Can a single predictable adaptive betting strategy select representations from past outcomes and improve conditional paired monitoring without an explicit per-kernel alpha split, while retaining anytime false-alarm control? This is not assumed to eliminate the statistical representation-selection cost: compare its regret/delay against the ICML kernel-family mixture and post-hoc best-fixed-kernel oracle.

## Arms

1. CMAQ-050 fixed product-Fourier baseline.
2. CMAQ-051 fixed context/product mixture.
3. CMAQ-054 fixed kernel-family mixture.
4. CMAQ-055 fixed kernel × start-time mixture (within its 1,024 horizon).
5. CMAQ-057 lagged-scale single-bet policy.
6. Faithful SCPT reproduction, including its past-covariate KNN neighborhood selection and history-derived covariate-dependent betting fraction. Do not compare against a static-kernel/static-bet simplification.
7. ICML 2026 kernel-family betting, if its conditional-pair adaptation preserves the paper assumptions.

PF-CCTM scalar-reference and NeurIPS Online RFF-MMD unconditional two-sample experiments remain separate protocol comparisons.

## Null families

- Exact paired equality with changing predictable covariates.
- Stationary conditional equality with serial dependence.
- Predictable heteroskedastic conditional-pair martingale processes.
- Common-cause, duplicate pair-ID and sequence/replay negative controls.
- Current-observation adaptation/leak negative control, explicitly excluded from valid-test arms.

## Alternatives

- Low-, medium- and high-frequency conditional differences.
- Equal X/Y marginals with altered conditional pairing.
- Sparse covariate support and local changes.
- Oscillatory signs and alternating covariate contexts.
- Conditional variance/tail changes.
- Alternatives orthogonal to the finite feature inventory.

## Horizon and sample sizes

Initial horizons H in {128,256,512,1024}. Change points must satisfy t0<H. At least 10,000 independent full-horizon null paths and 5,000 alternatives per primary cell. Alpha in {0.01,0.05,0.10}. Misses count H+1.

## Primary metrics

- Full-horizon ever-false-alarm with exact binomial confidence interval.
- Unconditional power, including misses.
- Restricted delay and p50/p90/p95.
- Worst-family power regression.
- Policy selection trajectory and direction trajectory.
- p99 update latency, peak memory and feature count.
- Number of candidate transforms evaluated per observation.

## Correctness requirements

- Every map, direction and rate is selected before current pair values.
- SCPT's existing predictable neighborhood/betting adaptation is preserved in the baseline. CMAQ-057 is compared on the additional Fourier feature-selection policy, not on the generic fact of being predictable.
- Report restricted delay and regret relative to the best fixed expert; do not claim the one-bet policy eliminates the log(K) family-adaptation lower bound.
- No same-sample expert maximum is used as the test statistic.
- A single wealth process and one alpha_test are used.
- All candidate feature maps have the same direct conditional paired null and uniform norm bound.
- The current covariate may be used only if it is available before the pair.
- The fixed-mixture versus single-predictable-bet comparison must report the exploration/regret trade-off; absence of an alpha split is not evidence of zero statistical selection cost.
- Replay, sequence, feature and arithmetic errors fail without partial state commit.
- A null/Type-I simulation is not a substitute for the conditional-expectation proof.

## Prospective target

Seek at least 20% improvement in restricted mean detection delay on three registered alternatives versus CMAQ-054 at the same alpha, while maintaining the null upper confidence bound and no more than 2 percentage points of worst-family power regression. This is a target, not an achieved result.
