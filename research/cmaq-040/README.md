# CMAQ-040 — Restart-time-universal predictive-rank spectrum

## Research question

Can a monitor cover two unknowns at once?

1. **Alternative geometry:** a shift may appear in a narrow rank region, tail, center, or higher-order shape, rather than the hand-selected order/dispersion features.
2. **Change time:** a shift may begin after thousands of valid observations, so a detector that has paid a fixed-bet wealth cost throughout the null period can react slowly.

CMAQ-038 broadens the first dimension with a dyadic Haar basis over all predictive-rank buckets. CMAQ-040 adds a summable mixture over all possible restart times. The resulting composition is a candidate **two-axis coverage architecture**: rank-space contrasts × shift-time restarts.

This is a new PRM composition experiment. It is not a reproduction of the October 2026 parameter-free CCTM paper and does not inherit that paper's universal-portfolio regret theorem.

## Mathematical construction

For a fixed calibration reference of size n, the online rank R_t takes values in {1,...,n+1}. Let N_{t-1,j} be the number of previous online ranks in bucket j. Under the fixed-reference exchangeability null, the exact predictive law is

    pi[t,j] = (1 + N[t-1,j]) / (n + t).

CMAQ-040 constructs n deterministic dyadic Haar contrasts h_k, scaled to [-1/2, 1/2]. Each is fixed before deployment. For each feature k, direction d ∈ {-1,+1}, and fixed betting rate lambda ∈ {1/4,1/2,1}, define

    Z[t,k] = h_k(R_t) - sum_j pi[t,j] h_k(j)
    f[t,k,d,lambda] = 1 + lambda * d * Z[t,k].

The predictive centering gives E[Z[t,k] | past ranks] = 0. Since |Z[t,k]| <= 1 and 0 < lambda <= 1, every factor is nonnegative and has conditional expectation one. Each fixed (k,d,lambda) process is therefore a PRM e-process under the exact predictive-rank null.

For every possible start time s, create a copy that holds cash before s and then multiplies by the factors from s onward. Allocate its initial wealth using

    rho_s = ln(2) * (1/ln(s+1) - 1/ln(s+2)).

The unstarted cash at time t is

    r_t = sum_{s>t} rho_s = ln(2)/ln(t+2).

The weights telescope to one. For one feature/direction/rate tuple, its restarted mixture is

    T_t = sum_{s=1}^t rho_s * product_{i=s}^t f_i + r_t.

It starts at one. It is a normalized mixture of nonnegative e-processes, not a fresh unit of wealth added at each round. Under the null, the conditional expectation of the new active wealth plus residual cash equals the previous total wealth.

The implementation uses the exact scalar recurrence

    A_t = (A_{t-1} + rho_t) * f_t
    T_t = A_t + r_t,

where A_t is the aggregate wealth of all started copies. It stores ln(A_t) and uses stable log-add-exp arithmetic. Negative infinity is permitted for an account whose active wealth reaches zero; NaN and positive infinity are rejected. The final monitor is a fixed normalized mixture across all feature, direction and rate tuples. It quarantines permanently when the mixture reaches 1/alpha.

## Computational structure

For n calibration observations:
- n+1 predictive-rank buckets;
- exactly n nonempty Haar contrasts;
- two directions per contrast;
- three fixed betting rates;
- exactly 6n restarted component aggregates.

Each observation requires one prefix pass over the n+1 predictive bucket weights, followed by a constant number of operations per component. The update is O(n) time and O(n) state per monitor, independent of how many possible start times have elapsed. It does not explicitly retain the O(t) individual restart accounts.

This is deliberately not the PF-CCTM algorithm's continuous Dirichlet universal-portfolio integral or its FFT acceleration. The fixed rate grid is pre-registered, not continuously optimized, and should be tested for sensitivity to its chosen values.

## Paired executable probe

Run from the repository root:

    cargo +1.100.0 run --release --manifest-path research/cmaq-040/Cargo.toml --locked --bin restart_probe -- 1000 5000 20261127 4000 > cmaq-040-results.csv

Arguments are replication count, horizon, seed, and number of null observations before the shift. Defaults are 64, 1,000, 20,261,127, and 400.

The probe compares:
- cmaq038_fixed_spectrum: CMAQ-038's fixed-lambda multi-resolution Haar mixture;
- cmaq040_restarted_spectrum: the restarted mixture over the same frozen Haar basis, both directions and three rates.

Both methods receive identical ranks in each replication. The null generator is the exact Pólya-urn predictive-rank process. The delayed alternative switches after the selected null prelude to a constructed rank-grid trimodal distribution on one-based buckets 11, 51 and 91, with weights 5:6:5.

CSV records per-replication alarm times. Stderr summaries report alarm counts and median detection delay measured from the actual onset. The alternative is deliberately a synthetic rank-space diagnostic; it is not an ImageNet-C, Waymo, nuPlan, CCTM-paper or real-vehicle reproduction. The probe's deterministic SplitMix64 generator is non-cryptographic.

A small CI smoke probe only demonstrates executable operation and state-path behavior. It does not estimate null calibration or detection power.

## Deterministic tests

The crate tests cover:
- unit initial wealth and the expected 6n component inventory;
- telescoping start-time mass and residual cash;
- the neutral recurrence preserving exactly one unit of wealth;
- predictive centering under the finite-rank predictive law;
- nonnegative factors for every rank, direction and registered betting rate;
- transactional valid updates and invalid-rank rejection;
- counter-overflow rejection before state mutation;
- log-domain zero-mass handling;
- permanent quarantine and rejection of post-quarantine updates;
- positive late-start mass and its logarithmic restart cost.

## Validity and trust boundary

The anytime argument is conditional on all of the following:
- the fixed calibration reference and score-to-rank mapping are immutable and frozen before monitoring;
- calibration and online effective blocks satisfy the required exchangeability / independence assumptions;
- rank ties use a valid, pre-registered randomized tie protocol when the score distribution has atoms;
- R_t is actually the rank against the same fixed reference, not a rank against a rolling or updated reference;
- no omitted common causes invalidate the declared effective blocks.

CMAQ-040 accepts an already-qualified rank. It does not verify score receipts, ties, signatures, authority, calibration provenance, dependence-key completeness, common-cause closure or physical independence. CMAQ-036 closes declared keys; it cannot prove that two disjoint dependency inventories are physically independent. The software does not predict hazard, certify recoverability, or guarantee a fallback action.

The restart construction protects against one specific source of slow adaptation: a long pre-change period in which an always-active fixed bet accumulates wealth loss. It does not make undetectable alternatives detectable, remove the rank feature's blind spots, or avoid the logarithmic start-time prior cost. The fixed feature/rate mixture also dilutes wealth across alternatives.

## Relationship to the 2026 literature

- Kuang & Xia (2026), *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales* (arXiv:2609.00536): exact conditional predictive rank law, finite-sample anytime marginal type-I control, order/dispersion features, and a power ceiling at fixed calibration size.
- Liu, Prinster & Orabona (2026), *Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales* (arXiv:2610.17343): a three-asset CCTM portfolio, restarting universal portfolios, heavy-tailed start-time allocation, and a logarithmic shift-time penalty.
- Shaer et al. (ICML 2026), *Testing For Distribution Shifts with Conditional Conformal Test Martingales*: fixed-reference CCTM with finite-reference DKW correction and ONS betting.

CMAQ-040 borrows the telescoping heavy-tailed start-time allocation idea and composes it with PRM's exact predictive centering and CMAQ-038's feature basis. It does not implement the three-asset market, Dirichlet(1/2,1/2,1/2) universal portfolio, or the PF-CCTM test-martingale betting construction. Those must remain separate faithful baseline arms in CMAQ-037.

## Required benchmark before any superiority claim

Use paired streams, identical effective calibration blocks, alpha, rank construction, tie handling and horizon. Register:
- shift times 0, 100, 500, 2,000 and 8,000;
- calibration sizes 19, 49, 99, 199, 499 and 999 effective blocks;
- alpha 0.01, 0.05 and 0.10;
- horizons 100, 500, 2,000 and long late-shift horizons;
- positive/negative location, scale expansion/contraction, tail, center concentration, multimodal, gradual, transient and adversarial feature-orthogonal changes;
- at least 10,000 full-horizon null streams per cell and at least 5,000 alternative streams per cell.

Compare CMAQ-031 fixed-four, CMAQ-033 ONS, CMAQ-038 fixed spectrum, faithful PRM ONS/portfolio, faithful CCTM, faithful PF-CCTM and CMAQ-040. Report full-horizon ever-alarm probability with intervals, power at fixed horizons, median/p90/p95 delay from onset, missed-shift fraction, worst-family behavior, memory, and p50/p95/p99 latency. Do not hide null-prelude length, mixture mass, family-specific failures or no-detection outcomes.

The deployment scorecard must then evaluate whether an alarm happens before recoverability margin exhaustion, whether fallback completes in time, how often quarantine is spurious, and how much safe useful operation is preserved. Separate simulator evidence from hardware-in-the-loop and physical deployment evidence.

## Claim status

This crate is a candidate compositional monitor with a proof sketch and deterministic software tests. It is not yet an independent peer-reviewed result, a reproduction of PF-CCTM, a demonstrated improvement over SOTA, a safety certification, or evidence of physical deployment efficacy. Exact-head CI and large pre-registered experiments are required before stronger claims.
