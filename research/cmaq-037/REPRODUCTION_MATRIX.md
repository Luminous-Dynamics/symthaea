# CMAQ-037 — PRM / CCTM faithful reproduction and frontier study

## Research question

Does the ONS-adaptive predictive-rank martingale provide a measurable detection-delay advantage over conditional conformal test martingales (CCTM) at the same anytime false-alarm budget, calibration inventory, score process, and horizon? How does that trade-off change with calibration sample size, alpha, serial dependence, and shift morphology?

CMAQ-035 established an internal comparison between fixed and ONS betting. CMAQ-037 adds the external CCTM baseline and reproduces the 2026 PRM paper before extending the experiment.

## Method arms

| Arm | Description | Role |
|---|---|---|
| CCTM | Faithful implementation of the paper's conditional conformal test martingale | Published baseline |
| PRM-ONS | ONS-adaptive order and dispersion predictive-rank martingale | Published adaptive PRM |
| PRM-fixed-4 | Fixed two-sided four-component order / dispersion portfolio | Fixed-bet baseline |
| PRM-positive-only | Fixed positive Order and Dispersion bets | Directional ablation |
| CS-risk | Anytime confidence-sequence monitor of a matched running/rolling average risk | Complementary estimand, not a like-for-like per-instance drift detector |

Do not implement CCTM by multiplying repeated marginal conformal p-values or by using a rolling empirical breach rate. Implement its actual conditional calibration and martingale construction from the reference paper/code.

## Frozen experiment matrix

| Factor | Values |
|---|---|
| Calibration block count | 19, 49, 99, 199, 499, 999 |
| Alpha | 0.01, 0.05, 0.10 |
| Horizon | 100, 500, 2,000 |
| Null replications | At least 10,000 per registered cell |
| Shift replications | At least 5,000 per registered cell, preferably 10,000 |
| Score distributions | Continuous and discrete/tied |
| Dependence | IID independent blocks; explicitly dependent negative-control streams |
| Features | Order, dispersion, upper/lower tail, center concentration, higher-order shape |

N=19 is only the mathematical finite-sample floor for a non-randomized 95% split-conformal quantile; it is not a claim that 19 clusters gives adequate power. Report effective closure-block count from CMAQ-036, never raw episode count.

### Shift families

- Positive and negative location shifts.
- Dispersion expansion and contraction.
- Upper- and lower-tail contamination.
- Center concentration and symmetric heavy tails.
- Multimodal and trimodal shape changes.
- Gradual, delayed and transient shifts.
- Autocorrelated / common-shock score streams.
- Orthogonal-shape alternatives designed to evade pre-registered rank features.

Any serially dependent cell must be labelled assumption-violating for PRM unless a valid dependence-aware reference argument is established. Do not report an IID exchangeability guarantee on it.

## Metrics

**Null:** probability of ever crossing the anytime threshold; exact or Wilson 95% binomial confidence intervals; average run length. No early alarm is a “win” under the null.

**Alternatives:** detection power by horizon; median / p90 / p95 detection delay conditional on detection; fraction undetected by horizon; paired detection outcomes on shared random streams; worst-family result.

**Systems:** throughput, p50/p95/p99 per-observation latency, allocations, peak memory, and full end-to-end ingestion-to-quarantine cost.

**Coverage / utility:** calibration-size versus detection-power frontier; false-alarm versus detection-delay frontier; per-family results, never just pooled averages.

## Reproduction protocol

1. Pin the paper version, reference implementation commit, compiler, OS, and exact dependencies.
2. Recover published simulator equations, parameter settings, seeds and CCTM setup; record any missing information explicitly.
3. Implement CCTM faithfully and write fixed example vectors against the paper/reference code.
4. Reproduce the paper's headline curves before adding CMAQ-specific shifts.
5. Use a shared immutable calibration reference and common-random-number deployment streams for all PRM/CCTM arms.
6. Randomize ties using the same valid mechanism for calibration and deployment; simulation RNG remains clearly non-cryptographic.
7. Pre-register all cell combinations, replication counts, quantiles and exclusions.
8. Store per-replication alarm times / no-alarm indicators plus seed, configuration and source hashes. Aggregate CSVs alone are insufficient.
9. Publish code, artifact hashes, confidence intervals and the full run command.

## Current internal CMAQ-035 result is not SOTA evidence

At n=99, alpha=0.05 and horizon 500, an internal 10,000-replication experiment found:
- Null alarm rates of approximately 3.4–3.7% across ONS and fixed-bet variants.
- ONS median delays were substantially lower on location shifts and dispersion contraction.
- Dispersion expansion showed a faster ONS median but a slightly worse p90/p95 tail.
- A higher-order trimodal alternative exposed low detection power for all current feature portfolios.

This is enough to motivate broader replication, not enough to claim reproduction of the PRM paper or superiority over CCTM.

## Prior art

- Kuang & Xia (2026), *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales* — predictive rank law, order/dispersion features, ONS betting and CCTM comparisons.
- Fang (2026), *Anytime Valid Conformal Prediction on Streaming Time Series* — confidence sequences for running / rolling average miscoverage; complementary target.
- WATCH (ICML 2025) — weighted conformal test martingales for covariate / concept shift diagnosis.
- Timans et al. (UAI 2025) — continuous monitoring of bounded risk violations under unknown shift with testing-by-betting.

CMAQ-037 is a fair reproduction and benchmarking programme, not a new statistical theorem.
