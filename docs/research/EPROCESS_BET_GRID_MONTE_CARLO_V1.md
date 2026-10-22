# Monte Carlo Calibration and Power of E-Process Bet Grids V1

Status: deterministic Rust Monte Carlo benchmark added to the task-risk research branch. It is not a live Symthaea control-loop change. The exact-head qualification must pass before reporting the measurements as Rust-run results.

## Hypothesis

For bounded binary risk loss (L_tin{0,1}) under the null
[
mathbb{E}[L_tmidmathcal{F}_{t-1}]leq epsilon,
]
a predeclared mixture of valid e-processes is still a valid e-process. The mixture's main practical advantage is not stronger false-alarm validity than a single valid process; it is the ability to distribute evidence across alternatives tuned to different risk increases without picking the best bet on evaluation data.

The existing three-bet grid [0.25, 0.50, 0.75] of (1/epsilon) emphasizes moderate-to-large alternatives. A wider grid [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95] includes more conservative bets, which can accumulate evidence with less volatility for risk increases just above the null bound. Whether the extra components are worth their cost is tested, not assumed.

## Benchmark design

- Risk bound: (epsilon=0.05).
- Family-wise budget: (delta=0.05) across five parallel monitors, threshold (K/delta=100) (log threshold (log 100)).
- Horizon: 300 binary observations per monitor; 1,000 cohorts per risk setting.
- Six Bernoulli risk levels: 0.01, 0.05, 0.075, 0.10, 0.20 and 0.40.
- Three policies on the exact same observation stream: a fixed 0.50 bet, a three-component mixture, and the wider seven-component mixture.
- Output: 30,000 raw stream records (one record per cohort/monitor/risk setting) with the full 300-bit event trace, event count, first event, each monitor's first alarm and final log-evidence; 18 summary rows (six risks × three methods).

Cohort-level family-wise alarm probability is the primary calibration outcome: for a cohort, did any of its five monitor instances alarm within 300 observations? The benchmark also reports monitor-level alarm frequency, detection delay among alarmed streams, event counts and 95% Wilson intervals for the cohort-level family-wise rate. Raw traces are retained so the shell qualifier can independently replay every fixed/mixture factor recurrence and verify first-alarm locations, not merely trust summary counters.

## Preliminary arithmetic check

A separate deterministic JavaScript replica of the same splitmix64 stream and factor recurrence produced these illustrative cohort-level alarm rates over 1,000 cohorts per risk level:

| True Bernoulli risk | Fixed bet 0.50 | Three-bet mixture | Seven-bet mixture |
|---:|---:|---:|---:|
| 0.010 | 0.2% | 0.2% | 0.2% |
| 0.050 (null boundary) | 2.5% | 3.2% | 3.1% |
| 0.075 | 7.2% | 10.7% | 44.0% |
| 0.100 | 17.9% | 43.6% | 98.9% |
| 0.200 | 97.2% | 100.0% | 100.0% |
| 0.400 | 100.0% | 100.0% | 100.0% |

These are exploratory recurrence-replica outputs, **not** yet qualified Rust benchmark results. They motivate the controlled experiment: the wider grid appears much more sensitive to small risk increases in this designed Bernoulli model, while its false-alarm rate at the null boundary was comparable in that seed panel. Monte Carlo frequencies are not proofs of the e-process guarantee; the guarantee follows from the bounded-loss conditional-risk null and the e-process construction. The Monte Carlo checks whether implementation and power trade-offs look plausible.

## Acceptance gates

- Both executions are byte-identical.
- Exactly 30,000 raw stream records and 18 summary rows; no duplicate risk/cohort/monitor tuples.
- Every 300-bit event trace matches the recorded event count and first event.
- The fixed-bet and both mixture evidence sequences are replayed independently; final log-values and first alarms must match the raw output.
- Summary false-alarm counts, Wilson intervals, per-monitor alarm rates, average detection time, average event count, and configured bet grids match independently reconstructed raw-stream results.
- At risk 0.01, all methods' empirical family-wise alarm rates stay under 2%. At risk 0.05, all methods' Wilson upper limits stay under 8% in this 1,000-cohort calibration panel.
- At risk 0.075, the wide grid's family-wise alarm rate exceeds the three-bet grid by at least 15 percentage points; at risk 0.10, by at least 20 points.
- At risk 0.20, each method detects in at least 95% of cohorts; at risk 0.40, at least 99%.

These numerical gates are regression thresholds for a deterministic benchmark, not universal statistical guarantees. The 5% conditional-risk null applies to a continuously maintained process over its declared observation lifetime; resetting and reusing the same delta creates new testing obligations. Each Monte Carlo cohort is an independent test replicate for estimating its risk-level alarm probability, not a claim that a deployment can repeatedly reset at no cost.

## Why this is a useful paradigm shift

Rather than selecting a betting rate from test outcomes or tuning one anomaly score to a benchmark, predeclare a small grid and evaluate its entire detection/calibration frontier. This makes the trade-off between extra alternatives, delay, and false alarms visible. It also separates two claims that are often conflated:

1. **Validity:** under the stated conditional expected-loss bound, a nonnegative e-process with threshold (K/delta) has an anytime false-alarm bound allocated over the (K) parallel monitors.
2. **Power:** a wider mixture may detect more alternatives, but that empirical advantage depends on the signal and must be tested on held-out conditions.

## Industry comparison and next step

The relevant sequential-risk reference is Timans et al., *On Continuous Monitoring of Risk Violations under Unknown Shift* (UAI 2025), which explores testing-by-betting, including fixed and adaptive betting approaches: [paper](https://proceedings.mlr.press/v286/timans25a.html), [reference code](https://github.com/alextimans/risk-monitor). This benchmark uses simpler predeclared fixed grids; it does not reproduce the paper's full methods or experimental tasks.

For runtime control, keep the controller benchmark comparisons separate from the pure Bernoulli power experiment:
- [SODA-MPC, L4DC 2025](https://proceedings.mlr.press/v283/contreras25a.html): conformalized OOD monitor and reachability fallback.
- [Robust OOD MPC via Conformalized System Level Synthesis, L4DC 2026](https://proceedings.mlr.press/v331/srinivasan26a.html): weighted conformal uncertainty and reachable-set optimization.

To claim improvement against those systems, the next step is to plug the best preregistered monitor into a pinned physical/high-fidelity control task; use disjoint development, calibration and final disturbance families; match false-alarm budgets; and report task reward, violation frequency/severity, false-safe exposure before fallback, no-certified-action rate, detection delay, runtime and uncertainty. Nothing in this scalar Bernoulli Monte Carlo establishes SOTA or physical safety.
