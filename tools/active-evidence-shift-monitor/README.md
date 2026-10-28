# Exact sequential shift-monitor benchmark

**Status:** dependency-free synthetic reference task; no production detector or deployment claim.

## Question

Can the active-evidence selector detect a change in its assumed measurement channel while keeping false alarms controlled when the channel is stable?

This fixture fixes a nominal Bernoulli probability p0 = 0.5 and a post-change probability p1 = 0.8 after 100 observations. The monitoring threshold represents one of 12 model-adequacy cells, with per-cell alpha = 0.05/12. The output is exact for this finite Bernoulli process, not Monte Carlo.

## Compared monitor policies

1. fixed_time_final_only: evaluate a fixed-time Hoeffding interval once at the prespecified horizon. This is valid for one planned look; it cannot stop early.
2. fixed_time_reused_each_look: deliberately reuses the fixed-time interval after every observation and stops at the first alarm. This is a negative control for optional-stopping inflation, not a valid anytime monitor.
3. time_uniform_alpha_spending: checks an anytime-valid Hoeffding sequence with per-cell/time error mass alpha/[M*n*(n+1)]. The sum over all n and M cells is at most alpha under independent stationary Bernoulli streams.

At observation n, the monitor alarms if the nominal p0 falls outside the confidence interval around the running empirical mean. The boundary only has a false-alarm guarantee when p0 is actually the stationary Bernoulli parameter and observations meet the sampling assumptions. Under the step-shift scenario, nonstationarity is the alternative; the measured detection power and delay are characteristics of this synthetic scenario, not a coverage guarantee.

## Exact dynamic program

The implementation tracks the probability mass of each success count among paths that have not yet triggered an alarm. Each Bernoulli branch is propagated exactly, with alarmed paths removed from future state. No random seeds or sampled trials are needed.

For each horizon (100, 250, 500, 1,000), report:
- cumulative false-alarm probability under the stationary null, or before the change in the step-shift scenario;
- post-change detection probability by the horizon;
- mean detection delay conditional on detecting after the shift;
- survival probability (no alarm by the horizon);
- per-cell alpha.

The output probabilities should sum to one. The null test checks that the single fixed-time look and the anytime sequence stay below alpha/M, while repeatedly reusing a fixed-time interval should exceed alpha/M by horizon 1,000. The step-shift test checks detection power and reports the delay/false-alarm tradeoff.

## Local run

    cargo test --locked --manifest-path tools/active-evidence-shift-monitor/Cargo.toml
    cargo run --locked --quiet --manifest-path tools/active-evidence-shift-monitor/Cargo.toml > /tmp/shift-monitor.csv
    python3 tools/active-evidence-shift-monitor/reference_check.py --self-test
    python3 tools/active-evidence-shift-monitor/reference_check.py /tmp/shift-monitor.csv

## Exploratory Python reference results

Exact dynamic programming in the independent standard-library prototype produced:

| Monitor | Horizon | Null false-alarm probability | Step-shift detection probability | Mean detection delay |
|---|---:|---:|---:|---:|
| Fixed-time look at final horizon | 1,000 | 0.000442 | — | — |
| Fixed-time boundary reused each look (invalid) | 1,000 | 0.009366 | 0.996183 | 78.82 |
| Time-uniform alpha-spending | 1,000 | 0.000000451 | 0.999999650 | 157.50 |

At horizon 500, the time-uniform monitor had pre-change false-alarm probability approximately 0.000000350, post-change detection probability approximately 0.999999604, and mean detection delay approximately 157.50 observations. The repeatedly checked fixed-time boundary was faster (mean delay about 78.82) but exceeded the per-cell alpha by horizon 1,000. The fixed-final-only test waits until the chosen horizon by design.

The alpha-spending method is extremely conservative here; that is a measurable engineering trade-off, not a reason to abandon anytime validity. A variance-adaptive empirical-Bernstein confidence sequence is a natural next comparison:
- Howard et al. (2021), *Time-uniform, nonparametric, nonasymptotic confidence sequences*: https://doi.org/10.1214/20-AOS1991
- Waudby-Smith and Ramdas (2024), *Estimating means of bounded random variables by betting*: https://doi.org/10.1093/jrsssb/qkad009

## Limits

- One Bernoulli cell and one known nominal probability are monitored.
- The shift is a single abrupt step, not gradual drift, oscillation, or a compound shift.
- Alpha is assigned to a cell; combining 12 concurrently monitored cells needs the stated multiplicity control.
- This monitor only indicates a mismatch with the nominal Bernoulli rate. It does not identify the cause, repair the predictive model, or prove that every shift is harmful.
- The fixed-time-reused arm is intentionally invalid when inspected repeatedly; it is included only to show why optional-stopping control matters.
- Current reported numbers are independent Python reference calculations. Rust compilation and Rust-versus-Python output parity remain subject to the exact-head workflow.
