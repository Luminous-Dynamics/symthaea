# Exact sequential shift-monitor benchmark

**Status:** dependency-free synthetic reference task; no production detector or deployment claim.

## Question

Can the active-evidence selector detect a change in its assumed measurement channel while keeping false alarms controlled when the channel is stable?

This fixture fixes a nominal Bernoulli probability p0 = 0.5 and a post-change probability p1 = 0.8 after 100 observations. The monitoring threshold represents one of 12 model-adequacy cells, with per-cell alpha = 0.05/12. The output is exact for this finite Bernoulli process, not Monte Carlo.

## Compared monitor policies

1. fixed_time_final_only: evaluate a fixed-time Hoeffding interval once at the prespecified horizon. This is valid for one planned look; it cannot stop early.
2. fixed_time_reused_each_look: deliberately reuses the fixed-time interval after every observation and stops at the first alarm. This is a negative control for optional-stopping inflation, not a valid anytime monitor.
3. time_uniform_alpha_spending: checks an anytime-valid Hoeffding sequence with per-cell/time error mass alpha/[M*n*(n+1)]. The sum over all n and M cells is at most alpha under independent stationary Bernoulli streams.
4. beta_mixture_e_process: checks a predeclared Beta(1/2,1/2)-mixture likelihood-ratio e-process against p0. Ville's inequality controls the first crossing over all looks.

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
- The alternatives are single abrupt steps of varied direction and magnitude; gradual ramps, oscillation, serial dependence and compound shifts are not yet covered.
- Alpha is assigned to a cell; combining 12 concurrently monitored cells needs the stated multiplicity control.
- This monitor only indicates a mismatch with the nominal Bernoulli rate. It does not identify the cause, repair the predictive model, or prove that every shift is harmful.
- The fixed-time-reused arm is intentionally invalid when inspected repeatedly; it is included only to show why optional-stopping control matters.
- Current reported numbers are independent Python reference calculations. Rust compilation and Rust-versus-Python output parity remain subject to the exact-head workflow.


Metric interpretation: in the step-shift rows, false-alarm probability counts alarms at or before the change point; detection probability counts first alarms strictly after the change point. The reported mean delay is conditional on detecting after the change. In stationary-null rows, false-alarm probability is the probability of any alarm by the horizon, while detection probability and detection delay are NA. The three probabilities (pre-change false alarm, post-change detection, and survival) sum to one.


## Follow-up: Beta-mixture likelihood-ratio e-process

A fourth arm, beta_mixture_e_process, is now implemented. For Bernoulli observations with S successes in n trials and a fixed null rate p0, it computes the marginal likelihood ratio of a Beta(1/2, 1/2) alternative mixture against the null:

    log E_n = log B(S+1/2, n-S+1/2) - log B(1/2,1/2)
              - S log(p0) - (n-S) log(1-p0)

Under the iid Bernoulli(p0) null, E_n is a nonnegative likelihood-ratio martingale with expectation one. Ville's inequality bounds the probability that it ever crosses 1/(alpha/M) by alpha/M. The implementation therefore checks whether log E_n exceeds log(M/alpha); it does not estimate a fixed-time confidence interval and then reuse it. The alternative prior is fixed before seeing the stream. This is a different valid sequential test from the alpha-spending confidence sequence and can be more efficient for abrupt alternatives well represented by the mixture, though it does not guarantee the same power or delay for every alternative.

Independent standard-library Python exact-DP result for this fixture:

| Monitor | Null false-alarm probability by horizon 1,000 | Step-shift detection by horizon 500 | Mean detection delay after change |
|---|---:|---:|---:|
| Fixed-time final look | 0.000442 | 1.000000 | 400.00 |
| Fixed-time interval reused each look (invalid) | 0.009366 | 0.996183 | 78.82 |
| Time-uniform Hoeffding alpha spending | 0.000000451 | 0.999999604 | 157.50 |
| Beta-mixture e-process | 0.003059 | 0.997539 | 94.76 |

The Beta-mixture e-process remains below the per-cell false-alarm allocation alpha/M = 0.004167 in the exact null DP and detects the selected p=0.5 to p=0.8 step earlier than the alpha-spending Hoeffding baseline. The repeated fixed-time method is faster in this scenario, but its null false-alarm probability at horizon 1,000 exceeds alpha/M and it is deliberately not a valid optional-stopping method.

The calculation is exact for the finite Bernoulli state model, not for real-world shift detection. The Beta mixture's prior and log-gamma implementation have a deterministic small-case check (n=2, S=1, p0=0.5 gives e-value 0.5); the independent Python checker recomputes every output row and requires the Beta-mixture null / shift controls to pass. Rust tests and Rust/Python parity still require the workflow to complete at the exact current source head.

Methodological references:
- Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting*, JRSS B 2024: https://doi.org/10.1093/jrsssb/qkad009
- Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*, Annals of Statistics 2021: https://doi.org/10.1214/20-AOS1991


## Shift-magnitude sensitivity (added)

The exact DP now runs a grid of ten alternatives after the same 100-observation nominal period: p1 in {0.10, 0.20, 0.30, 0.40, 0.45, 0.55, 0.60, 0.70, 0.80, 0.90}. This checks both downward and upward changes, including weak alternatives close to p0, rather than relying only on the dramatic 0.5→0.8 demonstration.

Independent Python exact-DP results at horizon 1,000 (detection probability and conditional mean delay):

| Post-change p1 | Time-uniform Hoeffding: detection / delay | Beta-mixture e-process: detection / delay |
|---:|---:|---:|
| 0.10 | 1.0000 / 103.8 | 0.9975 / 65.2 |
| 0.20 | 1.0000 / 157.5 | 0.9975 / 94.8 |
| 0.30 | 1.0000 / 301.4 | 0.9975 / 167.9 |
| 0.40 | 0.3095 / 720.4 | 0.9428 / 487.2 |
| 0.45 | 0.0005 / 731.7 | 0.1184 / 639.2 |
| 0.55 | 0.0005 / 731.7 | 0.1184 / 639.2 |
| 0.60 | 0.3095 / 720.4 | 0.9428 / 487.2 |
| 0.70 | 1.0000 / 301.4 | 0.9975 / 167.9 |
| 0.80 | 1.0000 / 157.5 | 0.9975 / 94.8 |
| 0.90 | 1.0000 / 103.8 | 0.9975 / 65.2 |

The conservative Hoeffding sequence can be markedly slower / lower powered for moderate shifts. The Beta mixture improves detection power for p1=0.40 from about 31% to 94% by horizon 1,000, while remaining under the per-cell alpha allocation in the stationary-null experiment. For p1=0.45/0.55, both detectors struggle, though the Beta mixture is materially better. These results are symmetric for upward/downward shifts because p0=0.5 and the Beta prior is symmetric.

These are exact probabilities for the specified iid Bernoulli step-shift simulator—not a universal ranking. The fixed alternative mixture may be less efficient on gradual drifts, bursts, or shifts whose direction and temporal pattern differ from this setup. The next confirmatory matrix should include gradual ramps, temporary bursts, serial dependence and selection-biased observations, and compare false-alarm budgets at matched detection delay as well as detection at matched false-alarm rate.


A local numerical sanity check compared the Rust Lanczos-style log-gamma formula used by the Beta-mixture e-process against Python's libm-backed lgamma for every (n,S) with 1 ≤ n ≤ 1,000 and 0 ≤ S ≤ n. The maximum absolute log-e-value difference was about 3.64e-12; no crossing/non-crossing decisions differed at the fixed threshold log(240) on this enumerated state grid. This checks the numerical formula, not compilation or end-to-end Rust output parity.
