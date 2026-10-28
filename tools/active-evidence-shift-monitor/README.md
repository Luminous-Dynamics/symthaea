# Exact sequential shift-monitor benchmark

**Status:** dependency-free synthetic reference task; no production detector or deployment claim.

## Question

Can the active-evidence selector detect a change in its assumed measurement channel while keeping false alarms controlled when the channel is stable?

This fixture fixes a nominal Bernoulli probability p0 = 0.5 and a post-change probability p1 = 0.8 after 100 observations. The monitoring threshold represents one of 12 model-adequacy cells, with per-cell alpha = 0.05/12. The output is exact for this finite Bernoulli process, not Monte Carlo.

## Metric semantics are scenario-typed

Every scenario is assigned a preregistered class: `stationary_null`, `target_shift`, `selection_bias_control`, `baseline_misspecification`, or `within_band_transition_control`. The export schema carries this class explicitly so a statistical alarm is not automatically described as successful environmental-shift detection.

- **Target-shift rows:** `scenario_or_pre_shift_alarm_probability` is first-alarm mass at or before the change point; `post_shift_detection_probability` is first-alarm mass after the change point; `mean_post_shift_detection_delay` is conditional on a post-change alarm.
- **Stationary, selection-bias, and baseline-misspecification controls:** the first metric is the probability of any alarm by the horizon; post-shift detection and delay are `NA`. A stable selected stream can alarm without any target-environment change, and a stable rate outside a detector's declared null is assumption sensitivity rather than evidence of when a change occurred.
- **Within-band transition control:** the conditional rate changes from 0.49 to 0.51, but both values remain in the composite null band [0.48, 0.52]. The first metric is the *total* alarm probability over the horizon and the detection/delay fields are `NA`. Counting post-transition alarms as successful detection would reward a monitor for rejecting its own stated null.

The checker independently recomputes this typed schema and rejects mutated results. Probability-mass conservation remains checked internally for target-shift trajectories and the within-band schedule. This classification does not change the underlying alarm process; it corrects what the benchmark claims each alarm means.

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
    cargo run --locked --quiet --manifest-path tools/active-evidence-shift-monitor/Cargo.toml -- --sentinel-budget-csv > /tmp/sentinel-budget.csv
    cargo run --locked --quiet --manifest-path tools/active-evidence-shift-monitor/Cargo.toml -- --eprocess-family-csv > /tmp/eprocess-family.csv
    cargo run --locked --quiet --manifest-path tools/active-evidence-shift-monitor/Cargo.toml -- --cusum-baseline-csv > /tmp/cusum-baseline.csv
    python3 tools/active-evidence-shift-monitor/reference_check.py --self-test
    python3 tools/active-evidence-shift-monitor/reference_check.py /tmp/shift-monitor.csv
    python3 tools/active-evidence-shift-monitor/reference_check.py --sentinel-budget-self-test
    python3 tools/active-evidence-shift-monitor/reference_check.py --sentinel-budget /tmp/sentinel-budget.csv
    python3 tools/active-evidence-shift-monitor/reference_check.py --eprocess-family-self-test
    python3 tools/active-evidence-shift-monitor/reference_check.py --eprocess-family /tmp/eprocess-family.csv
    python3 tools/active-evidence-shift-monitor/reference_check.py --cusum-self-test
    python3 tools/active-evidence-shift-monitor/reference_check.py --cusum /tmp/cusum-baseline.csv

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


## Finite-horizon CUSUM comparator

A separate `--cusum-baseline-csv` lane adds direction-specific reflected Bernoulli CUSUM arms. Each arm increments the score by 12/25 for an evidence event and subtracts 13/25 for a non-event; for lower shifts, a failure is the evidence event. The 0.52 event-rate edge of the declared null interval therefore has zero expected score increment.

The threshold is the *smallest integer threshold* whose exact dynamic-program alarm probability over 1,000 observations is at most half the per-cell error allocation. The upper arm is calibrated at p=0.52; the lower arm is its symmetric failure-rate transform at p=0.48. For any predictable Bernoulli conditional rate inside [0.48, 0.52], each directional event probability is at most 0.52. A coupling argument makes the corresponding CUSUM alarm event no more likely than under the calibration boundary; Bonferroni across the two directions bounds the two-sided finite-horizon family alarm probability by the per-cell allocation. This calibration applies only through the preregistered 1,000-observation horizon; it is not an infinite-horizon guarantee.

The output reports both directions separately, with scenario class, calibration threshold, per-direction alpha, scenario/pre-change alarm probability, post-shift power, conditional delay, and survival. It does not combine directional powers into a fictitious two-sided score: joint detection power would require a separate joint-state calculation. Use this as a faithful first comparator, not a claim of superiority. For industry ranking, thresholds must ultimately be compared at matched false-alarm operating points against version-pinned CUSUM/EWMA/Page–Hinkley/ADWIN and distribution-sensitive detectors, as tracked in [issue #8818](https://github.com/Luminous-Dynamics/symthaea/issues/8818).

The independent Python reference reproduces the exact CUSUM dynamic program, checks the minimum calibration threshold, tests both directions and accepted-band schedules, and rejects mutated output. Rust compilation and exact-head CI status are reported separately from the independent Python calculations.

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

These are exact probabilities for the specified iid Bernoulli step-shift simulator—not a universal ranking. The fixed alternative mixture may be less efficient on gradual drifts, bursts, or shifts whose direction and temporal pattern differ from this setup. A follow-up matrix should add gradual ramps, temporary bursts, serial dependence, and compare false-alarm budgets at matched detection delay as well as detection at matched false-alarm rate. The current revision adds one further mechanism-isolation case: selection-biased observations with no target-population change.


A local numerical sanity check compared the Rust Lanczos-style log-gamma formula used by the Beta-mixture e-process against Python's libm-backed lgamma for every (n,S) with 1 ≤ n ≤ 1,000 and 0 ≤ S ≤ n. The maximum absolute log-e-value difference was about 3.64e-12; no crossing/non-crossing decisions differed at the fixed threshold log(240) on this enumerated state grid. This checks the numerical formula, not compilation or end-to-end Rust output parity.


## A selector can manufacture an apparent shift

The new `selection_bias_no_environment_shift` scenario tests a failure mode that the iid Bernoulli shift scenarios cannot expose. The target population has two equally common strata with event rates 0.8 and 0.2, so its target rate is p0 = 0.5. An adaptive acquisition policy samples the high-event stratum with probability 0.9 and the low-event stratum with probability 0.1. Its observed event rate is then

```text
p_selected = 0.9 * 0.8 + 0.1 * 0.2 = 0.74
```

The target population has not changed; the queried sample mixture has. A monitor that treats adaptively selected labels as a representative stream can therefore interpret the policy's own selection effect as environmental drift. This is a design/estimand problem: the classical e-process false-alarm guarantee assumes its observed stream really follows the declared null, which the selected stream no longer does. It is not a counterexample to Ville's inequality.

Exact dynamic programming at horizon 1,000 gives:

| Observation stream | Target-population change? | Observed Bernoulli rate | Beta-mixture alarm probability |
|---|---|---:|---:|
| Randomized sentinel from the target population | No | 0.50 | 0.003059 |
| Adaptive selected stream (90% high-event stratum) | No | 0.74 | 1.000000 |

The per-cell alpha allocation is 0.004167. The randomized-sentinel stream remains below it in the stationary reference scenario, while the selected stream triggers an apparent shift alarm with essentially unit probability despite no environment change. This exact result is intentionally stark and only applies to the defined two-stratum mixture.

**Paradigm shift:** separate *what changed in the world* from *what the acquisition policy chose to observe*. A detector should use a randomized sentinel/audit stream, known inclusion probabilities with a justified design-based estimator, or explicitly condition its null on the acquisition policy. Adaptive observations and random audits must be tracked as separate evidence channels. Sentinel labels and compute must remain charged to the acquisition budget; this fixture demonstrates the identification problem, not that a specific weighted or adaptive correction is already validated.

The Rust DP, independent Python reference checker, and negative control now include this scenario. They still need exact-head workflow qualification before this is described as Rust-verified evidence.


## 19. Sentinel budget versus drift detection

The second executable report makes audit cost an explicit dimension. It fixes a total budget of 1,000 labels and a target-population change from p=0.5 to p=0.8 after 100 total labels. For sentinel fractions q in {0, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0} and post-shift event rates p1 in {0.55, 0.60, 0.70, 0.80, 0.90}, labels assigned to sentinels are sampled from the target population on a deterministic, regularly spaced audit cadence; the rest can be assigned to the adaptive policy. The zero-sentinel row is emitted once, because a representative audit stream does not exist at q=0. The drift detector consumes only the randomized sentinel stream. This keeps the null interpretable instead of quietly treating policy-selected labels as representative.

The following values come from a separate exact-DP Python prototype of the same finite Bernoulli model. The Rust implementation and independent CSV reference checker now encode the same frontier, but exact-head workflow parity has not yet been established.

| Sentinel share of total label budget | Sentinel labels | No-environment-shift alarm probability | Detection after a genuine p=0.5 to p=0.8 shift by end of budget | Mean delay after shift, in total-label equivalents |
|---:|---:|---:|---:|---:|
| 0% | 0 | ~1.000000 (selection-only stream; spurious) | Not identifiable from a representative audit channel | — |
| 5% | 50 | 0.002118 | 57.69% | 614 |
| 10% | 100 | 0.002461 | 96.96% | 473 |
| 20% | 200 | 0.002711 | 99.85% | 285 |
| 25% | 250 | 0.002774 | 99.84% | 241 |
| 50% | 500 | 0.002938 | 99.79% | 148 |
| 100% | 1,000 | 0.003059 | 99.75% | 95 |

This table is the p1=0.80 slice of the full CSV grid. The important caveat appears when the shift magnitude changes:

| Post-shift event rate p1 | 10% sentinel: detection | 20% sentinel: detection | 100% sentinel: detection |
|---:|---:|---:|---:|
| 0.55 | 0.56% | 0.82% | 11.84% |
| 0.60 | 3.07% | 10.01% | 94.28% |
| 0.70 | 43.60% | 89.73% | 99.75% |
| 0.80 | 96.96% | 99.85% | 99.75% |
| 0.90 | 100.00% | 99.85% | 99.75% |

Detection is the probability of an alarm *after* the true shift and by the end of the label budget. The monitor can alarm before the shift; that is reported separately as a false alarm and removed from post-shift detection. This explains why post-shift detection is not strictly monotone in sentinel fraction: longer pre-shift exposure creates more opportunities for a pre-shift false alarm.

**Read this as a cost–power frontier, not a universal recommendation.** In this fixture, a 10% randomized audit budget is enough to achieve about 97% detection by the 1,000-label horizon for the specified abrupt shift, while a 5% audit budget detects only about 58%. More sentinel labels improve detection delay, but spend less budget on exploitation/acquisition. The 0% row has almost-certain apparent drift because the policy-selected stream has event rate 0.74 even when the target is stable. This example does not demonstrate that 10% is optimal for other tasks or weak/gradual shifts.

A subtlety: the “no-environment-shift alarm probability” for q>0 is the probability of an alarm on the randomized sentinel channel over its own number of observations; for q=0 it is the spurious alarm probability of the adaptively selected channel over the full budget. The genuine-shift column excludes alarms that occur before the change; that pre-change false-alarm probability is also emitted separately in the CSV. Delay is conditional on detection after the shift and converted to total-label equivalents by dividing sentinel-count delay by q.

This benchmark now allows the next experiment to sweep not only sentinel share but also change magnitude, change onset, stratum prevalence, misspecified audit inclusion probabilities, and rare/unsupported target groups. A later version should compare a random sentinel channel against valid design-based reweighting and policy-conditioned nulls; this release does not claim any of those corrections is already qualified.


## 20. A fixed portfolio of anytime-valid e-processes

The new `--eprocess-family-csv` report compares six predeclared monitoring strategies under the same Ville threshold (1/\delta), with per-cell (delta=0.05/12), horizon 1,000, and a step change after observation 100:

- Beta(1/2, 1/2), the existing Jeffreys mixture.
- Beta(1, 1), a uniform alternative.
- Beta(5, 5), a moderately concentrated alternative.
- Beta(20, 20), a local alternative concentrated near the nominal rate p0=0.5.
- Equal-weight portfolio of all four point-null e-processes, with weights fixed at 1/4 before observations.
- Composite-null band-betting process for the preregistered interval p in [0.48, 0.52], using an equal mixture of five fixed stakes {0.05, 0.10, 0.20, 0.40, 0.80} in each direction. Its guarantee covers any conditional Bernoulli rate that stays inside the interval; its interval is part of the null hypothesis, not a learned estimate.

A convex mixture of e-processes is again an e-process under the same null, so the portfolio retains the anytime-valid Ville guarantee. It does not choose the prior after seeing the observed shift. The CSV includes stationary-null and selection-bias controls, downward and upward step shifts spanning p1=0.10–0.45 and p1=0.55–0.90, and six stable streams whose actual event rate differs from the declared p0=0.5 by 0.01–0.05. This makes symmetry explicit for the p0=0.5, symmetric-prior fixture and adds a negative control for baseline misspecification. It records stationary alarm probability, false alarms before a true shift, post-shift detection probability, and conditional delay. The 19 scenarios across six families produce 114 rows.

An independent exact-DP Python prototype gave the following horizon-1,000 results at the fixed anytime-valid threshold:

| E-process alternative | Stable-null alarm probability | Detect p1=0.55 | Detect p1=0.60 | Detect p1=0.70 | Detect p1=0.80 |
|---|---:|---:|---:|---:|---:|
| Beta(1/2, 1/2) | 0.003059 | 11.84% | 94.28% | 99.75% | 99.75% |
| Beta(1, 1) | 0.003108 | 14.17% | 95.43% | 99.78% | 99.78% |
| Beta(5, 5) | 0.002667 | 19.39% | 97.13% | 99.90% | 99.90% |
| Beta(20, 20) | 0.001884 | 22.83% | 97.92% | 99.99% | 99.99% |
| Equal portfolio | 0.002778 | 17.93% | 96.75% | 99.84% | 99.84% |
| Composite band [0.48, 0.52] | 0.000213 | 1.77% | 73.53% | 99.9867% | 99.9867% |

These are exact finite-Bernoulli probabilities for the predeclared model and horizon—not real-world results or evidence that a prior or composite interval is universally better. The single-prior policies have different realized null alarm probabilities below the common theoretical bound; the equal portfolio raises weak-shift power versus the original Jeffreys monitor in this fixture. The composite-band e-process buys validity for every conditional rate in [0.48, 0.52], but is deliberately less sensitive to shifts just outside that interval: at p1=0.55 its post-shift detection is 1.77%, versus 11.84% for the point-null Jeffreys process. At p1=0.60 the composite process detects 73.53%; at p1=0.70 and p1=0.80 it detects approximately 99.99%. This is a robustness/power trade-off, not an overall winner.

**Important protocol distinction:** the family report uses the same Ville threshold for every policy, so each remains anytime-valid. It is not a matched-realized-false-alarm leaderboard; actual null alarm probabilities are emitted so the trade-offs stay visible. The next calibration experiment should compare power at matched empirical false-alarm budgets without calling finite-horizon-calibrated thresholds anytime-valid.

The Rust e-process family, exact dynamic program, independent standard-library Python reference, and negative controls now compare 114 rows (six families × 19 scenarios) in the branch. Their output still requires exact-head CI before being described as Rust-qualified.


### Nominal-rate misspecification is a separate failure mode

The e-process guarantee assumes that the declared nominal probability is correct. The new stable-baseline scenarios hold the observed process stationary but set its true rate to p ∈ {0.45, 0.48, 0.49, 0.51, 0.52, 0.55} while the monitor continues to use p0=0.50. This is a deliberately misspecified null, not evidence of a new environmental change. It must not be interpreted as a violation of Ville's inequality.

Independent exact-DP Python prototype results for the e-process family at the same log(240) threshold and horizon 1,000:

| Actual stationary event rate | Beta(1/2,1/2) | Beta(1,1) | Beta(5,5) | Beta(20,20) | Equal portfolio | Composite band [0.48,0.52] |
|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 21.33% | 24.67% | 31.47% | 35.33% | 29.49% | 4.53% |
| 0.48 | 0.90% | 1.10% | 1.55% | 1.73% | 1.36% | 0.089% |
| 0.49 | 0.40% | 0.44% | 0.48% | 0.44% | 0.45% | 0.032% |
| 0.50 | 0.31% | 0.31% | 0.27% | 0.19% | 0.28% | 0.021% |
| 0.52 | 0.90% | 1.10% | 1.55% | 1.73% | 1.36% | 0.089% |
| 0.55 | 21.33% | 24.67% | 31.47% | 35.33% | 29.49% | 4.53% |

The allocated per-cell error is 0.4167%. At p=0.48, all five point-null alternatives alarm more often than that amount; the composite-band process remains at 0.089% because p=0.48 is inside its declared null interval. At p=0.45 and p=0.55 the composite process alarms 4.53%, but those rates are outside its null interval, so this is an assumption-sensitivity result—not a per-cell false-alarm guarantee or a violation of Ville's inequality. The point-null methods face a misspecified null at p≠0.5, while the composite process is designed to protect the wider interval.

A stronger next method should separately model uncertainty in the reference rate—for example, by comparing a reference-stream confidence sequence with the monitored stream, or by a valid composite-null betting process whose guarantee holds over a prespecified p0 interval. The baseline and test regimes must be separate; estimating p0 and choosing the monitor threshold on the same stream would introduce another source of optimism. The present branch exposes this limitation but does not claim to solve composite-null monitoring.

The Rust e-process family, exact dynamic program, independent standard-library Python reference, and negative controls now compare 114 rows (six families × 19 scenarios). Their output still requires exact-head CI before being described as Rust-qualified.


### Composite-null assumptions and trade-off

The composite-band process is not a free improvement over the prior families. It encodes a wider null (all conditional rates p_t in [0.48, 0.52]) and therefore should not alarm merely because the data depart a little from exactly 0.50. It is the right comparison when the nominal rate has uncertainty of about ±0.02, provided that range is justified independently of the monitored stream. That robustness costs detection power for weak shifts. The implementation pins the interval and betting-stake grid before evaluation, logs both point-null and band-null outcomes, and includes controls at each interval edge and just outside it. A further control changes the conditional rate from p=0.49 to p=0.51 after observation 100, with both values inside the accepted band. Exact DP gives total alarm probability 0.0263% across the horizon (0.0164% before the transition and 0.0099% after), below the per-cell budget. A second null schedule alternates between p=0.48 and p=0.52 every observation; its total alarm probability is 0.02098%, also below budget. These tests validate a step transition and an alternating endpoint schedule; broader serial/adaptive stress tests remain outstanding.

Mathematical reference for time-uniform betting-based confidence sequences: Waudby-Smith and Ramdas, “Estimating means of bounded random variables by betting,” https://arxiv.org/abs/2010.09686; the paper develops composite nonnegative martingales and time-uniform inference for bounded observations. This is conceptual foundation, not a claim that this specific Bernoulli band implementation reproduces their exact method.