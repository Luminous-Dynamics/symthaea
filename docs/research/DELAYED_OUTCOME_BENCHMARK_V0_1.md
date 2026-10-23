# Paired Delayed-Outcome Benchmark v0.1

Status: deterministic contract/regression harness committed; not yet executed and not a SOTA comparison.

## What is tested

The integration test at tests/delayed_outcome_benchmark.rs replays nine paired settings. In each setting, the alarm index, pre/post window, label delay, and label-missingness schedule are held fixed between two worlds:

- **Benign covariate shift:** a feature alarm occurs, but the measured outcome remains stable.
- **Harmful outcome shift:** the same feature alarm occurs, and the measured outcome becomes materially worse at the alarm boundary.

The stream has 10 observations in each comparison window, an 80% minimum label-coverage requirement, and a predeclared minimum effect of 0.5. It tests delays of 0, 3, and 10 observations. Each delay is paired with label-missingness schedules representing zero, one, or three missing outcomes out of each ten-index window.

The harness expects:
- 6/9 benign worlds to resolve as NoMaterialChange;
- 6/9 harmful worlds to resolve as Deteriorated;
- 3/9 benign and 3/9 harmful worlds to remain InsufficientLabelCoverage;
- the deliberately naive policy “every feature alarm confirms harm” to falsely confirm all 9 benign worlds;
- confirmation only after the post-window observations are complete and both windows satisfy the label-count and label-fraction gates. Because the contract requires 8 of 10 labels, it may resolve before every label arrives; the event-time index is the later of window completion and the last label needed to meet coverage. Increasing delay never shortens confirmation time, but it need not add delay while window completion remains the limiting factor.

These values are **test expectations derived from the synthetic fixtures**, not observed benchmark results. They are not evidence of real-world utility, statistical power, causal attribution, or superiority over a production monitoring system.

## Why this is only Phase 0

The controlled pair separates two events that monitoring systems often conflate: change in observed features, and consequential change in the outcome. The test also checks that an outcome-aware workflow does not manufacture certainty under inadequate label coverage.

The harness does not implement or compare a full family of SOTA drift detectors, does not create a predictive model, and does not claim to measure model degradation in a real deployment. Its alarm is supplied by the test fixture to isolate downstream outcome semantics. The upstream challenge—detecting feature-quiet harmful drift—needs its own detector benchmark.

## Next benchmark phases

1. **Detector-only phase:** implement identical data streams for fixed-threshold, pooled self-probe, stratified self-probe and faithful established detector baselines. Evaluate on held-out seeded streams and the published KDD 2026 evaluation approach.
2. **Outcome phase:** feed each detector's event-time alarms into this downstream monitor. Hold evaluation-only outcome truth away from detectors and monitor.
3. **Delayed-label phase:** sweep independent fixed delay, heavy-tailed delay, label coverage and bursty arrival, including outcome-dependent missingness.
4. **External validity phase:** inject controlled shifts into real-world stream datasets, use leave-one-dataset/domain-out tuning, and publish paired confidence intervals across independent seeds.
5. **Deployment phase:** measure event throughput, end-to-end alarm-to-confirmation latency, memory, dropped/duplicate receipts, and operational label cost.

## Baselines and metrics

Use the same stream instances and resource budgets for all systems. Compare the full detector panel to the best individual detector and to appropriate label-aware and label-free monitoring baselines. Report feature-shift detection delay separately from outcome-confirmation delay.

Required outputs include false alarms per 1,000 observations, detection precision/recall, outcome-confirmation precision/recall, unresolved rate by decision deadline, label coverage at each decision, confirmation latency, CPU time and peak memory. Separate results by shift family and label-delay regime; aggregate scores must not hide unprobed families or unresolved events.

The KDD 2026 framework highlights controlled Monte Carlo shifts on real-world datasets, event-aware timing metrics, and leave-one-dataset-out tuning as safeguards against weak or leaky comparisons: https://arxiv.org/abs/2606.07789. The October 2026 survey on partially labelled and delayed streams makes label availability part of the experimental design rather than a footnote: https://doi.org/10.1016/j.dsp.2026.106325.

## Reproducibility

Run the harness with:

    cargo test -p symthaea-physics-bridge --test delayed_outcome_benchmark -- --nocapture

The focused GitHub Actions workflow is configured to run the test and Rust formatting check. The command has not been executed in the current authoring environment; pending, queued, skipped or failed CI is not a pass. No SOTA claim is permitted until the end-to-end detector benchmark and external baselines run on exact revisions with fully reported results.
