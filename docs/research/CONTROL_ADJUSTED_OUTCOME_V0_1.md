# Control-Adjusted Outcome Contrasts v0.1

Status: Rust prototype and deterministic unit tests; not an executed empirical result.

## Why this step matters

A simple pre/post comparison can confuse a common time trend with degradation following an alarm. For example, the treated cohort's mean error might rise from 1 to 3 after a shift alarm, while an otherwise comparable reference cohort experiences the same rise due to a broad traffic surge, seasonal load, or a measurement change. Raw pre/post comparison reports deterioration; a contemporaneous comparison reveals that the observed change is not specific to the treated cohort.

This branch adds a difference-in-differences-style contrast to the delayed-outcome validator:

    adjusted worsening = treated pre/post worsening - control pre/post worsening

Positive values mean the treated cohort worsened more than the reference cohort under the configured outcome direction. The value is compared with a predeclared minimum material effect.

## Rust API

In symthaea-physics-bridge/src/delayed_outcome_validation.rs:

- DelayedOutcomeMonitor::summarize_observed_window(start, end) summarizes a fully observed event-time window without requiring an alarm. It uses only labels received by the current cursor and returns label coverage explicitly.
- compare_controlled_outcome_change(...) accepts a resolved treated assessment, control pre/post summaries and the frozen config. It requires the same exact time windows and expected window length in all four summaries.
- The comparator checks configured label count and coverage gates on the treated and control windows, rejects inconsistent coverage metadata and refuses unresolved treated assessments.
- ControlledOutcomeAssessment exposes raw treated worsening, raw control worsening and the adjusted contrast. The raw values remain visible rather than being hidden behind one score.
- New unit tests cover a shared trend that disappears after adjustment, an incremental treated-only deterioration that remains, and fail-closed behavior for mismatched windows and inadequate control labels.

Targeted command:

    cargo test -p symthaea-physics-bridge delayed_outcome_validation --lib

The code and tests are committed, but this command has not yet been executed in the authoring environment. Exact-head Actions remains required.

## Interpretation boundary: this is not automatically causal

The arithmetic is descriptive. Interpreting the adjusted contrast as a causal effect requires a defensible comparison cohort and assumptions that cannot be proven by this calculation alone, including plausible parallel untreated trends, no material spillover/interference between cohorts, consistent outcome measurement, and sufficiently comparable label missingness.

The first prototype has no estimator for uncertainty, no automatic control-group selection, no trend diagnostics, no seasonality model, and no correction for differential label censoring. It should not yet be described as causal impact estimation. If the control itself is affected by the intervention, or if treated/control composition changes after the alarm, the adjusted result may be misleading; publish the raw and adjusted values and preserve the limitation.

## Required experiments

Start with three deterministic controls:

1. **Shared-trend null:** treated and control both change by +2 in the worse direction. Expected adjusted contrast: zero, even though the treated raw pre/post comparison looks worse.
2. **Incremental harm:** treated worsens by +4 while control worsens by +2. Expected adjusted contrast: +2 and Deteriorated when the minimum effect is below 2.
3. **Control contract violations:** insufficient labels, inconsistent coverage fractions, mismatched event-time windows, missing history and non-finite values must fail closed rather than return a reassuring no-change result.

Then broaden to seeded Monte Carlo streams with trend breaks, seasonality, group composition changes, varying treatment intensity and non-random label delay. Freeze cohort definitions and effect thresholds before evaluating the test streams.

## Benchmarks and go/no-go gates

- Report raw treated change, raw control change, adjusted change, label coverage per cohort, unresolved fraction and decision latency separately.
- At a matched false-confirmation rate and label budget, compare raw pre/post, control-adjusted, and established outcome-monitoring baselines. Use paired seeds, confidence intervals and held-out domains.
- On the shared-trend null, test that the adjustment removes the planted shared mean change in deterministic fixtures, and report empirical false confirmation rates over independent noisy streams.
- On incremental harm worlds, report power and detection/confirmation delay by effect magnitude and delay regime; do not average away the zero-effect null.
- Run sensitivity analyses for alternative controls and pre-periods. A result whose sign changes under small defensible control/window perturbations must be labelled fragile.

The KDD 2026 drift framework recommends scenario coverage, time-aware metrics and leave-one-dataset-out tuning for detector evaluation; this addition extends the measurement layer, not the drift detector's detection power: https://arxiv.org/abs/2606.07789.

## Industry comparison and novel contribution target

Commercial monitoring suites already connect drift signals with downstream model performance or ground-truth evaluations. The defensible Symthaea research target is not merely to reproduce alerts. It is a composable, auditable stream contract that combines:

- explicit detector/probe-family coverage;
- online-only event-time receipts;
- delayed-label coverage and unresolved-state accounting;
- raw plus control-adjusted outcome evidence;
- identical-budget, held-out comparisons with baselines.

Whether that combination is more useful or cheaper than existing products remains an open benchmark question. This prototype is a first implementation of the comparison primitive, not an industry-leading result or a universal causal estimator.
