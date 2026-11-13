# MCR² v16 — Intervention-Censored Event-Time Reference v0.1

A deterministic, dependency-free research reference for separating untreated natural-history event time (T0) from policy-specific outcomes (T-pi) when preventive interventions change the physical trajectory.

## Why this exists

If a robot intervenes before a predicted event, the untreated event time is not observed. It must not be silently scored as a normal right-censored observation, a correct prediction, or a prevented event. The reference preserves this distinction while keeping post-intervention outcomes available for their actual policy estimand.

## Contents

- intervention_censoring.py — typed event/censoring taxonomy, minimum censor-adjustment admission gates, simple randomized fixed-horizon intention-to-treat summary, cohort-identifiability warning.
- fixtures.json — 15 outcome-classification fixtures and 8 synthetic randomized safe-policy trials.
- fixtures.schema.json — strict JSON Schema.
- run_reference.py — emits report.json.
- test_reference.py — 12 regression tests.
- report.json — deterministic output.
- INTERPRETATION.md — result and claim boundary.

## Run

```bash
python3 run_reference.py
python3 -m unittest discover -v
```

Uses only the Python standard library. This is an executable protocol reference, not a survival estimator, causal estimator, production controller, or motor-control component. Statistical censoring adjustment reuses Symthaea #9690 and is not reimplemented here.

## Key rule

- Administrative follow-up censoring before intervention may be eligible for a separately qualified #9690-style censor-adjusted method, conditional on censor-model, support and positivity receipts.
- Preventive intervention before boundary makes individual T0 counterfactual-unobserved. It is not treated as ordinary independent censoring by default.
- A boundary event after intervention is observed for the actual policy, not the untreated natural-history path.
- Competing terminal events, telemetry loss, invalid clocks, instrumentation failure and unknown action delivery remain distinct.

## Synthetic fixed-horizon result

The eight randomized synthetic safe-policy trials produce 2/4 adverse outcomes under reactive_safe and 1/4 under predictive_safe, a descriptive candidate-minus-control risk difference of -0.25. This is an authored fixture demonstration only. It has no uncertainty interval, no population inference, no physical units and no SOTA meaning.

## Claim boundary

The reference does not establish calibration, independent censoring, positivity, causal identification, a treatment effect, an individual counterfactual, physical event prediction, safety, real-time performance, or actuation authority. No untreated physical hazard should be deliberately left unprotected merely to create a control group. Use only independently qualified safe-policy comparisons in physical trials; use simulation/bench fault injection for unsafe counterfactuals.
