# CropFoundry CF-022: synthetic constrained irrigation-policy harness

**Status:** implementation smoke harness; synthetic only. No measured crop performance, agronomic recommendation, or field water-saving claim is produced.

## Why this benchmark exists

Reviews of irrigation decision support identify recurrent gaps around field-scale validation, uncertainty quantification, physics-informed modelling, data availability, and evaluation that stops at prediction accuracy rather than operational outcomes:

- Umutoni and Samadi, *Application of machine learning approaches in supporting irrigation decision making* (2024), [Agricultural Water Management 294, 108710](https://doi.org/10.1016/j.agwat.2024.108710).
- *Evaluating decision support systems for precision irrigation and water use efficiency* (2025), [Digital Engineering 4, 100038](https://doi.org/10.1016/j.dte.2025.100038).
- *Uncertainty-aware irrigation scheduling based on probabilistic site-specific soil moisture predictions with SWIM2* (2026), [Agricultural Water Management 328, 110300](https://doi.org/10.1016/j.agwat.2026.110300).

CropFoundry should therefore evaluate decisions under resource and sensing constraints, not just whether the next soil-moisture value was predicted accurately.

## Implemented prototype

Run from the workspace root:

```bash
cargo run -q -p xtask -- cropfoundry-water-policy-demo \
  --out reports/cropfoundry/water-policy-synthetic-v0.json
cargo test -p xtask cropfoundry_water_policy
```

The harness runs 4 fixed scenario profiles × 3 policies × 42 time steps:

- `reference`: nominal sensor/forecast assumptions and a fixed water cap.
- `water_rationing`: tighter cap, drought onset and mild under-forecast.
- `heat_and_drought`: heat/drought interaction, tighter cap and larger demand under-forecast.
- `forecast_bias_and_sensor_dropout`: biased forecasts, drought/heat onset and periodic missing sensor observations.

Policies are:

1. `equal_share_calendar`: apply the same irrigation amount to each plot, capped by the per-plot limit.
2. `soil_threshold`: use held/observed storage to refill the greatest deficit toward a fixed threshold.
3. `uncertainty_aware_marginal_risk`: allocate in small increments according to estimated marginal reduction of the declared shortfall proxy. It adds a fixed conservative demand margin and raises plot priority when readings are stale; these are explicit heuristics, not statistically calibrated prediction intervals.

The receipt includes per-scenario applied irrigation, an overflow proxy, synthetic stress-exposure proxy, per-scenario percentage change against equal-share allocation, plus policy-level mean/median proxies and water totals. It also reports sensor-dropout fraction and daily budget violations. The executable fails if any policy exceeds a daily water cap or emits non-finite core metrics.


**Unit note:** per-plot storage/demand values are synthetic depth-like values. Global irrigation totals add depths over equally weighted toy plots and are labelled `plot-mm`; without plot areas, flow rates and hydraulics, these sums are not volumetric water use and must not be compared with farm meter readings.

## Mathematical scope and claim ceiling

This is an intentionally small decision-policy harness, **not a crop model**. It defines artificial plot storage, demand traces, critical windows, weather transitions, sensor error, forecast bias, and a squared shortfall score. Those quantities are transparent test fixtures, not calibrated water balance, evapotranspiration, root-zone dynamics, yield response, nutritional outcomes, or field biology.

The initial hard acceptance condition is zero daily water-budget violations and deterministic replay. A provisional research target is at least 10% lower median synthetic stress-exposure proxy than equal-share allocation across the declared scenario suite, with every adverse scenario reported. The harness reports results but does not silently re-label the 10% target as achieved, and it does not fail merely because a proposed research-performance target is not met.

A policy that wins this synthetic score has only passed an algorithmic test. A defensible agricultural claim requires, in order: a versioned crop/soil water-balance model and its calibration record; comparison with fixed-calendar, soil-threshold, ET-based and a suitable established crop-model baseline; leakage-resistant retrospective or hindcast evaluation; local sensitivity and uncertainty analysis; then preregistered, replicated field trials with metered water, crop outcomes, energy, labor, cost and environmental measurements. Safety, water-allocation rights and any local water constraints remain hard limits rather than soft optimization rewards.

## What this benchmark is designed to reveal

- Whether the policy accounts for the cost of scarce water instead of always filling all plots equally.
- Whether decisions degrade gracefully with biased forecasts and intermittent/stale observations.
- Whether hard resource caps remain satisfied under all tested conditions.
- Whether the more complicated policy gives measurable decision value over simple baselines.
- Whether the answer changes under stress weights, storage thresholds, forecast error, dropout patterns and scenario mix.

## Follow-on measurements

Before proposing field deployment, add independently calibrated soil-moisture / root-zone observations, weather forecast vintages with issue-time timestamps, metered applied water, irrigation hardware limits, crop-stage definitions, plot-level outcomes and an agronomist-reviewed soil-water budget. Report runtime on target hardware and total instrumentation, connectivity, power, maintenance and labor cost. Preserve all scenario definitions and policy settings before looking at results.

```text
Claim ceiling:
synthetic harness -> implementation and invariant tests
retrospective calibrated replay -> evidence of model/policy performance on that dataset
replicated field trials -> bounded real-world efficacy estimates
multi-site multi-season validation -> transfer evidence within the sampled domain
```
