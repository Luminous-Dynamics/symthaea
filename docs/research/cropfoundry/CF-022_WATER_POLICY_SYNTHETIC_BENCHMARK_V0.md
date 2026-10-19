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

The harness runs 6 fixed scenario profiles × 3 policies × 42 time steps:

- `reference`: nominal sensor/forecast assumptions and a fixed water cap.
- `water_rationing_only`: lower water cap with the same weather, forecast and sensor settings as the reference.
- `heat_and_drought`: heat/drought interaction and a tighter cap, without forecast bias or sensor dropout.
- `forecast_bias_only`: negative demand-forecast bias with reference water/weather/sensor settings.
- `sensor_dropout_only`: periodic missing soil observations with reference water/weather/forecast settings.
- `combined_heat_drought_forecast_bias_and_sensor_dropout`: compound stress, biased forecasts and periodic sensor dropouts.

Policies are:

1. `equal_share_calendar`: apply the same irrigation amount to each plot, capped by the per-plot limit.
2. `soil_threshold`: use held/observed storage to refill the greatest deficit toward a fixed threshold.
3. `uncertainty_aware_marginal_risk`: allocate in small increments according to estimated marginal reduction of the declared shortfall proxy. It adds a fixed conservative demand margin and raises plot priority when readings are stale; these are explicit heuristics, not statistically calibrated prediction intervals.

The receipt records the scenario parameters, per-scenario applied irrigation, overflow proxy, synthetic stress-exposure proxy, cumulative allocation dispersion, percentage change against equal-share allocation, and absolute/relative stress change versus the same policy in the reference scenario, plus policy-level mean/median proxies and water totals. It also reports sensor-dropout fraction and daily budget violations. The executable fails if any policy exceeds a daily water cap or emits non-finite core metrics.


**Unit note:** per-plot storage/demand values are synthetic depth-like values. Global irrigation totals add depths over equally weighted toy plots and are labelled `plot-mm`; without plot areas, flow rates and hydraulics, these sums are not volumetric water use and must not be compared with farm meter readings.

## Pinned synthetic scenario math (v0)

These equations describe test fixtures, not agronomy.

- Initial storage is 6.0 synthetic depth units per plot; storage is capped at 10.0. Each plot can receive at most 3.5 units/day. Allocations for threshold/risk-aware policies are in 0.05-unit increments.
- For day (d) and zone (z), the illustrative base demand is ([2.2, 2.7, 1.8]). The demand phase factor is (0.88 + 0.22(\sin(\pi d/42 + 0.7z)+1)/2). After the scenario's heat-onset day, demand is multiplied by 1.30.
- The exposed demand forecast is (max(0.1, D(d,z)(1+b+0.04\sin(3d+5z)))), where (b) is the scenario's declared multiplicative forecast bias. The simulator retains the true demand only to score the resulting trace; the policy receives the generated forecast.
- Rain is 1.1 units when (d \bmod 7=4) before drought onset and zero otherwise. The fixed synthetic schedule is visible in the scenario ID and onset fields.
- The per-zone criticality weight is 1.5 in zone 0 on days 10–25, zone 1 on days 17–34, and zone 2 on days 4–19; it is 0.8 outside those windows. The minimum-storage threshold is (2.6+0.9c(d,z)), where (c) is the criticality weight.
- After irrigation, overflow proxy is (max(0,S+A-10)), post-use storage is (max(0,\min(10,S+A)-D)), and end storage is (min(10,\text{post-use}+R)). Daily synthetic stress exposure is (c(d,z)\max(0,\text{minimum-storage}-\text{end-storage})^2); the run score is its sum over all days and zones.
- Sensor error is a deterministic (0.18\sin(11d+7z)). On dropout days the controller holds the latest observed reading and increments stale age. The risk-aware policy applies a heuristic demand margin (1.10+0.05\min(\text{stale-age},4)) and criticality multiplier (1+0.12\min(\text{stale-age},4)). These are uncalibrated heuristics, not probabilistic coverage guarantees.
- Plot-allocation dispersion is the population standard deviation of each plot's cumulative applied irrigation across the 42-day run. Equal-share should therefore report zero dispersion apart from numerical-rounding tolerance.

The six named scenario profiles in the source make factor isolation explicit: reference, water-cap-only, forecast-bias-only, sensor-dropout-only, heat/drought, and compound stress/failure. The water-cap-only and single-failure profiles match the reference on all other declared parameters. Scenario comparisons can be interpreted only at this fixture level.

## Mathematical scope and claim ceiling

This is an intentionally small decision-policy harness, **not a crop model**. It defines artificial plot storage, demand traces, critical windows, weather transitions, sensor error, forecast bias, and a squared shortfall score. Those quantities are transparent test fixtures, not calibrated water balance, evapotranspiration, root-zone dynamics, yield response, nutritional outcomes, or field biology.

The initial hard acceptance condition is zero daily water-budget violations and deterministic replay. A provisional research target is at least 10% lower median synthetic stress-exposure proxy than equal-share allocation across the declared scenario suite, with every adverse scenario reported. The harness reports results but does not silently re-label the 10% target as achieved, and it does not fail merely because a proposed research-performance target is not met.

A policy that wins this synthetic score has only passed an algorithmic test. A defensible agricultural claim requires, in order: a versioned crop/soil water-balance model and its calibration record; comparison with fixed-calendar, soil-threshold, ET-based and a suitable established crop-model baseline; leakage-resistant retrospective or hindcast evaluation; local sensitivity and uncertainty analysis; then preregistered, replicated field trials with metered water, crop outcomes, energy, labor, cost and environmental measurements. Safety, water-allocation rights and any local water constraints remain hard limits rather than soft optimization rewards.

## ## What this benchmark is designed to reveal

- Whether the policy accounts for the cost of scarce water instead of always filling all plots equally.
- Whether decisions degrade gracefully with biased forecasts and intermittent/stale observations. The `water_rationing_only`, `forecast_bias_only` and `sensor_dropout_only` profiles are paired with the reference profile, changing only the designated factor. Their deltas can isolate each synthetic effect; heat/drought and compound-scenario deltas must not be attributed to one cause.
- Whether hard resource caps remain satisfied under all tested conditions.
- Whether the more complicated policy gives measurable decision value over simple baselines.
- Whether the answer changes under stress weights, storage thresholds, forecast error, dropout patterns and scenario mix.

## Candidate field-stage benchmark (not enabled by the synthetic harness)

Before a real trial, choose the crop, soil, irrigation hardware, locally accepted schedule and marketable-quality endpoints with a cooperating agronomist/farm partner. Then preregister a *candidate* practical success criterion: at least 15% lower metered irrigation volume than the strongest locally accepted comparator, with marketable yield non-inferior within a 5% margin. These are proposed design targets only; adjust them before seeing outcomes using agronomic justification, control variability, baseline yield, water scarcity context and a power analysis.

For a trial to pass those targets, report confidence intervals, not just point estimates. One possible rule is an upper confidence bound on the irrigation-volume ratio below 0.85 and a lower confidence bound on the marketable-yield ratio above 0.95, alongside acceptable produce quality and no violation of water or food-safety constraints. Use randomized blocked plots with calibrated flow meters, treatment blinding where feasible, preregistered exclusion rules, enough independent plots/sites/seasons for the stated power, and the same weather and crop-stage records across comparators. Also measure pumping energy, labor, maintenance, input costs, salinity/leaching where relevant, sensor failures and farmer workload. If water is saved by transferring stress or cost to another part of the system, do not declare overall success.

This is deliberately a separate evidence tier. The fixed synthetic scenarios can test that a policy respects coded constraints; they cannot justify those field-stage thresholds or predict the probability of achieving them.

## Follow-on measurements

Before proposing field deployment, add independently calibrated soil-moisture / root-zone observations, weather forecast vintages with issue-time timestamps, metered applied water, irrigation hardware limits, crop-stage definitions, plot-level outcomes and an agronomist-reviewed soil-water budget. Report runtime on target hardware and total instrumentation, connectivity, power, maintenance and labor cost. Preserve all scenario definitions and policy settings before looking at results.

```text
Claim ceiling:
synthetic harness -> implementation and invariant tests
retrospective calibrated replay -> evidence of model/policy performance on that dataset
replicated field trials -> bounded real-world efficacy estimates
multi-site multi-season validation -> transfer evidence within the sampled domain
```
