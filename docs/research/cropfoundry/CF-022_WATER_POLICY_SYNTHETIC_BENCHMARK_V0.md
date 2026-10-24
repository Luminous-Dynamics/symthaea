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

The harness runs 6 fixed scenario profiles × 5 receipt policies × 42 time steps:

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
4. `water_matched_rotating_equal_share`: a hindsight counterfactual that reallocates the risk-aware policy's exact daily total water uniformly across zones, rotating discrete rounding remainders. It isolates water-placement effects but is not a deployable policy.
5. `greedy_minimax_stress`: a greedy fairness-oriented policy that tests each candidate water increment and chooses the allocation that minimizes maximum predicted per-zone stress proxy, using summed proxy as a tie-breaker. This is a heuristic policy, not a global optimality or field-safety guarantee.

The v2 receipt records scenario parameters, per-scenario and per-zone irrigation, overflow proxy, aggregate and per-zone stress-exposure proxies, worst/best zone burden, worst-to-best ratio, cumulative allocation dispersion, percentage change against equal-share, and the risk-aware policy's stress delta versus the dose-matched rotating-uniform counterfactual. It also reports absolute/relative stress change versus the same policy in the reference scenario, plus policy-level mean/median aggregate and worst-zone proxies, both hard-constraint violation counts and water totals. It also reports sensor-dropout fraction and daily budget violations. The executable fails if any policy exceeds a daily total-water budget, exceeds any plot's actuator limit, or emits non-finite core metrics. It now scores the minimax policy alongside the two original optimization policies and the dose-matched hindsight counterfactual.


**Unit note:** per-plot storage/demand values are synthetic depth-like values. Global irrigation totals add depths over equally weighted toy plots and are labelled `plot-mm`; without plot areas, flow rates and hydraulics, these sums are not volumetric water use and must not be compared with farm meter readings.

## Pinned synthetic scenario math (v0)

These equations describe test fixtures, not agronomy.

- Initial storage is 6.0 synthetic depth units per plot; storage is capped at 10.0. Each plot can receive at most 3.5 units/day. Threshold/risk-aware policies allocate in 0.05-unit increments.
- For day `d` and zone `z`, base demand is `[2.2, 2.7, 1.8][z]`. The phase factor is `0.88 + 0.22 * (sin(PI*d/42 + 0.7*z) + 1)/2`. After the scenario's heat-onset day, demand is multiplied by 1.30.
- The exposed demand forecast is `max(0.1, actual_demand * (1 + bias + 0.04*sin(3*d + 5*z)))`, where `bias` is the scenario's declared forecast bias. The simulator retains true demand only to score the trace; the policy receives the generated forecast.
- Rain is 1.1 units when `d % 7 == 4` before drought onset, and zero otherwise.
- Per-zone criticality is 1.5 in zone 0 on days 10–25, zone 1 on days 17–34, and zone 2 on days 4–19; it is 0.8 outside those windows. The minimum-storage threshold is `2.6 + 0.9*criticality`.
- After irrigation, overflow proxy is `max(0, storage + allocation - 10)`. Physical post-use storage is clamped as `max(0, min(10, storage + allocation) - actual_demand)`, then physical end storage is `min(10, post_use + rain)` for the next step. **Receipt v2 scores a separate water margin** `end_storage_margin = min(10, storage + allocation) - actual_demand + rain` without clipping the margin at zero. Daily synthetic stress is `criticality * max(0, minimum_storage - end_storage_margin)^2`. Negative margin is an unmet-demand severity proxy, not physically negative stored water; it preserves score sensitivity even when that day's demand exceeds all available water.
- Sensor error is the deterministic term `0.18*sin(11*d + 7*z)`. On dropout days, the controller holds the latest observed reading and increments stale age. The risk-aware policy applies a heuristic demand margin `1.10 + 0.05*min(stale_age, 4)` and criticality multiplier `1 + 0.12*min(stale_age, 4)`. These are uncalibrated heuristics, not probabilistic coverage guarantees.
- Plot-allocation dispersion is the population standard deviation of each plot's cumulative irrigation across the 42-day run. Equal-share should therefore report zero dispersion apart from numerical-rounding tolerance.

The six named scenario profiles are reference, water-cap-only, forecast-bias-only, sensor-dropout-only, heat/drought, and compound stress/failure. The water-cap-only and single-failure profiles match the reference on all other declared parameters. Scenario comparisons can be interpreted only at this fixture level.

## Mathematical scope and claim ceiling

This is an intentionally small decision-policy harness, **not a crop model**. It defines artificial plot storage, demand traces, critical windows, weather transitions, sensor error, forecast bias, and a squared shortfall score. Those quantities are transparent test fixtures, not calibrated water balance, evapotranspiration, root-zone dynamics, yield response, nutritional outcomes, or field biology.

The initial hard acceptance condition is zero daily total-water-budget violations, zero per-plot actuator-limit violations, and deterministic replay. A provisional research target is at least 10% lower median synthetic stress-exposure proxy than equal-share allocation across the declared scenario suite, with every adverse scenario reported. The harness reports results but does not silently re-label the 10% target as achieved, and it does not fail merely because a proposed research-performance target is not met.

A policy that wins this synthetic score has only passed an algorithmic test. A defensible agricultural claim requires, in order: a versioned crop/soil water-balance model and its calibration record; comparison with fixed-calendar, soil-threshold, ET-based and a suitable established crop-model baseline; leakage-resistant retrospective or hindcast evaluation; local sensitivity and uncertainty analysis; then preregistered, replicated field trials with metered water, crop outcomes, energy, labor, cost and environmental measurements. Safety, water-allocation rights and any local water constraints remain hard limits rather than soft optimization rewards.

## ## New counterfactual: same daily water, different allocation

The harness now adds a fourth trace for each scenario: `water_matched_rotating_equal_share`. It reuses the risk-aware policy's exact daily total irrigation dose, but redistributes the discrete 0.05-unit increments as evenly as possible across the three synthetic plots, rotating any remainder between plots. The implementation asserts that each day's total dose matches the risk-aware trace and reports stress, per-zone allocation, worst-zone burden, disparity and dose-matched relative stress change.

This is a **counterfactual analysis baseline**, not a directly deployable controller: the matched baseline receives the risk-aware policy's realized daily dose trace, so it uses hindsight to isolate the spatial allocation question. That is deliberate and must be labelled as such. It answers: *given the same water on every day, does the placement of that water change the toy objective?* It does not answer whether the policy can forecast or achieve that water trace in a real field.

Keep the two comparisons separate:
- **Equal-budget-cap comparison:** policies may choose to use less than the permitted daily cap; this reflects both allocation and the decision to withhold water.
- **Dose-matched comparison:** the per-day total water is fixed to the risk-aware trace; differences in the synthetic score then come from spatial allocation, state feedback and the fixture's plot dynamics, not the total daily dose.

The second counterfactual is intended to diagnose whether the risk-aware score gain comes from allocating water to particular plots rather than simply using less water. It is still possible to hide burden by sacrificing one plot, so inspect zone-level stress and distributional metrics alongside aggregate stress. No agronomic or biological claim follows from either comparison.

## Historical v1 finding: aggregate stress is not a sufficient objective

The first matched-dose receipt (schema v1) exposed a serious disparity failure, but subsequent review found a scoring defect in that version. The table below is preserved as a historical debugging artifact, not as a valid final benchmark result. All four v1 receipt policies had zero daily total-water-budget violations and zero per-plot-limit violations across the six scenarios.

| Synthetic policy | Mean aggregate stress proxy | Median worst-zone stress proxy | Maximum worst-zone stress proxy | Mean worst-to-best zone stress ratio | Total irrigation (plot-mm) | Mean allocation dispersion |
|---|---:|---:|---:|---:|---:|---:|
| Equal-share calendar | 958.356 | 539.651 | 597.378 | 7.312 | 1,386.0 | 0.000 |
| Soil threshold | 1,225.457 | 508.365 | 577.237 | 2.069 | 1,373.1 | 9.016 |
| Uncertainty-aware marginal risk | 643.215 | 562.845 | 610.689 | 6,151.882 | 909.7 | 33.400 |
| Water-matched rotating-uniform counterfactual | 1,497.231 | 563.419 | 607.899 | 1.447 | 909.7 | 0.057 |

All scores are from the toy scenario equations. "plot-mm" is a sum of synthetic depths, not physical water volume. The matched-uniform row uses the risk-aware policy's exact daily total-water trace, so both policies use the same amount of water on every day.

**Why these v1 scores are superseded:** the old scoring function clipped the predicted end-of-step storage margin to zero before computing stress. When a zone was already predicted to be empty, a small additional water increment could produce zero marginal improvement until it covered that day's demand. That created an artificial dead zone in the optimizer and allowed it to starve a plot while scoring well elsewhere. Receipt schema v2 now retains negative end-of-step water margin as unmet-demand severity for scoring, while the simulated physical storage state remains clamped between zero and capacity. All policy scores must be recomputed under v2; never compare v1 and v2 score levels directly.

The risk-aware policy has a much lower **aggregate** stress proxy than the dose-matched uniform counterfactual, but its worst-zone stress is not lower, its maximum worst-zone exposure is slightly higher, and its mean worst-to-best zone-stress ratio is extreme. In several scenarios it allocated almost all water to zones 0 and 2 while zone 1 accumulated almost all the stress proxy. The score rewarded protecting two zones and allowed one to be sacrificed. That is a **benchmark failure of the aggregate objective**, even though arithmetic, determinism and resource constraints passed.

Do not describe the historical -57.36% matched-dose stress delta as success. The next research gate is to recompute all policies under the v2 stress proxy and assess the new greedy minimax policy; acceptance must include a predeclared bound on worst-zone stress and disparity alongside aggregate performance. No crop safety or agronomic guarantee can be inferred from the v1 synthetic numbers.

## Receipt v2 exploratory policy comparison

A successful exact-head synthetic smoke run (`39242840320`, head `373a4f4d0859b4e31ac257e46bf3d301f6d28c90`) emitted the following receipt-v2 metrics before the disparity-ratio denominator count was added. The policy calculations were unchanged by that subsequent metadata-only addition; the newest head is being rerun for a fresh receipt. All values below come from the synthetic fixture and do not represent crop outcomes.

| Policy | Mean stress proxy | Median stress proxy | Median worst-zone stress | Maximum worst-zone stress | Mean worst/best ratio | Scenarios with ratio defined | Total irrigation (plot-mm) | Mean dispersion | Mean change vs equal-share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal-share calendar | 1,471.634 | 1,323.888 | 842.409 | 1,283.220 | 13.405 | 6/6 | 1,386.00 | 0.000 | — |
| Soil threshold | 1,692.042 | 1,578.861 | 734.069 | 1,243.666 | 2.746 | 6/6 | 1,373.10 | 9.016 | +15.215% |
| Uncertainty-aware marginal risk | 1,170.348 | 1,029.263 | 385.099 | 892.691 | 1.420 | 6/6 | 1,353.85 | 18.945 | −26.034% |
| Greedy minimax stress | 1,358.645 | 1,241.649 | 436.532 | 890.410 | 1.198 | 6/6 | 1,353.85 | 19.890 | −10.462% |
| Water-matched rotating-uniform counterfactual | 1,664.704 | 1,532.213 | 868.059 | 1,301.430 | 5.694 | 3/6 | 1,353.85 | 0.020 | +15.873% |

Every policy had zero daily total-water-budget violations and zero per-plot actuator-limit violations in that run. The ratio is undefined when the minimum zone stress is zero; the mean uses only scenarios with a defined ratio. v2 has corrected the severe starvation blind spot seen in v1, so **the v1 scores are superseded and must not be numerically compared with this table**.

This still isn't one universal "winner." The marginal-risk policy has the lower mean aggregate proxy; the minimax heuristic improves the average worst/best ratio and slightly improves the maximum worst-zone proxy, but pays with a higher aggregate proxy and allocation dispersion. Those trade-offs motivate a Pareto report and explicit limits on worst-zone burden, not a single optimized scalar. The matched-uniform counterfactual is hindsight-based and should not be treated as an achievable controller result.

## Acceptance tests for the matched-dose comparison

The implementation must demonstrate all of the following before a receipt is considered structurally valid:
- For every scenario, the risk-aware and water-matched records have identical **daily** irrigation totals, not just matching run totals.
- The matched record has zero daily total-water-budget violations and zero per-plot actuator-limit violations.
- The matched-uniform policy's cumulative per-plot allocation dispersion is no greater than the risk-aware policy's dispersion in the declared test scenarios; this is an algorithmic property of this deterministic fixture, not a universal farm guarantee.
- The risk-aware record reports the stress difference relative to its matched-dose baseline; the baseline itself has no self-comparison delta.
- Every receipt stays `synthetic_only: true`, `biological_performance_claim: false`, and `training_authorized: false`.

If any invariant fails, fix the allocator or weaken the claim to match the exact invariant that is proved; do not hide the failure behind a better aggregate score.

## Per-zone safety and distributional reporting — receipt schema v2

The first working receipt made an important weakness visible: the uncertainty-aware policy concentrated more water on a subset of plots than equal-share allocation. In the earlier exploratory synthetic receipt, equal-share had median aggregate stress proxy 861.376 and total irrigation 1,386 plot-mm; the uncertainty-aware policy had median proxy 562.967 and total irrigation 909.7 plot-mm. Its mean plot-allocation dispersion was 33.4 plot-mm versus 0 for equal share, although all tested policy/scenario combinations had zero water-cap violations. These are outputs from an artificial fixture, not crop observations or physical water volume.

That uneven allocation is why the harness now emits **per-zone cumulative irrigation, per-zone stress exposure, worst-zone and best-zone stress, and worst-to-best stress ratio** in receipt schema v2. Policy-level summaries also report the median and maximum worst-zone stress across scenarios. Inspect these beside total stress and allocation dispersion. A reduction in aggregate proxy must not be considered sufficient if the burden is shifted to one plot or a crop/household subgroup.

The earlier receipt was an **exploratory implementation smoke run under the aggregate-only schema**. It is not a completed evaluation of the expanded reporting contract. Rerun the current harness and preserve the new per-zone metrics before evaluating CF-022 further. The provisional 10% aggregate target is unchanged; no permissible worst-zone burden or fairness bound has been locked, so this synthetic benchmark cannot authorize operational deployment.

## What this benchmark is designed to reveal

- Whether the policy accounts for the cost of scarce water instead of always filling all plots equally.
- Whether decisions degrade gracefully with biased forecasts and intermittent/stale observations. The `water_rationing_only`, `forecast_bias_only` and `sensor_dropout_only` profiles are paired with the reference profile, changing only the designated factor. Their deltas can isolate each synthetic effect; heat/drought and compound-scenario deltas must not be attributed to one cause.
- Whether hard resource caps remain satisfied under all tested conditions.
- Whether the more complicated policy gives measurable decision value over simple baselines.
- Whether aggregate improvement concentrates stress or irrigation burden on only a subset of plots; report worst-zone burden and disparity, not only the mean.
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
