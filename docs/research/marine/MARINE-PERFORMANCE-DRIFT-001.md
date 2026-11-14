# MARINE-PERFORMANCE-DRIFT-001
## Time-causal reference-condition monitoring and cause attribution

**Status:** research specification v0.1.0
**Issue:** #12198
**Parent:** MARINE-BIOFOULING-ENERGY-ECOLOGY-001 #11608
**Related:** #11612, #8171, #8092, #8102, #8109, #8130, #8138, #11064, #12095, #12096, #12135, #12136, #12187

> Research and benchmark specification only. It does not diagnose real fouling, establish hydrodynamic validity, authorize cleaning/maintenance, certify vessel performance or authorize physical control.

## Parentage and owner boundaries

This is a marine-specific performance-drift benchmark, not a second biofouling model, conformal-prediction library, voyage-energy calculator or vessel digital twin.

Compose:
- Biofouling, hull condition, cleaning and biosecurity: Symthaea [MARINE-BIOFOULING-ENERGY-ECOLOGY-001 #11608](https://github.com/Luminous-Dynamics/symthaea/issues/11608).
- Contextual acoustic / hull / propulsion record join: Symthaea [#11612](https://github.com/Luminous-Dynamics/symthaea/issues/11612).
- Conditional calibration under hydrodynamic regime shift: Symthaea [MARINE-SHIFT-001 #8171](https://github.com/Luminous-Dynamics/symthaea/issues/8171). Reuse its method/evaluator; do not implement another conformal or calibration engine.
- Voyage energy and measured propulsion traces: Symthaea [#8092](https://github.com/Luminous-Dynamics/symthaea/issues/8092), [#8102](https://github.com/Luminous-Dynamics/symthaea/issues/8102).
- Matched-condition A/B trials: Symthaea [#8109](https://github.com/Luminous-Dynamics/symthaea/issues/8109); exact operating-envelope coverage [#8130](https://github.com/Luminous-Dynamics/symthaea/issues/8130); fidelity/validation cost [#8138](https://github.com/Luminous-Dynamics/symthaea/issues/8138).
- Campaign, sensor, calibration, sample-rate and clock binding: Symthaea [#11064](https://github.com/Luminous-Dynamics/symthaea/issues/11064).
- Product consumers: PWC #12095 / waterjet transient #12135; tug #12096 / EMS #12136.
- Optional shaft-load diagnostic channel: Symthaea [MARINE-SHAFT-OBSERVABILITY-001 #12187](https://github.com/Luminous-Dynamics/symthaea/issues/12187).

This issue owns the **time-causal, vessel-specific performance reference / residual benchmark, confounder challenge panel and cause-attribution abstention contract**. It does not own a generic model-serving runtime, generic ML calibration algorithm, a hull/propulsion solver, an organism classifier, cleaning authorization or physical-control authority.

## Why this work is distinct

The existing biofouling evidence auditor records hull zones, coating state, inspection/cleaning evidence and a matched before/after performance comparison. It does not yet qualify a continuously updated reference-condition model or determine whether an observed performance residual is attributable to biofouling rather than another physical or measurement change.

The July 2026 Lang et al. paper, *Physics-guided machine learning for ship biofouling assessment in support of maritime decarbonization*, Transportation Research Part D 156, 105364 (DOI https://doi.org/10.1016/j.trd.2026.105364), is a fresh method comparator. It describes a physics-informed unfouled warm-up baseline and incremental model updates over sliding temporal windows, with gradual degradation and post-cleaning recovery assessment against conventional retraining and ISO 19030. The paper's data statement says the authors do not have permission to share the data. It is not an open reproduction dataset or proof of transfer to PWC waterjets or tug azimuth/VSP propulsion.

## Research question

Can a vessel-specific reference-condition model detect meaningful performance drift early, maintain calibrated residual uncertainty under route / season / mission shift, and distinguish **what has changed** from **why it changed**?

The system must separate:
1. **Observed residual:** measured speed, shaft/electrical power, fuel or energy versus a frozen reference condition.
2. **Model discrepancy / domain shift:** whether the current case remains inside the reference model's support and calibration domain.
3. **Candidate cause:** hull fouling, propulsor/intake condition, drivetrain efficiency, machinery, battery/generator, loading, environment, sensor/configuration change, or unknown.
4. **Independent evidence:** hull-zone inspection, shaft-load/torque evidence, intake/propulsor inspection, calibrated machinery measurements and cleaning records.
5. **Decision disposition:** monitor, inspect, request competent review, or indeterminate. A performance residual cannot authorize cleaning or maintenance.

A performance monitor may identify that the vessel changed. It must not claim to know the biological or mechanical cause when the evidence does not distinguish them.

## Paradigm shift: causal, time-causal performance retention

Move beyond a single baseline fuel/speed KPI to a continuous, versioned reference model whose predictions are reproducible as of each decision time.

- **Time-causal learning:** each prediction uses only data available by that decision time. Future telemetry, post-cleaning data, later inspection labels and revised calibrations cannot leak backward.
- **Immutable historical predictions:** an update creates a versioned successor model; it never rewrites previous predictions or the input window that generated them.
- **Reference-condition counterfactual:** normalize comparisons only for declared variables with adequate support (e.g. speed, displacement/draft, trim, water density, wind/current/waves, depth, route/task and propulsion state). Unsupported normalization is an abstention, not an extrapolated value.
- **Confounder challenge:** create known cases where performance changes without fouling (weather/load/trim change, engine or inverter repair, impeller/propeller damage, intake debris, battery SOH/temperature change, shaftline change, sensor drift, calibration/remount).
- **Cause attribution requires additional evidence:** independent inspected hull zones or qualified machinery/shaft data can support bounded attribution; the residual alone cannot.
- **Selective prediction:** report DRIFT_DETECTED, NO_DRIFT_DETECTED, INDETERMINATE, OUT_OF_DOMAIN, and cause-specific attribution only where independently supported. Do not collapse uncertainty and cause into one score.

## V0 benchmark

Build a synthetic, deterministic, event-sequenced trace corpus first. It is a harness for arithmetic, time ordering, leakage and attribution semantics—not hydrodynamic training data.

### Required scenarios

1. Stable clean reference, no drift.
2. Gradual synthetic resistance/power residual with independently labelled fouling progression.
3. Step-change after a synthetic cleaning event followed by recovery.
4. Wind/current/wave shift with no hull-condition change.
5. Payload/displacement/draft/trim change.
6. Mission/speed/route mix change.
7. Propeller/impeller/nozzle/azimuth/VSP degradation or damage.
8. PWC intake blockage / debris event distinct from biological fouling.
9. Engine/motor/inverter/gearbox efficiency change.
10. Battery capacity/resistance/temperature derating or generator/auxiliary change.
11. Shaftline/transient load change with possible #12187 advisory input.
12. Sensor drift, remount, recalibration, clock skew, missing samples, clipping and sample-rate change.
13. Hull coating repair or configuration revision.
14. Distribution shift with no labelled target data, including support gap and concept shift.
15. Mixed simultaneous causes and insufficient observability, requiring abstention.

Each sequence binds vessel, propulsion and coating configuration; service/mission fingerprint; reference model version; source and availability times; sensor/calibration identity; raw-data digest; operating conditions; inspection/cleaning evidence; train/update cutoff; and held-out grouping.

## Mandatory baseline comparisons

- Frozen physics/empirical reference model.
- Conventional periodic retraining.
- Physics-guided warm-up baseline plus incremental/sliding-window updates (Lang et al. as a method comparator).
- Drift-triggered update with predeclared thresholds and a no-update/abstain policy.
- Condition/group-specific calibration through #8171 where minimum support exists.
- Oracle truth-informed diagnostic only as a non-deployable upper bound.

Do not claim reproduction of Lang et al. without a licensed or independently admitted dataset and sufficient methodological detail. Do not transfer ISO 19030-1 conformance from conventional fixed-pitch propeller ships to waterjet PWC or azimuth/VSP tugs. Use it as a reference-method comparator within its declared scope only.

## Metrics

Report separately:
- residual bias, MAE/RMSE and interval coverage/width by vessel, mission, route, season, speed/depth/load and propulsion configuration;
- detection delay and missed-drift rate for declared synthetic change points;
- false alarm rate under each no-fouling confounder;
- cause-attribution precision/recall only where independent labels exist;
- abstention rate and selective risk/coverage;
- worst-slice undercoverage, not pooled coverage alone;
- data-leakage and time-cutoff violations;
- update frequency, stale-model exposure, rollback/recovery behavior and runtime;
- cost per useful inspection triggered and cost per independently confirmed cause, once measured data exist.

Do not choose one pooled metric that can hide a false fouling diagnosis or critical-regime undercoverage.

## Falsifiers and hard gates

- Zero confirmed fouling attribution from fuel/speed/shaft-power residual alone.
- Zero future-data or post-intervention leakage into historical predictions.
- Zero silent rewrite of reference baselines, training windows or historical prediction records.
- Zero claimed confidence outside calibrated support; return INDETERMINATE/OUT_OF_DOMAIN.
- No cleaning or physical maintenance action is automatically authorized by this model.
- A matched post-cleaning performance recovery is not proof of causality without matched conditions, independent inspection and measurement uncertainty.
- All censored, missing, failed and out-of-domain windows remain visible in the denominator and audit output.

## Product transfer boundaries

- **PWC-SIT:** distinguish hull drag, intake blockage, debris, impeller/nozzle condition, ventilation/cavitation, rider loading and powertrain drift. ISO 19030-1 is not a direct waterjet-PWC conformance method.
- **HARBOUR-TUG:** distinguish hull/propeller/azimuth/VSP performance, shaft/gearbox/motor/generator efficiency, battery state, daily job mix, assist/escort mode and port conditions. Score energy per completed job, not just raw hourly consumption.
- **Both:** use the same causal semantics but separate product-specific models, operating domains, instrumentation and baseline missions.

## Deliverables

1. Versioned benchmark protocol and deterministic synthetic reference corpus.
2. Independent time-causality / leakage oracle and negative controls.
3. Model-version and prediction-history manifest (reference existing identity/digest owners rather than new provenance infrastructure).
4. #8171 calibration adapter and grouped holdout protocol.
5. Confounder and cause-attribution matrix with explicit abstention.
6. Integration map to #11608 inspection/cleaning evidence, #11612 contextual join and PWC/tug product owners.
7. Future measured-data admission plan requiring raw bytes, rights, calibration, timebase, configuration and independent labels.

## Claim ceiling

A passing synthetic benchmark shows only that the declared temporal, leakage, residual and abstention rules work on the frozen synthetic cases. It does not identify real biofouling, prove model accuracy, validate hydrodynamics, establish energy savings, authorize cleaning, certify a vessel or grant physical control authority.

