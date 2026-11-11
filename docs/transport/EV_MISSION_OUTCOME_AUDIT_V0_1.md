# EV mission outcome audit v0.1

## Purpose and ownership

This is the first executable implementation slice for [EV-MISSION-OUTCOME-001 #11633](https://github.com/Luminous-Dynamics/symthaea/issues/11633).

The mission capability evaluator [#11589 / draft PR #11591](https://github.com/Luminous-Dynamics/symthaea/pull/11591) checks a requested duty profile against time-indexed pack/cell capability envelopes. This crate answers the next independent question:

**Did that frozen mission-feasibility prediction agree with a physical execution of the same pack, configuration and mission?**

This is not a battery model, SOC/SOH estimator, mission simulator, BMS, charge controller or safety authority. It is a blind validation scoreboard.

Related owners:
- #11252: dataset rights, physical-group splits, estimators and censor-aware statistical scoring.
- #11390: support-conditioned uncertainty/calibration and OOD audit.
- #11271: topology-aware pack capability and limiting-unit evidence.
- #11445: cross-fidelity electrothermal validation for aircraft/EV-adjacent physics.
- #11589/#11591: mission capability prediction.
- Symtropy #2112/#2114: vehicle mission, whole-vehicle verification and physical test programme.
- Symtropy #2164: industry and benchmark comparison.

## Paradigm shift: evaluate decisions, not prediction averages

A low SOH MAE, strong R² or well-calibrated average regression error does not establish that an EV mission-feasibility decision is correct.

A predicted FEASIBLE mission that physically exceeds a qualified battery constraint is a **false-feasible** outcome, even if its average state estimate error is small. Conversely, a physically completed mission rejected by the model is a **false-infeasible** outcome and can destroy useful-service value.

The audit makes both visible. It also records abstentions, non-battery mission aborts, inconclusive tests and contradictory references, instead of silently dropping them or forcing an incorrect label.

## Evidence contract

Every case binds:
- vehicle, pack, configuration and electrical topology;
- exact mission-profile reference and SHA-256;
- exact mission-evaluator report reference and SHA-256;
- evaluator version and frozen model-artifact reference/digest;
- the full frozen split-manifest payload and recomputed digest, with exact pack/configuration/physical-group assignment and TEST_BLIND partition;
- split-freeze and prediction-freeze timestamps, both preceding physical outcome observation;
- the mission's frozen required physical measurement contract (constraint, scope, unit, limit and minimum/maximum relation);
- independent reference protocol and digest;
- reference evidence object and digest;
- time of physical observation;
- per-constraint measurement references, digests, units and observed values; each measurement reference and mission report reference must be unique within the audit corpus;
- sensor identity plus calibration reference/digest for each physical measurement, with measurement time after prediction freeze and no later than the reference observation time.

The reference must bind to the *same* physical asset/configuration/topology and mission-profile digest as the prediction. A near-match, another pack, a different firmware/configuration or a different mission profile is not a valid reference.

Each observed constraint is replayed against the frozen limit:
- `MINIMUM_REQUIRED`: observed value must be greater than or equal to the declared limit.
- `MAXIMUM_ALLOWED`: observed value must be less than or equal to the declared limit.

The audit checks relation/constraint compatibility (for example, temperature is a maximum-allowed constraint; discharge power is a minimum-required capability). The reference outcome label must also agree with the numeric observations. A battery-constraint failure, non-battery abort or inconclusive test must include an explicit outcome reason. A prediction outside the TEST_BLIND partition, or a prediction/model/split freeze after the physical result is observed, is not eligible for scoring.

### Outcome classifications

| Classification | Meaning |
|---|---|
| `TRUE_FEASIBLE` | Predicted FEASIBLE and independent physical reference completed within all required battery limits |
| `FALSE_FEASIBLE` | Predicted FEASIBLE but independent physical evidence demonstrates a battery constraint violation |
| `TRUE_INFEASIBLE` | Predicted INFEASIBLE and reference demonstrates at least one battery constraint violation |
| `FALSE_INFEASIBLE` | Predicted INFEASIBLE but physical execution completed within all declared battery limits |
| `ABSTAINED` | Evaluator returned NOT_COMPUTABLE, UNSUPPORTED_DOMAIN, CONFLICTING_EVIDENCE or REJECTED_INPUT |
| `NOT_SCORABLE` | Non-battery abort, incomplete reference coverage or inconclusive physical outcome |
| `CONFLICTING_REFERENCE` | Reference identity or its stated mission outcome conflicts with its own measured values |
| `REJECTED_REFERENCE` | Malformed reference evidence or invalid input binding |

A non-battery abort (route closure, operator action, charger/site fault unrelated to the battery, etc.) is not a battery failure and cannot create a false-feasible label. An inconclusive reference remains visible as NOT_SCORABLE. A completed mission requires coverage of every measurement declared by the frozen prediction.

## Independent sample unit and statistical bounds

The primary v0.1 independent unit is **one pack–mission pair per physical pack**. The audit rejects duplicate pack identities in a corpus. Multiple time intervals, cells, cycles or repeated missions from the same pack do not become independent qualification samples.

The false-feasible denominator is the number of independent physically scorable pack–mission cases where the frozen evaluator predicted FEASIBLE. The false-feasible numerator is the subset of those cases physically shown to exceed a battery constraint.

The report computes a one-sided exact Clopper–Pearson confidence upper bound. For (k) false-feasible outcomes in (n) independent predicted-feasible cases, the bound (p_U) solves:

[
P(X leq k mid X sim mathrm{Binomial}(n,p_U)) = 0.05
]

At zero observed false-feasible outcomes:
- 0/59 independent predicted-feasible cases gives a one-sided 95% upper bound of approximately 4.95%;
- 0/299 gives a one-sided 95% upper bound below 1%.

These are sample-size implications, not measured performance claims. The bound is calculated only from cases marked `INDEPENDENT_PHYSICAL_REFERENCE`. Synthetic known-truth fixtures never contribute to the physical denominator or confidence bound.

Repeated missions from the same pack require pack-cluster-aware statistical treatment and a larger effective sample. They must not be counted as independent Bernoulli trials.

## Benchmark tournament

Compare the same pack-disjoint physical holdouts and matched mission boundaries across:
1. instantaneous peak-power-only screen;
2. usable-energy-only screen;
3. static derated power/energy screen;
4. time-indexed power without cumulative reserve;
5. mission capability evaluator with time-indexed cell/pack envelopes;
6. the same evaluator with uncertainty, thermal/SOC/voltage and charge-acceptance constraints enabled.

Report:
- false-feasible rate and exact one-sided upper confidence bound (primary);
- false-infeasible rate and exact bound;
- true feasible/infeasible counts;
- abstention and NOT_SCORABLE coverage;
- false-feasible violations by discharge power, charge acceptance, energy reserve, SOC, voltage and temperature;
- subgroup outcomes by chemistry, temperature, SOC, age, topology, firmware/configuration and duty class, only where independent group counts support them;
- completion, useful work, energy and charge/regen accounting, missed service and downtime at matched whole-vehicle boundaries.

Do not optimize the model against the final physical holdout. Freeze predictor, model, mission profile, required measurements, outcome definitions and physical pack roster before test execution.

### Proposed physical promotion gates (not achieved)

- At least 59 independent predicted-feasible pack–mission cases and zero false-feasible outcomes before making a one-sided 95% upper-bound claim below 5%.
- At least 299 independent predicted-feasible pack–mission cases and zero false-feasible outcomes before making that bound below 1%.
- Zero asset/configuration/topology/profile mismatches in scored physical cases.
- All declared required measurements covered by independent calibrated instruments.
- Report false-infeasible and abstention rates; abstaining on every mission is not a successful system.
- No aggregate claim without worst-subgroup results and adequate independent sample sizes.

## Frozen synthetic known-truth controls

Eleven synthetic cases exercise:
- true feasible;
- false feasible due to excessive cell temperature;
- false feasible due to reserve depletion;
- true infeasible due to inadequate discharge power;
- false infeasible where the reference mission completed;
- model abstention;
- non-battery abort;
- inconclusive reference;
- asset/configuration reference mismatch;
- reference outcome label contradicting measured temperature;
- incomplete measurement coverage.

All identifiers, reference values, measurements, limits and digests are synthetic. They test schema enforcement, classification logic, statistical arithmetic and fail-closed behavior only. No physical EV performance rate is claimed.

## Industry context

Battery state estimation and cloud analytics are already commercial categories:
- AVILOO publishes a rapid battery certificate, checks BMS/sensor and cell-voltage data, and claims at least 95% of reference measurements have a maximum SOH deviation no greater than ±3%. Its own page says the result describes state at test time and cannot detect faults not exposed through vehicle diagnostics: https://aviloo.com/en/aviloo-certified
- Qnovo markets edge physics-based battery monitoring, adaptive charging and fleet cloud diagnostics; its public page states a 98.7% statistical fault-prediction accuracy claim. That figure is a vendor claim, not a directly comparable independent mission-level benchmark: https://qnovo.com/technology
- Bosch Battery in the Cloud markets fleet monitoring, battery digital twins, fault prediction, aging prediction and lifetime optimization: https://www.bosch-mobility.com/en/solutions/software/battery-in-the-cloud/battery-in-the-cloud/
- TWAICE markets BESS performance, safety, SOC imbalance, self-discharge, DC resistance, RTE and root-cause/maintenance insights: https://www.twaice.com/products/energy-storage-analytics/performance-engineers

Therefore, “battery AI”, “digital twin”, SOH estimates or BESS KPI dashboards alone are not a differentiator. The proposed competitive target is a transparent, independent, pack-disjoint and statistically bounded mission-feasibility decision with both false-feasible and false-infeasible risk reported. This page makes no claim that competitors lack undisclosed internal mission models or evidence controls.

## Standards and validation maturity

The current SAE J1634 revision is **J1634_202604**, revised 13 April 2026. It defines standardized light-duty BEV, whole-vehicle energy/range testing using UDDS/HFEDS and supplementary US06/SC03/cold-FTP procedures; it is not a pack-only capability certificate: https://doi.org/10.4271/J1634_202604

Le (28 September 2026) proposes a conceptual six-level battery validation progression from protocol/parameter/stress-oriented testing toward mechanism-oriented, physics-informed and predictive validation. The paper emphasizes application-specific operation, physically observable dominant degradation mechanisms and mechanism-relevant measurements rather than simply adding more stresses or model complexity: https://doi.org/10.3389/fbael.2026.1966110

This audit is the predictive-outcome comparison layer. It does not replace standardized testing, mechanism-specific physical validation or independent engineering authority.

## Safety and claim ceiling

The audit reads evidence and scores predictions. It cannot authorize a physical test, operate a vehicle, set charge/discharge limits, change BMS or charger settings, isolate a pack, approve repairs or certify functional safety. Physical testing must be planned and authorized by qualified engineering and safety personnel.