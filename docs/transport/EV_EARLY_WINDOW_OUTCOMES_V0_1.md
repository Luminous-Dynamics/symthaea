# EV early-window multi-output outcome evidence audit v0.1

## Purpose and ownership

This implementation is the first executable slice of [EV-EARLY-WINDOW-MULTIOUTPUT-001 #11495](https://github.com/Luminous-Dynamics/symthaea/issues/11495).

It does not create another SOH split/manifest system or battery simulator. It composes with:
- SOH data/split/target manifest #11342;
- feature-time and reference-lineage audit #11419;
- support-conditioned uncertainty/OOD audit #11390;
- fixed-pulse conservative power envelope #11384 and model-derived beta-power #11438;
- dynamic 2-RC electrothermal model owner #11485;
- dynamic robust test-planning reference #11404 / draft PR #11567.

This crate audits **whether an early-window experiment's features and output labels are admissible and correctly classified**. It is not an estimator, prediction model, survival-analysis implementation, thermal solver, battery safety controller or physical qualification.

## Research trigger: early-window multi-output benchmark

Subramanian and Stonier, “Toward Battery Intelligence: A Multi-Output Benchmark of Seven Machine Learning Architectures and a Shared-Trunk Multi-Task Network for Retired Lithium-Ion Cells Under Constant Load Discharge,” *Frontiers in Artificial Intelligence* (accepted 9 November 2026; final formatted version pending), evaluated 120 constant-load discharges from six retired 18650 cells (SOH 22–85%, resistance loads 1–3 Ω).

Their audit found 73 full-discharge features with target/future-derived information and showed row-level random-split leakage of cell identity. With the same feature set, random-split R² values of 0.88–0.98 fell to 0.65–0.83 under leave-one-cell-out. Restricting the input to 54 causal features from the first 300 seconds, for 112 discharges long enough, gave reported best LOCO results:
- SOH: R² 0.933, MAE 0.050 (5 SOH percentage points); bootstrap 95% R² interval [0.61, 0.98].
- Remaining discharge time (RDT): R² 0.946, MAE 236 seconds, WAPE 10.3%; nested LOCO conformal intervals covered 94% at nominal 90%, but were wide.
- Peak temperature: R² 0.876, MAE 1.1 °C.

The architecture differences were not statistically significant for SOH/Tmax, and the shared multi-task model did not significantly outperform single-task networks (Wilcoxon p=0.44). The study explicitly limits its claim to six retired cells, one chemistry and one ambient temperature; it calls for further validation before real-time BMS deployment.

Source: https://doi.org/10.3389/frai.2026.1963531

These are useful reproduction anchors, not universal SOTA scores or fleet claims. A six-cell LOCO test is not equivalent to leave-one-pack-out or cross-chemistry transfer.

## Paradigm shift: predict an evidence-qualified capability state, not one health percentage

At a declared prediction cutoff, the system should preserve separate target contracts for:
1. SOH / usable capacity;
2. remaining discharge time under an exact load profile;
3. peak temperature over a defined future horizon and at a defined physical resolution;
4. a model-derived or independently measured power/energy envelope.

Every target retains its own:
- exact asset and configuration identity;
- units and reference protocol;
- reference-condition digest (SOC, temperature, load/rest/cooling boundary as relevant);
- target time/window and horizon;
- target-generation method and evidence digest;
- measurement/label uncertainty;
- observability coverage and physical resolution;
- evidence grade and computability/censoring state.

A common feature window does not make these target labels equally trustworthy. A measured capacity reference, an operational SOH estimate, a censored discharge duration, a cell-sensor temperature maximum and a model-derived power envelope are different evidence classes and must remain different.

## v0.1 contract

### Causal feature receipt

Every feature receipt binds:
- feature identity and digest;
- source class;
- source window start/end;
- time the feature became available;
- whether it uses target/future information.

Only `OBSERVED_PREFIX` and `DERIVED_PREFIX` are admitted. The source window must end at or before the prediction cutoff, and the feature must have arrived by that cutoff. Target-derived, full-lifetime, post-cutoff and late-arriving features are rejected.

The experiment event and physical asset/configuration are explicit. The parent SOH manifest remains responsible for sample/physical-group split identity and train/calibration/test chronology; this crate does not reimplement those checks.

### SOH target

SOH labels require a post-cutoff reference time, protocol and exact measurement-condition receipt, target uncertainty and a declared source:
- `INDEPENDENT_REFERENCE_TEST` is recorded as an independent physical reference;
- `OPERATIONAL_DERIVED` is retained as an operationally derived reference, **not** promoted to independent physical ground truth.

The unit is explicitly fraction or percent and the value is range-checked.

### Remaining discharge time and censoring

RDT is only comparable under a declared load-profile identity/digest and test protocol.

The audit distinguishes:
- `EXACT_ENDPOINT_EVENT`: an observed endpoint under the declared load; target time equals observed duration;
- `RIGHT_CENSORED`: the endpoint was not reached before follow-up ended; no exact point label is permitted;
- `COMPETING_EVENT`: a different event (for example, a thermal-protection abort) ended observation; preserve its event code and duration rather than calling it an endpoint event.

A censored or competing-risk observation must not be turned into a fabricated exact RDT by dropping the row, substituting the last observed time, or treating it as an ordinary regression label. The v0.1 audit preserves those states; it does not yet calculate survival curves, IPCW scores or competing-risk probabilities.

### Peak temperature

Peak temperature is a maximum over a declared future horizon at CELL, MODULE or PACK resolution. A numeric maximum is only admissible when the declared population has complete required observation coverage. If thermal coverage is incomplete, the result may be preserved as `NOT_COMPUTABLE`; it may not be called the global cell/pack peak.

Surface sensor evidence must not be silently described as internal core temperature. Cell/module/pack resolution and measurement method remain part of the label definition.

### Power envelope

A v0.1 `POWER_ENVELOPE` target must cite a qualified physical model, pack identity/configuration, condition and horizon. Its evidence grade is `MODEL_DERIVED_NOT_PHYSICAL_MEASUREMENT`. The audit deliberately does not accept a generic ML forecast or operational estimate as an independent physical power label.

The fixed-pulse envelope #11384 and beta-power #11438 are limited model products; the dynamic 2-RC electrothermal owner #11485 is the path toward time/SOC/temperature-conditioned estimates. None becomes a physical rating or safety limit merely because its software audit passes.

### Asset and condition binding

Every target repeats the exact case asset and configuration digest and carries its own reference-condition reference/digest. A target from another pack, vehicle configuration or test condition is rejected even if its numeric value looks plausible.

## Frozen synthetic corpus

The v0.1 corpus has 14 cases: five accepted and nine rejected.

Accepted:
1. A 30-second multi-output example with SOH, exact-event RDT, cell-level Tmax and a model-derived pack power envelope.
2. A right-censored RDT preserved without a point estimate.
3. A competing event preserved separately from endpoint completion.
4. Incomplete thermal observability retained as `NOT_COMPUTABLE`.
5. An operationally derived SOH label kept distinct from independent reference-test evidence.

Rejected controls:
- a feature arriving after the prediction cutoff;
- a full-lifetime feature in an early-window claim;
- a target reference predating the prediction cutoff;
- a right-censored RDT converted to a point label;
- RDT without a declared load profile;
- an incomplete thermal population presented as a numeric global peak;
- an ML/operational forecast passed off as a qualified physical power envelope;
- duplicate target identity;
- target evidence bound to a different pack identity.

All data, values, identities, conditions and digests are synthetic. They exercise the software/evidence contract and establish no real battery prediction performance.

## Benchmark protocol and baselines

Freeze four early-window cutoffs independently: 30, 60, 120 and 300 seconds. Do not train one model with longer information and report it as a 30-second model. Separate partial-charge and partial-discharge tasks.

For every cutoff, compare:
- simple physical/statistical baselines where the measured inputs support them;
- ridge/SVR/tree/boosting/MLP and ensembles;
- single-output models per target;
- shared-trunk multi-task model;
- physical-model-only and hybrid residual variants;
- explicit no-estimate / abstention.

Keep exact feature budgets, preprocessing fit scopes, model digests and compute budgets in the parent manifests. Use same-cell temporal only for future-in-same-cell claims; use leave-one-cell-out, leave-one-pack-out, leave-one-vehicle-out and leave-one-lab/dataset-out for the corresponding transfer claims. Never use row-random splits to claim unseen-asset generalization.

### Metrics by output

| Output | Required primary metrics | Additional evidence |
|---|---|---|
| SOH | MAE/RMSE in percentage points; worst-cell/pack | Reference-protocol uncertainty, interval coverage/width, domain calibration |
| RDT exact events | MAE/RMSE/WAPE on uncensored events only | Survival C-index, IPCW integrated Brier, time-dependent calibration/AUC |
| RDT competing risks | Cause-specific or cumulative-incidence scores | Event-type confusion and calibration |
| Peak temperature | MAE/RMSE in °C; worst group | 90/95 interval coverage, max-sensor coverage, constraint-event lead time |
| Power/energy envelope | Bound error and conservative coverage vs matched physical test | Limiting-cell attribution, time-to-limit, underprediction/overprediction cost |

For safety-relevant temperature/voltage limits, report false-safe events and one-sided confidence bounds, not only mean temperature error. A forecast interval is not itself a safety limit; the limit and acceptable risk must be set by the competent engineering authority.

### Proposed research gates (not achieved)

- Zero post-cutoff or target-derived feature leakage.
- Zero physical-asset/configuration mismatch between features and labels.
- 100% preservation of right-censoring and competing-event status.
- Zero numeric global peak claims with incomplete required thermal population coverage.
- Reproduce the six-cell LOCO results and quantify uncertainty before claiming improvement.
- For a future larger independent cohort: SOH MAE ≤3 percentage points on supported unseen assets as an initial target, ≤2 as stretch; RDT WAPE ≤10% on exact endpoint events with calibrated survival metrics for censored cases; Tmax MAE ≤1 °C on matched conditions, with separate worst-group and interval-coverage gates.
- A multi-task model must show statistically significant improvement, lower test/data burden or lower compute at matched accuracy. Shared parameters alone are not an improvement.
- Publish risk/coverage and compute/data/accuracy Pareto fronts; include unsupported-domain abstention.

These are research acceptance targets, not current results, safety criteria or global SOTA guarantees.

## Industry comparison and differentiation

This market already has serious capabilities:
- Bosch Battery in the Cloud markets fleet battery digital twins, anomaly detection, aging prediction up to eight years ahead, and charging/lifetime optimization.
- AVILOO offers independent EV battery certificates and reports cell/pack diagnostics. It states that at least 95% of reference measurements have a maximum SOH deviation no greater than ±3%; it also makes clear that its result is a snapshot of condition at test time, not a prediction of future performance.
- Qnovo markets edge/cloud diagnostics, adaptive charging and states 98.7% accuracy for its battery-health issue identification. That is a vendor-published performance claim, not an independently matched benchmark result.
- TWAICE markets BESS analytics across SOC imbalance, DC resistance, round-trip efficiency, availability, root-cause analysis and maintenance recommendations.

Sources:
- https://www.bosch-mobility.com/en/solutions/software/battery-in-the-cloud/battery-in-the-cloud/
- https://aviloo.com/en/aviloo-certified
- https://qnovo.com/technology
- https://www.twaice.com/products/energy-storage-analytics/performance-engineers

We should not claim superiority over any of these vendors using different datasets or metrics. Our differentiated research target is narrower and independently testable: **one time-aligned, asset-bound output contract that preserves different evidence grades, censoring, missing physical observability and the distinction between measured and model-derived capability**. This is a foundational layer for a complete battery lifecycle intelligence system, not proof that the overall platform is already superior.

## Promotion and safety boundary

Promotion sequence:

causal observation window -> source/asset/configuration admission -> target-specific evidence audit -> estimator and support/OOD audit -> capability envelope with physical model validity -> dynamic diagnostic/test policy -> separately authorized intervention -> independent outcome verification -> lifecycle record.

A predicted SOH, RDT or Tmax is not an intervention or operating permit. This crate does not run tests, authorize maintenance, change charging profiles, set BMS thresholds, control contactors or derate the vehicle. It does not certify compliance with SAE/IEC/ISO standards. A passing synthetic audit only proves that the declared software contract behaves as expected on the frozen synthetic cases.
