# AIRROAD sparse motor-map identification and uncertainty benchmark

**Programme:** AIRROAD-MAP-IDENTIFICATION-001  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11604  
**Status:** research/benchmark protocol; no physical map dataset or fitted result is claimed

---

## Parent and composition

- Strict context-bound efficiency map oracle: [#11577](https://github.com/Luminous-Dynamics/symthaea/issues/11577), [PR #11579](https://github.com/Luminous-Dynamics/symthaea/pull/11579).
- Gearbox-aware, map-interval drive consumer: [#11590](https://github.com/Luminous-Dynamics/symthaea/issues/11590), [PR #11592](https://github.com/Luminous-Dynamics/symthaea/pull/11592).
- Calibrated raw propulsion measurement contract: [#11529](https://github.com/Luminous-Dynamics/symthaea/issues/11529).
- End-to-end aircraft/mission correlation: [#11445](https://github.com/Luminous-Dynamics/symthaea/issues/11445).
- Mission-level coupled solver: [#11411](https://github.com/Luminous-Dynamics/symthaea/issues/11411).

## Purpose

Define and eventually implement a fair, leakage-safe benchmark for constructing motor torque-speed efficiency maps from sparse or costly measurements. Compare interpolation and model families on held-out operating regions and held-out physical configurations, with uncertainty calibration and mission-weighted performance.

This programme owns map-construction/evaluation methodology and benchmark corpus. It does not implement another motor electromagnetic solver, does not authorize test points, and does not claim physical map validity without calibrated measurements.

## Why this benchmark matters

The current v0.2 strict map oracle performs deterministic bilinear interpolation inside the measured rectangular support and returns conservative nodal-plus-systematic absolute bounds. That is a useful independent reference, but not automatically the best map-construction method. A regular grid may require many expensive test points; bilinear interpolation can miss curvature and loss ridges; a more flexible interpolator can overfit sparse noisy measurements; and uncertainty quality matters more than nominal fit error where motor sizing or thermal margin decisions are sensitive.

External reference points:

- Roshandel et al., *Efficiency Maps of Electrical Machines: A Tutorial Review*, IEEE Transactions on Industry Applications 59(2), 2023, reviews efficiency-map calculation methods and their accuracy/computational-cost trade-offs: https://doi.org/10.1109/TIA.2022.3210077.
- The 2024 study *An Application of Uncertainty Quantification to Efficiency Measurements and Validating Requirements through Correlating Simulation and Physical Testing Results* reports torque-dependent uncertainty, particularly in low-torque regions where transducer full-range calibration offsets are proportionally important; its investigated peak-efficiency region had roughly 2% uncertainty at 95% confidence: https://pmc.ncbi.nlm.nih.gov/articles/PMC11315023/.
- A flight-validated small-UAS electric-powertrain study used direct propeller torque/RPM measurements; its integrated model predicted battery voltage within 5% during a 5+ minute mission even though component motor/controller efficiency errors reached up to 10%: https://www.mdpi.com/2226-4310/11/1/16.
- A July 2026 SAE paper describes a modular real-time electric powertrain digital twin built from testbench measurements using physically informed features and component/subsystem hybrid models: https://saemobilus.sae.org/papers/development-a-digital-twin-electrical-powertrain-a-testbench-environment-efficiency-estimations-2026-01-0744.
- An April 2026 IEEE paper studies multivariate global sensitivity analysis for electric-machine efficiency maps under design uncertainty and compares Monte Carlo sampling with polynomial-chaos expansions: https://doi.org/10.1109/TEC.2026.3687758.

These are different platforms, methods and evidence classes. The values above are comparison anchors, not directly interchangeable pass thresholds.

## Paradigm shift

Move from **uniform map fitting followed by one aggregate error number** to **decision-aware, uncertainty-calibrated map identification**.

A candidate map is good only if it:
1. predicts held-out measurements;
2. quantifies when its prediction may be wrong;
3. identifies where the map is unsupported;
4. preserves physical constraints and exact configuration context;
5. provides better mission-relevant decisions at a comparable measurement budget.

Prioritize map samples by their expected effect on design/mission decisions, not solely by geometric grid spacing or largest pointwise uncertainty. Use a separate test authority to approve physical measurements.

## Methods to compare

All methods receive exactly the same training points, context variables, calibration evidence and held-out data:

1. Nearest-neighbour (simple baseline; must be clearly labelled, never used as silent fallback).
2. Bilinear interpolation on regular grids (current oracle baseline).
3. Triangulated piecewise-linear interpolation for irregular samples.
4. Tensor-product splines / shape-preserving local interpolation (only where physical constraints are preserved).
5. Radial-basis / Gaussian-process regression with predictive uncertainty (multiple kernels and noise models declared).
6. Physics-informed loss-surface model using separately identified copper, iron, switching and mechanical loss terms where data support those components.
7. Hybrid physics + residual model, with residual uncertainty and applicability domain explicit.

Do not assume a Gaussian process or neural model is superior. Compare against the simple bilinear baseline first, on identical data splits and measurement budget.

## Data and split protocol

- Pin raw measurement dataset, calibration, test procedure, motor serial/lot, winding/rotor construction, inverter, gearbox, cooling, firmware/control, DC voltage, thermal state and environmental conditions.
- Preserve raw synchronized torque, RPM, electrical terminal measurements, temperature, cooling and quality metadata.
- Correct for sensor calibration and uncertainty through a documented versioned pipeline; never overwrite raw data.
- Primary evaluation: held-out contiguous torque-speed patches, not random neighboring points.
- Secondary: held-out speed bands, torque bands and high-curvature/low-efficiency regions.
- External generalization: leave-one-asset/configuration/lot out where multiple physical units exist.
- Environmental generalization: hold out temperature, DC voltage, cooling or altitude context; such cases are OOD unless the model explicitly conditions on those variables and has support.
- Any preprocessing, feature selection, model tuning and interval calibration must occur inside the training folds.
- Keep one final sealed test campaign and do not use it for experiment selection or model tuning.
- Include repeat measurements to quantify within-point repeatability and hysteresis; do not treat repeated samples as independent physical configurations.

## Uncertainty decomposition

Report separately:
- sensor/metrology uncertainty and covariance;
- test repeatability/reproducibility;
- map fitting/parameter uncertainty;
- interpolation/local support uncertainty;
- model discrepancy/residual uncertainty;
- configuration/asset variability;
- environmental/context uncertainty.

Do not collapse all terms into a single percentage without a justified propagation model. Avoid double-counting systematic calibration uncertainty across correlated points. Prediction intervals must state whether they are pointwise or simultaneous and whether they are frequentist bounds, Bayesian credible intervals or conservative engineering envelopes.

## Required metrics

### Point and regional prediction
- efficiency MAE/RMSE in absolute percentage points and relative terms;
- input power bias/RMSE, because energy decisions depend on power not only efficiency;
- torque/speed-stratified residuals, especially low-torque and high-speed/high-torque corners;
- peak efficiency and loss-ridge location error;
- integrated mission energy bias/RMSE using a frozen operating-point trajectory.

### Uncertainty quality
- empirical 50/80/90/95% interval coverage and average interval width;
- coverage by torque/speed region and by held-out asset/configuration;
- interval calibration error and proper scoring rule (e.g. interval score/CRPS when distributional);
- false-safe rate when lower efficiency bounds are used for power sizing;
- abstention/OOD precision and recall on held-out patches and context changes.

### Physical/structural checks
- efficiency in (0,1] over the physically defined map domain;
- non-negative losses at all declared motoring points;
- no invented support beyond measured/qualified domain;
- continuity and gradient diagnostics at cell boundaries;
- monotonicity only where the specific motor physics/design implies it—do not impose a universal monotonic constraint on efficiency surfaces;
- deterministic replay and exact map/evidence digest identity.

### Decision and test-budget value
- mission energy error and robust bus-power sizing error;
- architecture-ranking regret under the frozen candidate portfolio;
- number of measurements required to meet a preregistered uncertainty/decision threshold;
- test time/cost and value-of-information at equal budget.

## Proposed internal goals

These are research objectives, not universal standards or achieved results:
- Reproduce a bilinear-grid baseline with 100% agreement on synthetic analytical fixtures.
- On an independent physical holdout, achieve regional efficiency prediction error at or below the measured metrology floor; do not force a target tighter than instrumentation permits.
- For supported operating regions, target 95% empirical coverage for a preregistered 95% interval, reported by region and asset.
- For high-consequence mission-weighted regions, target integrated DC energy bias within ±3% and RMSE below 5% on a sealed mission holdout, subject to input/model uncertainty.
- Zero false-safe promotions in the frozen negative-control set.
- Demonstrate a ≥20% reduction in required measurement count or test time relative to a frozen regular-grid baseline at equivalent held-out predictive/uncertainty quality (stretch goal; not a test pass until budget and scoring are preregistered).
- A more complex method must beat bilinear interpolation on a held-out physical metric or measurement-budget metric, with uncertainty intervals; otherwise keep bilinear as the preferred simple baseline.

## Experimental design

Compare:
- uniform/space-filling grid;
- low-discrepancy space-filling designs;
- sensitivity-guided sampling;
- maximum predictive uncertainty;
- decision-value sampling (expected reduction in architecture regret / mission infeasibility uncertainty per test cost).

Prevent adaptive sampling from contaminating the final test set. Every proposed physical test point remains subject to independent test-envelope approval and facility procedure.

## Integration

The winning qualified map is exported with:
- exact map-content digest;
- source evidence-manifest digest;
- configuration and context digests;
- grid/representation metadata;
- method and software revision;
- support domain and OOD policy;
- uncertainty decomposition/calibration manifest;
- training/holdout split and test campaign identities.

The strict map oracle (#11577) remains the deterministic consumption boundary. The drive map-envelope consumer (#11590) propagates lower/nominal/upper efficiency to DC demand. The thermal loss ledger (#11593) remains a separate allocation boundary. No fitted map is silently installed into aircraft mission analysis without exact identity and domain match.

## Claim ceiling

Synthetic analytical success establishes only algorithmic correctness. Held-out laboratory map performance establishes only the declared motor/configuration/context. It does not establish another motor, production lot, aircraft installation, battery pack, mission or airworthiness. Model superiority must be shown on matched held-out physical data and measurement budgets, not asserted from training fit or a different benchmark.

---

## Immediate sequence

1. Freeze the map schema and digest semantics in #11577, including separate map-content and source evidence-manifest digests.
2. Acquire a qualified motor/inverter map dataset with raw synchronized terminal measurements, calibration, configuration and operating-context identity.
3. Reproduce a regular-grid bilinear baseline from raw observations and establish held-out torque-speed patches.
4. Add an offline research adapter for competing interpolators without changing the Rust production/runtime dependency boundary.
5. Compare uncertainty coverage and mission-integrated power/energy, not only pointwise efficiency MAE.
6. Freeze the final test campaign before decision-aware sampling; do not use sealed holdout regions to select new test points.
7. Promote only the exact map artifact, model revision, support domain and evidence manifest that passed the preregistered evaluation.

## External evidence comparator

- Roshandel et al. (2023), efficiency-map methods and accuracy/computational-cost trade-offs: https://doi.org/10.1109/TIA.2022.3210077
- Experimental uncertainty mapping, including torque-dependent low-load uncertainty: https://pmc.ncbi.nlm.nih.gov/articles/PMC11315023/
- Flight-validated small-UAS powertrain model, battery voltage prediction within 5% over a 5+ minute mission despite component efficiency errors up to 10%: https://www.mdpi.com/2226-4310/11/1/16
- SAE 2026-01-0744, modular measurement-based powertrain digital twin: https://saemobilus.sae.org/papers/development-a-digital-twin-electrical-powertrain-a-testbench-environment-efficiency-estimations-2026-01-0744
- 2026 multivariate global sensitivity analysis of electric-machine efficiency maps: https://doi.org/10.1109/TEC.2026.3687758

These comparators are not directly matched aircraft performance claims. The test article, component boundary, mission, measurement uncertainty, and validation split must be matched before quantitative comparison.
