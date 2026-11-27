# EV Battery Observation Contract v0.1

**Status:** Synthetic software contract and verifier proposal; not automotive-qualified.  
**Owner:** Symthaea EV-HEALTH-ADVISORY-001 #11239.  
**Integration:** Symtropy ROAD-EV-REFERENCE-001 #4346; Mycelix Mobility Configuration #3709.

## Why this is the first implementation slice

The existing Symthaea grid-physics Battery is a simple grid-scale energy-storage model. It is not an automotive pack/BMS estimator. The existing digital-twin crate is a generic telemetry and health-estimate substrate, not a battery-specific source-of-truth. This package adds only the missing observation boundary and a deterministic audit; it does not fork either model.

## Observation versus inference

Every record carries an explicit measurement class:

- direct sensor observation;
- BMS-reported value;
- derived estimate;
- model prediction;
- simulation output;
- synthetic fixture;
- vendor claim.

A cryptographically authentic BMS message is still a BMS-reported value. It is not automatically calibrated sensor truth. A valid-shaped estimate is not counted as a direct observation.

## Identity and resolution

The batch binds the exact vehicle configuration digest, pack configuration digest, BMS hardware/firmware revision and chemistry declaration. Each observation names its pack/module/cell subject, and the batch declares both a scope inventory and the maximum resolution it can actually observe.

The auditor rejects invented cell-level identity, a subject absent from the scope inventory, and a pack reference that disagrees with the asset record. Pack-level telemetry cannot be expanded into fabricated module/cell readings.

## Time and calibration

Every sample carries sample time, receive time, monotonic sequence, sensor identity and quality state. Direct sensor observations bind to a batch-registered calibration identity, digest and validity interval. A clock uncertainty and maximum transport delay are explicit.

Out-of-order, stale, clock-skewed or calibration-invalid observations may remain in the record only when they are explicitly quarantined with the matching reason flag. They are withheld from the usable direct-observation count. A valid label cannot override the evidence.

## Missingness and units

Unavailable measurements use a null value plus an unavailable state and missing_value flag. Numeric zero is never a missing-value sentinel. Units are strict UCUM-style tokens for the channels in v0.1: V, A, Cel, 1 and kW. SOC is a fraction in [0,1]. Current uses positive-discharge sign convention at the batch level.

The contract validates syntax, identity consistency, basic time/calibration lineage, stream order and unit/scope semantics. It does not prove the sensor was truthful, correctly installed, electrically connected or physically calibrated. Those require independent instrumentation and physical checks.

## Frozen synthetic corpus

The initial corpus contains 18 fixtures: 5 accepted shapes and 13 expected rejections. It includes:
- pack and module resolution;
- stale-but-retained and explicitly unavailable readings;
- a BMS-reported SOC value kept separate from direct sensor observations;
- wrong pack identity and unsupported cell-level claims;
- unit mismatch, duplicate observation ID and out-of-order stream;
- stale-valid, unknown calibration, expired calibration and clock-skew cases;
- missing-as-zero, unexplained suspect state, fault-flagged valid state and missing calibration for a direct observation.

Synthetic fixtures exercise the contract only. They are not battery measurements, calibrated SOH labels, fault-injection results on hardware, safety evidence or model-accuracy results.

## 2026 research evidence and benchmark implications

The following results are recent external research anchors, not results produced by Symthaea:

- Barua et al. (2026), *Predictor Structure Modulates Validation Inflation and Cell-Transfer Reliability*: on four NASA cells, median random 80:20 split RMSE improved from 5.00 to 1.52 SOH points with richer predictors, while median leave-one-cell-out RMSE moved only from 5.83 to 4.61. This shows how within-cell validation can materially overstate transfer performance in that corpus; the four-cell result is not a universal error law. https://www.mdpi.com/2313-0105/12/9/342
- Wu et al. (first published 1 October 2026), *Uncertainty-Aware Transformer*: four NASA PCoE cells; its 95% interval coverage was near nominal on in-range held-out cells but fell substantially on cells aging beyond the training range. The paper reports 0.984 / 0.945 PICP on two in-range held-out cells and 0.735 / 0.660 on two shifted cells; its within-cell test PICP was only about 0.653. This is a direct argument for reporting calibration by domain and test protocol rather than only pooled interval coverage. https://journals.sagepub.com/doi/full/10.3233/ATDE260884
- Applied Energy (1 November 2026), *Geometry-aware curation for data-efficient SOH estimation*: reported a cross-vehicle study spanning 400 EVs and more than 65,000 charging segments; a geometry-aware curation method retained 10–20% of training data while reporting RMSE below 2% and large training-time reductions. Treat this as a study-specific data-efficiency baseline, not an independently reproduced universal result. https://doi.org/10.1016/j.apenergy.2026.128284
- Rizanov et al. (15 July 2026), *Toward Trustworthy and Transferable SOH/RUL Estimation*: proposes a multi-fidelity validation ladder from laboratory cells through modules/packs and fleet shadow operation, with calibration, transfer and deployment criteria. https://www.mdpi.com/2313-0105/12/7/255
- Battery-Hub indexes public battery datasets across cycling, operational traces, EIS and degradation tasks. Use it for discovery, but evaluate the exact original source, revision, label definition, licence and redistribution conditions for every dataset before admission. https://battery-hub.org/datasets

### Strategic consequence

The research target is not "lowest NASA RMSE." It is **calibrated transfer with useful abstention**: maximize coverage of decisions that meet a predeclared error/risk budget, while widening intervals or abstaining when the operating domain changes. Compare data curation methods at matched held-out performance and report data fraction, compute, memory, energy and update cost. A small calibrated model that is correct about its limitations may be more useful than a larger model with silent extrapolation.

## Benchmark protocol to implement next

Before fitting an estimator, freeze a source/rights registry and dataset manifests. Initial dataset candidates include NASA PCoE, CALCE, Oxford, Severson/MATR, XJTU and rights-cleared real-world fleet/BMS records where available. Dataset names alone do not grant reuse or redistribution rights.

Use these mandatory splits:
1. chronological within-cell/pack;
2. leave-one-cell-out;
3. leave-one-pack/vehicle-out;
4. leave-one-temperature/protocol-out;
5. chemistry holdout only when there are enough independent cells and genuinely distinct chemistries;
6. cross-lab transfer where label and measurement protocols are harmonized and provenance supports it.

Never randomly split adjacent windows from the same cell and call that cross-cell generalization. Retain censored lives and missing labels. Record public-test exposure and hyperparameter/leaderboard use.

Baselines must include coulomb counting plus OCV correction where assumptions hold, a simple equivalent-circuit/thermal estimator, linear/ridge regression and a tree-based regressor before TCN/Transformer/Mamba/physics-informed models. A complex model earns its place by improving held-out cell/pack performance, uncertainty calibration, failure detection or compute/energy—not by parameter count.

Report:
- SOH MAE/RMSE in percentage points, worst-cell and worst-domain error;
- SOC error and long-rest/high-C-rate drift;
- calibration coverage and sharpness for nominal 90%/95% intervals;
- selective risk versus coverage as the model abstains on unsupported inputs;
- OOD detection AUROC/AUPRC plus false alarms at fixed operating-hour rates;
- diagnostic recall per predeclared fault family and false alerts per cell/vehicle-hour;
- RUL interval coverage and a censoring-aware metric, with censored observations retained;
- compute latency P50/P95, memory, energy per inference and offline capability.

A single pooled MAE or random-cycle split is insufficient. Numeric gates must be preregistered only after dataset label resolution, sample size and measurement uncertainty are known. The previously proposed SOH MAE <=3 percentage-point goal remains a research hypothesis, not an acceptance criterion until those conditions are established.

## Authority boundary

The verifier is a data-contract audit. It does not estimate SOC/SOH, diagnose a physical pack, certify a BMS, determine whether a battery is safe, or authorize charging/driving. No output grants actuator authority. Certified/local BMS protection, fuses, thermal cut-outs, contactors and independently engineered safety controls remain authoritative.

## Reproduce

```bash
cargo test --manifest-path tools/ev-battery-observation-audit/Cargo.toml --all-targets
cargo run --manifest-path tools/ev-battery-observation-audit/Cargo.toml -- \
  --schema schemas/transport/ev_battery_observation_batch_v0_1.schema.json \
  --corpus fixtures/ev_battery/observation_batches_v0_1.json
```
