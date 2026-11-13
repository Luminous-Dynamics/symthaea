# PLASMA-VM-001 — Telemetry-to-product virtual metrology for plasma surface processing

**Date:** 13 November 2026  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11847  
**Parents:** PLASMA-SURFACE-001 #11795, PLASMA-ML-GENERALIZATION-001 #11839, MICROPLASMA-001 #11780  
**Status:** synthetic-only contract/evaluator tranche  
**Claim ceiling:** no trained model, measured process data, manufacturing release, quality improvement or control authorization.

## 1. The real differentiation opportunity

Industrial suppliers already measure plasma optical emission, power/voltage, process-gas flow, pressure, temperature, rotation and equipment state. Monitoring by itself is not novel.

The next research question is whether those upstream signals, joined to exact material/adhesive/lot context, can predict the actual downstream product result—adhesion distribution, failure mode, aging, substrate preservation and scrap risk—on new lots, devices and production days.

This is **outcome-linked virtual metrology**, not plasma-state monitoring and not the literature-only WCA regression benchmark.

## 2. Literature and industry context

The 2026 review *Machine Learning-Enabled In Situ Diagnostics for Intelligent Plasma-Based Semiconductor Manufacturing* reports virtual metrology, early fault detection and predictive maintenance, while identifying limited labeled data, interpretability, poor domain transfer and lab-to-fab transfer as barriers. It is semiconductor plasma evidence; those models do not automatically transfer to atmospheric polymer surface activation. DOI: https://doi.org/10.1021/acsami.6c04768.

A 2026 manufacturing-metrology review highlights metrological traceability, standardized uncertainty quantification, explainability and reproducibility as essential to reliable ML-assisted measurement. DOI: https://doi.org/10.1016/j.cirp.2026.04.109.

Our industrial comparator map records PCU/SPM telemetry already sold by Plasmatreat and Surfx. The differentiator cannot be an OES probe, a dashboard or more sensor channels.

## 3. Paradigm shift: the process can be nominal while the product is not

An unchanged plasma signal can produce different outcomes after changes in:
- polymer grade, lot, contamination, additive package or surface finish;
- adhesive/coating formulation, lot, mix/cure and bondline;
- cleaning, storage, humidity, treatment-to-bond delay;
- nozzle/head wear, geometry, standoff, traverse and overlap;
- calibration, sensor drift, clock synchronization or utility boundary.

Therefore, a virtual metrology system needs the complete run-to-specimen lineage, not a time-series classifier alone.

The primary product targets are:
- probability distribution for downstream bond performance;
- failure-mode risk;
- predicted bondability-window retention;
- post-ageing performance and material-preservation risk;
- out-of-domain detection and calibrated abstention;
- measurable inspection, scrap and rework value.

A process-state alarm is not a product-quality certificate.

## 4. Required identity and data linkage

Every prediction must bind to:
- exact run ID, device/cell ID, configuration digest, recipe revision and timestamps;
- substrate grade and lot, surface history/conditioning and geometry;
- adhesive/coating identity and lot, preparation, cure/bondline/deposition details;
- sensor identity, quantity, unit, raw trace digest, calibration record/status/uncertainty and acquisition timestamp;
- observed electrical/optical/gas/flow/pressure/temperature/motion signals;
- complete grid-to-service energy-boundary identity;
- specimen IDs physically mapped to the process run;
- downstream test method, calibration, sample result, failure mode and aging timepoint;
- all missing, invalid, excluded and censored observations.

The model training partition must be grouped by material lot, device/configuration and production day. Reuse PLASMA-ML-GENERALIZATION-001 #11839 for group split and leakage safeguards; do not duplicate its evaluator.

## 5. Baselines

Compare:
1. existing vendor process alarms / recipe limits and the current QC sampling policy;
2. material + recipe only (no plasma telemetry);
3. plasma telemetry only (no material/adhesive/lot context);
4. joined material + process telemetry;
5. interpretable linear/tree baseline;
6. calibrated ensemble / uncertainty-aware model.

The baseline is not allowed to skip destructive QC on the training/qualification data. Initial deployment is shadow mode; model outputs remain advisory.

## 6. Critical metrics

**Quality:** false-accept rate (out-of-spec product predicted in-spec), false-reject rate, escaped defect rate, failure-mode sensitivity, aging failure risk.

**Uncertainty:** nominal 90%/95% interval empirical coverage on grouped holdouts, interval width, calibration by material lot/device/day, abstention coverage-risk curves.

**Transfer:** worst-lot/worst-device/worst-day result, leave-one-lot/device/day-out, model drift and OOD refusal.

**Economic value:** destructive inspections avoided, scrap/rework avoided, yield and line downtime, prediction latency, incremental sensor/calibration maintenance cost, full energy/utility boundary.

False acceptance is the dominant safety/quality loss. Do not optimize aggregate accuracy while hiding escaped defects.

## 7. Proposed targets — after the baseline is frozen

- Zero automated quality releases in initial stages; predictions are advisory.
- At the same false-reject/inspection budget, reduce false accepts relative to incumbent alarms/recipe limits.
- Nominal 90% prediction intervals achieve reported empirical coverage on held-out lots/devices/days, including per-group worst cases.
- At least 25% reduction in destructive inspection or scrap/rework cost only if escaped-defect risk does not increase.
- Missing calibration, stale/clock-skewed telemetry, configuration mismatch, unknown material/adhesive or OOD return HOLD / UNKNOWN.

These are falsifiable research targets, not present achievements or standards.

## 8. S0 synthetic contract

Fixture: `data/research/plasma_virtual_metrology_fixture_v0_1.json`.

The sixteen synthetic scenarios exercise:
- fully linked telemetry/outcome baseline;
- confident false accept;
- uncalibrated sensor;
- stale sensor clock;
- configuration digest mismatch;
- process-run/specimen outcome mismatch;
- unsupported material profile;
- incomplete energy boundary;
- missing downstream outcome label;
- unverified training group holdout.

All values are invented. They validate only linkage and refusal logic, not the learned relation between plasma state and bond strength.

## 9. Safety and authority boundary

- Initial model deployment is shadow mode only.
- No direct HV/RF/gas/motion setpoint actuation.
- No automatic product release or replacement of required physical QC.
- Missing/stale data, calibration drift, sensor conflict, OOD inputs and unknown material profiles cause abstention.
- Any future closed-loop control/quality release requires a separate process/safety/security validation and customer/authority approval.

## 10. Ownership

- PLASMA-SURFACE-001 #11795 owns the physical bondability endpoint and acceptance floor.
- PLASMA-ML-GENERALIZATION-001 #11839 owns grouped split/leakage and cross-source model transfer.
- PLASMA-VM-001 #11847 owns the joined telemetry-to-downstream-product contract and shadow-mode value benchmark.
- PLASMA-ENERGY-001 #11718 / #11758 own complete energy and metrology semantics.
- Existing FIELD / SE-OBS / ROB-INSTR / ROB-EXP owners retain measurement/calibration/experiment authority.

No duplicate calibration registry, generic ML engine, experiment optimizer, material registry, evidence receipt or hardware-control authority is introduced.


## Rust-native S0 contract audit

The synthetic fixture is checked by:

    cargo run -p xtask -- plasma-virtual-metrology-audit
    cargo run -p xtask -- plasma-virtual-metrology-audit --json

The evaluator checks run/specimen identity, exact device configuration digest, required telemetry channels, raw-trace and calibration identity, calibration status, sensor time offset, exact grid-to-service energy-boundary identity and completeness, downstream test identity, synthetic-vs-physical evidence boundary, OOD material abstention, group-holdout provenance and virtual-metrology false accepts/rejects.

Twenty Rust unit tests are authored. Independent source-side JavaScript replay of the fetched 16-scenario JSON matched 16/16 frozen dispositions: one PASS, six REJECTED and nine BLOCKED. The deliberately high-confidence IN_SPEC prediction with a synthetic bond below the acceptance threshold is explicitly caught as a false accept and REJECTED.

**Execution status:** all four implementation files are fetched back; fixture JSON parses; independent replay matched all ten dispositions. The current authoring environment has no Cargo/Rust toolchain, so the Rust module and 20 tests remain uncompiled/unexecuted. No trained model, original industrial data, measured process prediction, scrap reduction or quality-release performance is claimed.


### Frozen energy-boundary identity

The fixture root now freezes `GRID_TO_ACCEPTED_BOND_V1`. Every run must use the same complete grid-to-accepted-bond perimeter. A run that substitutes a generator-only or other narrower perimeter is REJECTED even if its local energy record says complete. A dedicated Rust test mutates the energy-boundary identity and verifies refusal. Independent replay still matches 10/10 scenario dispositions (1 PASS, 3 REJECTED, 6 BLOCKED).


## S0 hardening update — explicit holdout manifests and metered energy

The first review found that a boolean `training_group_holdout_verified` could not demonstrate that the test lot/device/day were absent from model training. The evaluator now requires a training-group manifest with:
- a manifest ID and partition-receipt digest;
- explicit grouping dimensions;
- explicit training material-lot, device and production-day identities;
- a disjointness check against the scenario's held-out lot/device/day.

A missing manifest is BLOCKED; an overlap is REJECTED. This is stronger than a boolean assertion, but it remains an S0 fixture contract: the digest is not by itself a cryptographic proof of model-training lineage. A future real-data qualification must bind the manifest to a reproducible training-data/fold receipt generated by PLASMA-ML-GENERALIZATION-001 and independently verify that receipt.

The energy contract now requires an individually identified meter record for every declared subsystem (power conversion, gas supply, cooling, extraction, motion, controls and standby/start-up). Each record carries energy, uncertainty, meter ID, calibration record/status/validity and raw-trace digest. The evaluator verifies:
- exact grid-to-accepted-bond boundary identity;
- complete subsystem coverage and no undeclared load;
- per-meter calibration valid at the process-run timestamp;
- non-empty raw trace and meter identity;
- finite, non-negative measurements / uncertainty;
- arithmetic closure of subsystem energy to the declared complete-system total.

Additional negative controls cover training-group overlap, a missing group manifest, future-dated telemetry, expired sensor calibration, missing energy meter identity, subsystem energy sum mismatch and an in-spec classification below the frozen probability threshold.

The signed sample offset now treats both stale and future-dated samples as outside the allowed run window. Sensor and energy-meter calibrations must be valid at the actual run timestamp, not just carry the string `QUALIFIED`.

The fixture now has 16 scenarios (1 PASS, 6 REJECTED, 9 BLOCKED) and the Rust source has 20 authored tests. Independent source-side replay matched 16/16 expected dispositions. These tests remain uncompiled/unexecuted in the current environment.

## 2026 virtual-metrology benchmark context

The August 2026 preprint *Virtual Metrology for Plasma Etching: A Two-Stage Framework for Sensor Data Compression and Prediction* surveys 105 plasma-etch virtual-metrology studies published from 2012–2025. It distinguishes sensor representation/compression from downstream regression, classification, endpoint and fault prediction, and reports unresolved issues including sparse labels, model drift, chamber transferability, uncertainty reporting, latency and integration with advanced process control. This is an industrial-plasma comparator, not evidence that etch models transfer to atmospheric polymer activation. DOI: https://doi.org/10.21203/rs.3.rs-10577324/v1. The accompanying structured literature dataset catalogues 105 contributions but is not a paired raw telemetry-to-bond dataset: https://doi.org/10.48366/R1907748.

A September 2026 plasma-etch optimization paper demonstrates heteroscedastic Gaussian-process uncertainty and reliability-based robust design optimization using remaining-thickness measurements at nine wafer locations. It reinforces the need to model input-dependent uncertainty and spatial product variation rather than report one global point error. Its wafer etch process and proprietary code/data are not a direct atmospheric surface-treatment baseline. DOI: https://doi.org/10.1080/0305215X.2026.2711103.

These sources sharpen our SOTA comparison:
- compare raw telemetry against compact sensor representations, not just model families;
- include material/lot/device/day transfer as a first-class test;
- evaluate input-dependent uncertainty and worst-group coverage;
- report prediction latency and label/inspection cost;
- test multi-output prediction only where downstream endpoints are jointly measured;
- compare predictive value against current alarm limits and QC policy, not against random guessing alone.

The 2026 reviews establish that virtual metrology itself is not novel. The narrow research opportunity is its validation for atmospheric surface treatment, where a qualified model must predict bond/coating outcomes from exact material and process context under independently verified holdout and metrology conditions.
