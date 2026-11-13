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

The ten synthetic scenarios exercise:
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

The evaluator checks run/specimen identity, exact device configuration digest, required telemetry channels, raw-trace and calibration identity, calibration status, sensor time offset, complete grid-to-service energy boundary, downstream test identity, synthetic-vs-physical evidence boundary, OOD material abstention, group-holdout provenance and virtual-metrology false accepts/rejects.

Twelve Rust unit tests are authored. Independent source-side JavaScript replay of the fetched 10-scenario JSON matched 10/10 frozen dispositions: one PASS, three REJECTED and six BLOCKED. The deliberately high-confidence IN_SPEC prediction with a synthetic bond below the acceptance threshold is explicitly caught as a false accept and REJECTED.

**Execution status:** all three branch files are fetched back; fixture JSON parses; independent replay matched all ten dispositions. The current authoring environment has no Cargo/Rust toolchain, so the Rust module and 12 tests remain uncompiled/unexecuted. No trained model, original industrial data, measured process prediction, scrap reduction or quality-release performance is claimed.
