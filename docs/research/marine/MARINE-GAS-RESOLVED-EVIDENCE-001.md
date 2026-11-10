# MARINE-GAS-RESOLVED-EVIDENCE-001
## CH4/N2O measurement integrity and engine-resolved emissions evidence

**Status:** research specification v0.1.0  
**Created:** 2026-11-10  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** MARINE-TRIAL-EVIDENCE-REGISTRY-001 (#11430), MARINE-REGULATORY-ROUTE-001 (#11469)

> This is an evidence and benchmark specification. A schema-valid record is not a verified emissions report, statutory compliance result or sensor-authenticity claim.

## 1. Paradigm shift: make each gas a measured, attributable engineering output

Marine energy projects commonly optimize fuel and CO2 while methane slip and nitrous oxide are handled through fuel/engine factors or separate measurement work. IMO MEPC 84 now provides three distinct methodological references:
- MEPC.414(84): test-bed and onboard measurements of CH4 and/or N2O from marine diesel engines;
- MEPC.415(84): engine-load monitoring and calculation of emission values;
- MEPC.416(84): continuous emission monitoring systems (CEMS) for CH4 and/or N2O.

Luminous should not collapse these into one undifferentiated CO2e number. Capture per-engine, per-gas mass rate, operating load, measurement method, calibration, timebase, trace identity, uncertainty and source rights. Integrate these outputs with route, fuel batch and delivered cargo service only after the measurement record is qualified.

The three methods are not interchangeable. CEMS/test-bed/onboard observations carry analyzer/calibration evidence; engine-load monitoring carries engine-load input traces, calculation method and model identity. Their disagreement is a benchmark result, not something to average away.

## 2. New evidence contract

The initial v0.1.0 contract is committed at:
- `marine-emissions-evidence-v0.1.0.schema.json`
- `fixtures/synthetic-gas-emissions-v0.1.0.json`

It requires:
- vessel configuration and engine identity/revision/rated power/fuel;
- an explicit method and IMO resolution reference;
- RFC3339 measurement window, timebase, sample-rate and completeness;
- expected gas set, observed gas results and explicit unmeasured gas reasons;
- each gas's mass rate in kg/h and either quantified uncertainty or explicit unknown reason;
- measured evidence (instrument, calibration, status, timebase, raw trace digest, rights) or engine-load-calculated evidence (load trace, calculation method, model identity, input source, rights);
- limitations and record status.

An unmeasured gas is not a numeric zero. Unknown uncertainty remains unknown. Digests identify supplied bytes; they do not establish measurement authenticity.

## 3. Falsifiable programme

1. Compare MEPC.414 measurement and MEPC.416 CEMS results on matched engine/load/fuel/ambient windows where a qualified facility supports both methods.
2. Compare MEPC.415 engine-load calculated values against measured values, preserving model residual and condition strata.
3. Repeat by engine load, transient/stationary state, fuel batch, temperature, after-treatment state and engine family.
4. Preserve start/end clock alignment, sample gaps, calibration range/validity, analyzer drift and detection limit.
5. Report gas-specific bias, MAE, p95 absolute error, relative error only where denominator is valid, uncertainty coverage, repeatability and detection-limit/censoring.
6. Do not convert to CO2e without an explicitly versioned GWP basis, gas mass integration boundary, time horizon and applicable regulatory/accounting method.
7. Do not treat a synthetic fixture, vendor datasheet or engine certification cycle as a full-voyage in-service label.

## 4. Proposed targets

Targets are research aspirations, not results or requirements:
- 100% expected-gas disposition: each expected gas is observed or explicitly unmeasured with reason.
- 0 calibration-out-of-tolerance records accepted as eligible measurements.
- 0 unknown uncertainty silently replaced with zero.
- 0 gas measurements without method/resolution, engine identity, timebase and trace/model lineage.
- 100% known-answer integration agreement for mass-rate × interval arithmetic.
- Inter-method bias remains within predeclared combined measurement uncertainty on qualified matched bench conditions.
- 90% uncertainty interval target coverage of 85–95% on independent holdout windows, with interval width and load-stratified coverage published.
- Zero false-complete cases in negative controls for missing gas, stale calibration, mismatched engine, missing trace, malformed window, bad sample coverage and unlinked calculation method.

Do not claim a universal percentage accuracy target until analyzer class, detection range, engine type, operating point and method are declared. Measurement limits and uncertainty differ by pollutant and instrument.

## 5. How this could exceed current practice

The opportunity is not to claim better gas analyzers or replace IMO measurement procedures. It is to join the otherwise fragmented evidence:
- trace gas rates to engine/load/fuel and the exact vessel configuration;
- compare CEMS, test-bed/onboard and engine-load calculated values as distinct methods;
- propagate calibrated uncertainty through voyage aggregation;
- connect emissions to route segments and the applicable ETS/FuelEU/IMO rule revision;
- normalize environmental impact by delivered cargo work without replacing statutory denominators;
- publish a reproducible chain from engine measurements to voyage and fleet outcomes.

This complements MEPC 414/415/416 rather than inventing a competing standard.

## 6. Claim ceiling

A passing evidence audit means only that the declared record is structurally and semantically complete under this schema. It does not prove analyzer calibration validity, true acquisition, statutory compliance, emissions superiority, or class/flag acceptance. Measurement procedure, independent verification and legal applicability remain with qualified test facilities, verifiers, operators, class and flag authorities.

## 7. Sources

- IMO MEPC 84 resolutions 414(84), 415(84), 416(84): https://www.imo.org/en/knowledgecentre/indexofimoresolutions/pages/mepc-2026-27.aspx
- IMO MEPC 84 meeting outcome: https://www.imo.org/en/mediacentre/meetingsummaries/pages/temporary-mepc-84th-session.aspx
