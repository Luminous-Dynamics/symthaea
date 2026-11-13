# AIRROAD high-voltage insulation ageing evidence contract

**Programme:** AIRROAD-HV-INSULATION-001  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11569  
**Status:** research contract; no physical ageing data, service-life model or certification evidence is claimed

---

## Parent and composition

- AIRROAD measured propulsion maps and test provenance: [#11529](https://github.com/Luminous-Dynamics/symthaea/issues/11529).
- Independent drive-chain reference and kinematic extension: [#11509](https://github.com/Luminous-Dynamics/symthaea/issues/11509), [PR #11511](https://github.com/Luminous-Dynamics/symthaea/pull/11511), [PR #11568](https://github.com/Luminous-Dynamics/symthaea/pull/11568).
- Mission electro-thermal feedback and TMS architecture: [#11411](https://github.com/Luminous-Dynamics/symthaea/issues/11411), [#11428](https://github.com/Luminous-Dynamics/symthaea/issues/11428), [#11467](https://github.com/Luminous-Dynamics/symthaea/issues/11467).
- Cross-fidelity validation and evidence promotion: [#11445](https://github.com/Luminous-Dynamics/symthaea/issues/11445).

## Purpose

Establish an evidence-qualified engineering contract for high-voltage insulation-system degradation in electric and hybrid-electric aircraft propulsion. Cover electrical machines, inverter/drive assemblies, cables, connectors, busbars, feedthroughs and relevant interfaces, while keeping component design and physical test authority with their existing owners.

This issue owns the **degradation evidence model, stress-history provenance, diagnostic/inspection observations, uncertainty, life-model validation boundaries and maintenance evidence interfaces**. It does not design a high-voltage system, issue test voltages or procedures, operate a test rig, replace approved standards, determine service-life limits, authorize continued operation or make a certification finding.

## Why this deserves a separate programme

High-voltage insulation is exposed to coupled stressors that are not represented by a steady-state efficiency map:

- operating voltage and waveform (AC, DC and PWM);
- repetitive switching transients and dv/dt;
- temperature and thermal cycling;
- reduced pressure / altitude;
- humidity and contamination;
- mechanical vibration, strain, motion and connector fretting;
- manufacturing defects, assembly variation, ageing and maintenance history.

An efficiency map may be excellent while the insulation system remains unqualified for its electrical/environmental duty. A motor/inverter test that records only power, current and temperature does not automatically establish partial-discharge behaviour, dielectric lifetime or fault containment.

## Current standards and research references (verify against controlled editions)

- **SAE AIR7374 (issued 2024-02-16):** information report describing electrical-insulation ageing mechanisms and indicators for high-energy aircraft systems. It provides background, not an acceptance test or service-life certificate.
- **SAE ARP7375 (issued 2026-08-24):** Aerospace Recommended Practice with test guidance for assessing electrical-insulation material ageing in high-voltage systems, including AC, DC, PWM and operational-altitude context.
- **EUROCAE ED-344 (published 2026-08-25):** process specification / test guidelines for assessing high-voltage insulation ageing under combined temperature, reduced pressure, humidity and other environmental stresses.
- **EUROCAE ED-332 (published 2025-01-22):** guidance for aircraft high-voltage power quality; use its scope and applicable system boundaries rather than assuming generic terrestrial power-quality limits transfer to aircraft.
- **SAE AIR6127:** aerospace higher-voltage and partial-discharge/component-interaction information report; consult the controlled current edition and applicable architecture context.

Public descriptions:
- https://saemobilus.sae.org/standards/air7374-aging-mechanisms-electrical-insulation-materials-a-high-energy-system
- https://saemobilus.sae.org/standards/arp7375-test-guidelines-assessment-electrical-insulation-material-aging-a-high-voltage-system
- https://www.eurocae.net/product/ed-344-process-specification-test-guidelines-for-assessment-of-electrical-insulation-material-ageing-in-a-high-voltage-system/
- https://www.eurocae.net/new-publication-ed-332-guidance-for-aircraft-high-voltage-power-quality/
- https://saemobilus.sae.org/standards/air6127-managing-higher-voltages-aerospace-electrical-systems

The programme must not reproduce copyrighted standards or invent their detailed acceptance criteria. Controlled editions, the aircraft certification basis and qualified test-facility procedures govern any physical work.

## Paradigm shift

Move from **one-time dielectric pass/fail at component release** to **lifecycle insulation evidence**:

```
design + material + process + assembly identity
    -> approved stress/test plan
    -> traceable baseline diagnostics
    -> controlled environmental/electrical stress history
    -> repeat diagnostics + uncertainty
    -> degradation trend and model applicability
    -> configuration-specific maintenance / inspection evidence
    -> independent engineering and airworthiness review
```

A second shift is from **accelerated-test hours treated as equivalent service hours** to **mechanism-aware life inference**. Accelerated ageing is meaningful only when the induced degradation mechanisms are shown to remain representative of the service environment. Extrapolation to aircraft life must carry model-form uncertainty, environmental mission spectrum, test censoring and competing failure modes.

A third shift is from **threshold-only condition monitoring** to **degradation observability**: trends, measurement uncertainty, configuration lineage and evidence gaps are visible before a maintenance decision. This is engineering decision support, not autonomous release-to-service authority.

## Scope and evidence boundaries

Cover where applicable:

1. Electrical machine winding insulation, slot/turn/phase insulation, impregnation and end-winding support.
2. Inverter internal insulation, DC-link structures, switching devices' isolation boundaries, encapsulation and coolant-adjacent interfaces.
3. HV cables, terminations, connectors, busbars, contactors, feedthroughs and enclosure interfaces.
4. Cooling-fluid, moisture, contamination and condensation interfaces that can affect insulation condition.
5. Altitude/pressure and environmental conditions relevant to the declared aircraft operating envelope.
6. Manufacturing process, lot/material identity, assembly workmanship, repair and rework history.

Keep the measurement locations and diagnostic methods explicit. Candidate observations may include partial-discharge inception/extinction and patterns, insulation resistance, leakage current, dielectric response/loss, withstand or proof-test outcomes, visual/microscopic findings and sensor/inspection diagnostics. The applicable controlled standard and qualified procedure decide which methods, settings and acceptance limits are appropriate; this issue does not prescribe hazardous test settings.

Separate:
- direct physical observation;
- inferred degradation state;
- manufacturer-reported limit;
- published literature;
- accelerated-test result;
- fitted life model;
- synthetic fixture;
- engineering target.

Never convert one evidence class into another without a documented promotion review.

## Data and lifecycle contract

Every specimen, component and test/inspection event must bind to:

- unique asset/specimen and material/lot identifiers;
- exact design, winding, insulation system, manufacturing process and assembly/rework revisions;
- electrical architecture, nominal/maximum declared voltage envelope, waveform family and switching configuration where applicable;
- environmental and thermal history, pressure/altitude, humidity, contamination and vibration context;
- approved test plan and procedure revision, facility, responsible authority and safety review;
- instrument/channel identity, measurement location, calibration traceability, bandwidth/range, synchronization and measurement uncertainty;
- raw-data digest, acquisition metadata, analysis/transformation version and exclusions with reasons;
- diagnostic method, detection threshold/limitations as defined by the controlled procedure, noise floor and false-positive/false-negative characterization;
- stress exposures, interruptions, censoring, failed specimens and survival/competing-event records;
- model version, fitted parameters, covariance, training specimen identities and independent validation cohort;
- inspection/repair/replace decisions, reviewer identity and disposition authority.

Preserve raw readings and the full exposure history. Do not replace missing observations with a nominal state. A missing calibration, mismatched specimen identity, invalid test procedure revision or unbounded uncertainty yields UNKNOWN / non-PASS.

## Model and validation rules

- Do not assume a single universal Arrhenius or voltage-acceleration model. Select a mechanism-appropriate model only when justified by the data and controlled reference.
- Record which stressors are coupled, independent, sequential or competing; avoid adding unrelated acceleration factors without validation.
- Distinguish material coupon evidence from representative insulation-system, component and installed-aircraft evidence.
- Account for censored specimens, surviving specimens, competing failure mechanisms and lot-to-lot variation.
- Fit only on designated training cohorts; validate on held-out lots/specimens and, where possible, independent facilities.
- Report parameter uncertainty, confidence/prediction intervals, calibration domain and out-of-domain status.
- Track diagnostic repeatability, detection probability and false indication rates; an undetected defect cannot be treated as absent without a declared detection model.
- Correlate degradation indicators with destructive or higher-authority inspections where approved and available.
- Reject extrapolation beyond supported voltage waveform, altitude/pressure, temperature, humidity, specimen construction and stress duration.
- Never treat a passed short-duration proof test as a quantitative lifetime demonstration by itself.

## Benchmarks and proposed targets

These are internal engineering targets, not standard requirements, physical claims or regulatory acceptance criteria.

### Tier 0 — provenance and replay
- 100% of derived degradation claims link to specimen identity, revision, raw-data digest, procedure and calibration evidence.
- 100% known-answer synthetic fixtures reproduce the expected diagnostic/state transitions.
- Zero stale-configuration, missing-calibration or unsupported-domain results promoted as PASS.
- Deterministic replay of the complete data transformation and model result from frozen digests.

### Tier 1 — diagnostic repeatability
- Report repeatability and reproducibility by method, operator, fixture and facility.
- Quantify detection probability and false-alarm probability over a declared defect/condition corpus.
- Report measurement uncertainty relative to decision thresholds; no accuracy target may be tighter than the instrumentation supports.
- Preserve an explicit UNKNOWN result when diagnostic sensitivity or configuration identity is inadequate.

### Tier 2 — degradation model validation
- Nominal prediction-interval coverage preregistered (e.g. 90% and 95%) and assessed on held-out specimens/lots.
- Report calibration error, likelihood/scoring metrics, residual structure and time-to-event error, not only one aggregate R².
- Zero unsupported acceleration extrapolations in the frozen negative-control corpus.
- Publish false-safe and false-replacement rates with confidence intervals and consequence classes.

### Tier 3 — lifecycle decision support
- Demonstrate stable inspection/maintenance ranking on held-out assets and mission stress histories, or abstain where ranking is not robust.
- Quantify inspection burden, false removal cost, unplanned maintenance exposure and uncertainty in replacement timing.
- Any operational disposition remains with the approved maintenance organisation / responsible airworthiness authority.

Do not set a universal acceptable partial-discharge level, dielectric margin, service life or inspection interval in this issue. Those depend on insulation system, component, operating environment, test method, design data and applicable approved basis.

## Safety and authority boundary

This issue is an evidence and engineering decision-support programme. It shall not:
- provide direct instructions for energizing high-voltage equipment;
- command or interlock physical test equipment;
- replace facility containment, isolation, discharge, lockout/tagout, emergency shutdown or qualified-person controls;
- bypass equipment protection, aircraft protection, BMS, inverter protections or maintenance release controls;
- autonomously authorize test, continued operation, maintenance release or flight.

All physical stress/ageing tests require qualified facilities, approved procedures, competent personnel, documented risk assessment and independent authority. No test voltage, waveform, dwell time or acceptance limit is inferred by this repository work.

## Initial work packages

1. Standards applicability matrix with edition/date, scope, component family, evidence class and controlled-copy owner.
2. Canonical insulation specimen/component identity and stress-history schema.
3. Synthetic diagnostic corpus and typed UNKNOWN/OOD/non-PASS handling.
4. Degradation-model evaluation protocol with lot-held-out validation and censored-data treatment.
5. Lifecycle evidence adapter linking propulsion configuration and operating history to inspection/maintenance records.
6. Review by qualified electrical insulation and aircraft certification specialists before any physical programme or compliance claim.

## Claim ceiling

A versioned schema or synthetic benchmark validates only data semantics and software behavior. Coupon ageing data do not qualify a motor. Component tests do not qualify installed wiring or an aircraft. A validated life model does not itself authorize maintenance intervals or continued airworthiness. Only the responsible design, maintenance and regulatory authorities determine what evidence and findings are accepted.

---

## Implementation order

1. Freeze the standards applicability matrix using controlled document editions and responsible owner review.
2. Define versioned specimen/component identity, stress-history and diagnostic-event schemas.
3. Add synthetic known-answer and negative-control fixtures for provenance, calibration, stale configuration and OOD handling.
4. Define a lot/specimen-held-out degradation-model comparison protocol, including censored observations and competing failure modes.
5. Connect qualified component stress histories to propulsion configuration and maintenance records without inferring a release-to-service disposition.
6. Obtain qualified electrical-insulation, test-facility and certification review before specifying or conducting any physical test campaign.

## Evidence promotion gates

- **Schema/synthetic:** proves only the recorded data semantics and software behavior.
- **Coupon:** characterizes only the declared material/specimen under the tested stress and procedure.
- **Insulation system/component:** requires representative construction, manufacturing identity and independent correlation.
- **Installed configuration:** requires exact aircraft wiring, routing, enclosure, cooling, altitude and electrical waveform context.
- **Lifecycle decision:** requires validated model applicability, inspection uncertainty and responsible maintenance/airworthiness review.

No stage automatically promotes the preceding evidence to a service-life claim or aircraft approval.
