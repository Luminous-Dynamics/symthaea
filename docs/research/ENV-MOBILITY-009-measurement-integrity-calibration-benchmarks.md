# ENV-MOBILITY-009: Measurement Integrity and Calibration-Aware Sensor Benchmarking

**Status:** Research protocol and supplier leads; not a calibration certificate, purchase order, or demonstrated sensor result.  
**Parent work:** ENV-MOBILITY-001 through -008; SPM-0 dry-side environmental logger.  
**Decision rule:** No environmental claim is stronger than the measurement chain, calibration evidence, deployment conditions, and uncertainty budget that support it.

## 1. Paradigm shift: from cheap sensors to defensible measurements

A low-cost sensor network is not successful merely because it produces many readings. It is successful when a reader can determine what each reading means, how uncertain it is, whether the instrument was healthy, and whether a change is environmental or instrumental.

Build a **measurement evidence graph** for every accepted observation:

device identity + sensor/firmware/BOM revision + raw sample + timestamp/clock status + environmental context + calibration/reference run + correction version + uncertainty + QC disposition + maintenance history

Keep raw values immutable. Derived/corrected values must link to raw values and name the correction algorithm/version. Never silently overwrite the observation with a corrected value. If calibration evidence expires, a sensor is replaced, or drift exceeds its limit, flag the affected interval and downgrade the claim rather than retroactively treating it as valid.

This extends the manufacturing digital-thread idea to field measurement evidence: design, assembly, verification, calibration, deployment, repair, and data interpretation remain linked. NIST describes the digital thread as integrating product definition, manufacturing, and inspection data, while emphasizing conformance and data trust (https://www.nist.gov/programs-projects/digital-thread-manufacturing).

## 2. Define the measurand before choosing a sensor

Freeze the SPM-0 profile before requesting quotations or asserting accuracy.

Minimum profile fields:
- Measurands: ambient air temperature, relative humidity, and/or barometric pressure; explicitly state which are in scope.
- Intended operating envelope and deployment geometry.
- Required resolution, maximum permissible error, response time, sampling cadence, uptime, and service interval.
- Whether readings are exploratory, screening-grade, or used for a regulated/decision-critical purpose.
- Reference method, reference instrument identity, calibration points, stabilization criteria, uncertainty budget, and acceptance limits.
- Enclosure, airflow, self-heating, solar loading, condensation, and mounting effects that could bias measurements.

Do not equate digital resolution with accuracy. Do not assume a calibration of one bare sensor or one assembled unit applies to every unit or to a changed enclosure. Calibration is specific to a defined measurement result and chain; traceability requires documented references and contributions to uncertainty (NIST policy: https://www.nist.gov/metrology/metrological-traceability).

## 3. Sensor-profile discipline

Keep two provisional profiles separate until requirements decide otherwise:

- **SPM-0-P:** temperature + barometric pressure. A BMP280-family breakout is a candidate only after delivered-board identity and interface verification.
- **SPM-0-H:** temperature + relative humidity + barometric pressure. A BME280-family breakout is a candidate only after delivered-board identity, interface, and humidity response verification.

Do not compare their catalog subtotals as if they provide equivalent capabilities. A sensor module is not a complete calibrated instrument: enclosure, mounting, wiring, power, firmware, reference comparison, uncertainty, labor, delivery, taxes, and failures may change the actual cost and fitness.

## 4. Bench qualification protocol

Use a pre-registered protocol; freeze the acceptance limits before seeing results.

### A. Incoming inspection
1. Record supplier, order/invoice, delivered date, manufacturer/markings, board revision, photographs, and lot/serial identifier where available.
2. Check connector pinout, voltage domain, current draw, communication address, firmware identity, and sensor identity.
3. Preserve the raw bring-up log. A board that responds electrically has not yet demonstrated measurement accuracy.

### B. Repeatability and environmental comparison
1. Co-locate at least five assembled units with a suitable reference and log synchronized raw data. Five units is an initial engineering screen, not enough to establish production-wide reliability.
2. Test multiple predeclared stable conditions across the intended envelope, including a return to the initial condition to detect hysteresis/drift. Record chamber/fixture uniformity and stabilization criteria.
3. Randomize or rotate device positions to avoid confusing fixture gradients with unit-to-unit variation.
4. Capture reference readings, time alignment, ambient conditions, power state, enclosure configuration, and all exclusions.
5. Repeat after a controlled power cycle and after a defined repair/reassembly event.

If a controlled chamber or suitable reference is unavailable, label the work as a comparative screening test, not calibration or traceable accuracy validation.

### C. Analysis
For each measurand and test condition report:
- bias (mean sensor minus reference);
- MAE and RMSE, with units;
- maximum absolute error and 95th percentile absolute error;
- repeatability within each unit and between units;
- drift over the test duration and after power cycle;
- missing/corrupt sample rate and timestamp error;
- uncertainty contributions from reference, resolution, repeatability, stability, gradients, timing, and correction fitting;
- before/after correction performance, with held-out conditions or time blocks not used to fit the correction.

Do not report R² alone as evidence of agreement. Do not use a fitted correction and its training data as the sole validation. Publish failures and negative controls, not only passing runs.

### D. Calibration transfer and drift
A correction fitted to one unit or site must not silently transfer to another. Validate transfer by holding out complete units and, where possible, a site or environmental condition. Schedule periodic co-location checks and trigger review on sensor replacement, enclosure change, abnormal drift, or failed quality checks. Derive intervals from observed stability and use requirements, not a universal calendar rule.

## 5. Initial benchmark targets (proposed, not achieved)

These are **engineering targets to ratify after freezing the use case**, not claims about current hardware.

| Dimension | Initial target | Gate |
|---|---|---|
| Evidence completeness | 100% of accepted samples link to device, firmware/config, raw value, time, QC status and correction version | Hard gate; missing provenance cannot pass |
| Reference evidence | Every reported accuracy result names the reference, conditions, points and uncertainty | Hard gate |
| Temperature | Set a maximum error appropriate to the mission before test; a provisional screening target may be ±1.0 °C over a declared indoor/ambient envelope | No claim until reference test |
| Relative humidity | Set a mission-specific limit before test; provisional screening target ±5 percentage points RH over a declared range | No claim until reference test |
| Pressure | Set a mission-specific limit before test; provisional screening target ±1.0 hPa over a declared range | No claim until reference test |
| Missing samples | <1% during a 72-hour bench run, excluding explicitly logged power-off intervals | Must report denominator and exclusions |
| Reproducibility | Five units tested with the same frozen protocol; publish unit-level distributions, not just pooled average | Screening gate, not fleet qualification |
| Repair recovery | 100% of intentionally introduced, safe-to-test faults detected; after repair, all required checks rerun | Any undetected critical fault fails |
| Correction robustness | Held-out error does not exceed the pre-registered limit; disclose any condition where correction worsens error | Hard gate |
| Repair/data lineage | 100% of repairs link removed/installed part, reason, operator/process record, and post-repair test | Hard gate |

The provisional ±1.0 °C / ±5 %RH / ±1.0 hPa values are starting points for a modest screening instrument only. They are not appropriate for every mission, nor do they imply that the candidate boards meet them. If the use case cannot justify these thresholds, replace them before collecting acceptance data.

## 6. South African calibration and fabrication leads (scope confirmation required)

These are public service descriptions, not endorsements, confirmed quotes, or evidence that a specific bare breakout board will be accepted.

- **Testo South Africa:** lists SANAS-accredited temperature and humidity calibration scopes and traceable pressure service; Johannesburg address in Kempton Park. Ask whether the exact assembled SPM-0 logger and its required points/ranges fit the applicable accredited scope. https://www.testo.com/en-ZA/services/calibrations
- **Applied Metrology Services:** describes temperature and pressure calibration, including ISO 17025 (SANAS) service, in Midrand. Ask for the exact sensor/logger form factor, range, points, uncertainty, turnaround, and certificate scope. https://www.appliedmetrology.co.za/
- **TriLab:** describes temperature, pressure, and humidity services, with Johannesburg contact details and a Pinetown address. Confirm the current service location and scope before shipping. https://trilab.co.za/temperature-calibration/
- **3D printing / enclosure prototype:** obtain a dated, delivered quote from a local service only after the mounting envelope, material, UV/temperature exposure, fastening, and ingress requirements are frozen. A printed enclosure is not automatically weatherproof or suitable for outdoor exposure.

No prices, availability, acceptance of the exact SPM-0 assembly, or calibration results have been verified. A calibration certificate for a reference instrument does not automatically confer traceability on SPM-0 readings; the actual measurement chain and uncertainty must be documented.

## 7. Quote request packet

Send vendors the same request so replies are comparable:
1. Exact instrument/module and assembled configuration (attach photographs, schematic, firmware hash, datasheet and mechanical drawing).
2. Measurands, ranges, target points, intended operating conditions, and required uncertainty / acceptance limits.
3. Requested service: feasibility review, calibration or comparison, certificate type, accreditation scope, as-found/as-left results, uncertainty, and adjustment policy.
4. Unit count and whether each unit needs an individual result.
5. Delivery address, collection/shipping costs, VAT, lead time, turnaround, quote validity, and exclusions.
6. Explicitly ask whether the requested work is within the provider's accredited scope; do not infer scope from a general accreditation claim.

Record each response as formal_quote, budgetary_quote, public_catalog_listing, or unpriced_dependency. A public service page is a sourcing lead, not a quote. Missing price/scope is unknown, never zero.

## 8. Reproducibility and fail-closed gates

- Freeze fixture, protocol, analysis code, and acceptance thresholds before qualification.
- Hash raw logs and preserve immutable source data.
- Include synthetic negative controls for swapped sensor identities, stale calibration, incorrect units, clock drift, missing reference points, correction leakage, enclosure revision mismatch, and repair without requalification.
- Independent verifier must recompute metrics from raw observations rather than trust summary fields.
- Unknown sensor identity, absent reference metadata, expired/invalid calibration evidence, or a failed required test must downgrade/abstain; never silently pass.
- Keep synthetic fixtures separate from measured evidence. A fixture replay verifies logic only; it does not prove sensor performance.
- Do not represent this document or supplier leads as evidence that any physical test has happened.

## 9. Next execution sequence

1. Freeze whether SPM-0-P, SPM-0-H, or both are required and state the actual measurands.
2. Send the same scope request to at least two providers; record replies and exclusions.
3. Obtain exact delivered BOM quotes and confirm component revisions before purchasing.
4. Assemble five units only after a minimum test fixture and reference method are available.
5. Run the pre-registered bench protocol, publish raw-data hashes and per-unit results, and let an independent verifier recompute the metrics.
6. Use the results to decide whether to recalibrate, redesign the enclosure, change supplier, or narrow the instrument's claim.

**Stop condition:** If a suitable reference method cannot be obtained within the project budget, continue only with clearly labelled comparative screening and do not claim traceable accuracy.
