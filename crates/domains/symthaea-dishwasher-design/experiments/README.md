# Physical experiment data contract

Record one row per independently repeated run. Do not collapse repeats into an average before archiving raw data.

## Mandatory provenance

- `run_id`: globally unique experiment identifier.
- `replicate_id`: independent replicate number, not repeated image frames from one run.
- `protocol_version`: immutable protocol revision.
- `utc_timestamp`: ISO 8601 UTC timestamp.
- `geometry_sha256`: digest of the CAD/mesh/configuration used.
- `calibration_ids`: identifiers and versions for every instrument calibration.

## Units

All units are encoded in column names. Flow is litres/minute, pressure is pascals, power is measured real electrical watts, mass is grams, concentration/hardness are milligrams/litre, and duration is seconds.

## Experimental discipline

1. Measure pump real power with an instrument capable of measuring non-sinusoidal motor/drive loads; voltage multiplied by current is not a substitute for real power.
2. Measure flow and pressure at the branch being compared, not only at the pump outlet.
3. Keep soil recipe, applied mass, drying/aging time, detergent batch, water hardness, inlet temperature and surface material fixed within a matched comparison.
4. Record delivered fresh water separately from recirculated sump volume.
5. Obtain coverage fraction from a separately specified and calibrated imaging or witness-surface method. Do not label geometric visibility as measured soil removal.
6. Use independent replicates and retain every raw measurement and image.
7. Include blocked-nozzle, leak, disconnected-sensor, unit-mismatch and mass-balance negative controls.
8. Mark missing values as empty, not zero. Zero is a valid measurement and must remain distinguishable from missing data.

This schema does not itself implement IEC 60436 or a sanitization test.

## Occupancy-curve constructor mapping

The run-level Rust API in `symthaea-dishwasher-scheduler::resource_curve` consumes fields from a frozen experiment archive:

- `PhysicalRunMeasurement.run_id` ← `run_id`
- `cleaning_pass` / `rinse_pass` ← the corresponding independent ground-truth acceptance calls
- `energy_kwh` ← `total_wall_real_energy_kwh` (include controls, pump, heater, drying and declared standby boundary)
- `fresh_water_l` ← `fresh_water_l`
- `duration_s` ← `cycle_duration_s`
- `cleaning_evidence_ref` / `rinse_evidence_ref` ← immutable assay / coupon / residue record references
- `calibration_bundle_sha256` ← digest of the calibration-certificate and meter-configuration bundle

The curve scope must be built from the exact machine/programme, soil/surface, load-set, soil/detergent/water batch, soil age, water hardness, inlet temperature, ambient temperature and humidity, protocol ID and raw-data archive digest. Do not combine runs with different fixed conditions into one curve.

The CSV header is a data contract only; it does not authenticate the evidence references or automatically compute the pass/fail fields. Preserve original meter exports and assay records. A run failing a cleaning/rinse endpoint stays in the raw archive and the failed occupancy must not be used as an eligible scheduler curve.

## Phase-resolved detergent events

The cycle-level `measurement-template.csv` records formulation identity/version, formulation-evidence digest, dose-plan identity/digest, commanded and actual cycle-total component mass, and the digest of a separate dose-event archive.

Use `dose-events-template.csv` for **one row per physical component-release event**. Record the exact formulation/component batch, component calibration, cycle phase and phase window, commanded and actual release start/duration, water temperature at release, commanded and measured actual mass in integer milligrams, delivery status, dispenser-specific calibration/tolerance, sensor evidence digest, pressure/flow and raw-event digest.

Keep cycle rows and dose-event rows linked by `run_id`. Do not infer that a commanded dose was delivered. Actual mass should come from a calibrated gravimetric measurement or calibrated volume plus batch-specific density. A missing value is empty, not zero. Do not aggregate several component events into one row. Cycle-level aggregate mass remains in grams; the per-event archive uses integer milligrams.

The Rust `symthaea_dishwasher_design::chemistry::validate_dose_plan` validates a proposed schedule against a caller-supplied `FormulationCalibrationProfile`: component identity, soil/surface scope, phase ordering, release start and full duration, stage-specific temperature, water hardness, integer-milligram per-event/per-component-cycle/total mass limits using checked arithmetic, and evidence-backed separation between complete dispense intervals. The separate `check_reported_dose_delivery` compares reported actual milligrams and timing with a caller-supplied dispenser tolerance using integer basis-point arithmetic. It binds the report to the commanded event ID, exact component ID, physical dispenser channel and delivery-calibration ID; it also receives the exact stage window and rejects a measured release interval that crosses its phase boundary. The event CSV captures commanded versus actual start/duration and those calibration/tolerance fields. It does not authenticate the underlying telemetry, prove evidence truth, select a formulation or dose, certify ingredient compatibility, or establish cleaning success. Calibration and measurement hashes are provenance identifiers and must be independently audited against the formulation owner's and laboratory's original records.

## Independent per-item outcome evidence

The outcome qualification constructor also requires balanced physical run coverage: every expected item and every required endpoint must be represented in the same set of distinct run IDs. A result for one item cannot be substituted for a missing result on another item, and item observations from different cycles cannot be mixed to create a false cross-contamination pass. Each reading is bound to the exact criterion ID, metric ID, unit, exact assay method ID/version and protocol; the method ID in the reading must match the preregistered method ID in the criterion. The pilot envelope retains the exact endpoint criteria and assay-method identities so the scheduler adapter can enforce criterion/method consistency across occupancies for each soil class.


Use `outcome-evidence-template.csv` for one item × endpoint × physical run measurement. Every observation binds to:
- exact run and item ID;
- endpoint, criterion ID and metric ID;
- independent assay method and protocol;
- measured value, expanded uncertainty and unit;
- evidence reference and calibration bundle digest.

Use `endpoint-criteria-template.csv` for preregistered threshold, unit and comparison direction. Freeze the criteria before collecting the test series. Thresholds must be sourced from the licensed method/standard or a documented independent research protocol; this software only checks the recorded criterion identity/digest shape and does not certify that the threshold is scientifically or legally appropriate.

The Rust `quality::qualify_pilot_evidence` gate computes pass/fail/indeterminate using expanded uncertainty. Soil residual, final-rinse residue and cross-contamination are mandatory. Drying and relative microbiology are separate opt-in endpoints. Calibrated process proxies (turbidity, conductivity, camera confidence, pump current, temperature and commanded detergent mass) cannot substitute for independent outcome assays in this pilot qualification path. Five independent runs is a pilot floor only, not a confirmatory sample size. See `docs/outcome-evidence-and-stop-rule.md` and issue #10225.
