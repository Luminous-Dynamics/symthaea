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

Use `dose-events-template.csv` for **one row per physical component-release event**. Record the exact formulation/component batch, component calibration, cycle phase and phase window, release start and full duration, water temperature at release, commanded mass and measured actual mass in integer milligrams, delivery status, dispenser pressure/flow, interaction evidence and raw-event digest.

Keep cycle rows and dose-event rows linked by `run_id`. Do not infer that a commanded dose was delivered. Actual mass should come from a calibrated gravimetric measurement or calibrated volume plus batch-specific density. A missing value is empty, not zero. Do not aggregate several component events into one row. Cycle-level aggregate mass remains in grams; the per-event archive uses integer milligrams.

The Rust `symthaea_dishwasher_design::chemistry::validate_dose_plan` validates a proposed schedule against a declared `FormulationCalibrationProfile`: component identity, soil/surface scope, phase ordering, stage-specific temperature, water hardness, integer-milligram per-event/per-component-cycle/total mass limits using checked arithmetic, and evidence-backed separation between complete dispense intervals. It does not measure delivery, select a formulation or dose, prove evidence authenticity, certify ingredient compatibility, or establish cleaning. The compatibility and calibration hashes are provenance identifiers and must be independently audited against the formulation owner's and laboratory's original records.
