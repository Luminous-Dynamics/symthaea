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

The cycle-level `measurement-template.csv` records formulation identity, commanded/actual cycle-total detergent mass, dose-plan identity and the digest of a separate dose-event archive.

Use `dose-events-template.csv` for **one row per physical component-release event**. Record the exact formulation/component batch, calibration, cycle phase, release time, phase temperature, commanded mass, measured actual mass (blank when not measured), delivery status, dispenser pressure/flow, interaction evidence and raw-event digest.

Keep cycle rows and dose-event rows linked by `run_id`. Do not infer that a commanded dose was delivered. Actual mass should come from a calibrated mass/volume measurement with a batch-specific density or a directly measured mass change. A missing value is empty, not zero.

The current Rust `chemistry::validate_dose_plan` validates a proposed schedule against a declared calibration profile; it does not measure delivery, choose a formulation, prove ingredient compatibility or establish cleaning. The compatibility and calibration hashes are provenance identifiers and must be independently audited against the original formulation-owner and laboratory records.
