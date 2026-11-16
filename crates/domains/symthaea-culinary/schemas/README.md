# Thermal Observation Schema V1

Status: proposed, schema-only, not deployed. Parent issue: https://github.com/Luminous-Dynamics/symthaea/issues/12537.

- `ThermalObservationV1.schema.json` defines the raw observation envelope and evidence-class-specific structural requirements.
- `fixtures/adversarial-v1.json` contains synthetic-only fixtures. A simulated trajectory is explicitly marked simulated and must never be promoted to measured.

The schema does not validate monotonic sample time across the array, clock rollover, inter-sensor skew, sample gaps, calibration range/expiry, probe placement truth, thermal lag correction, physical plausibility, model residuals or microbiological safety. Those are typed runtime-verifier responsibilities.

A calibrated-measured evidence class requires an instrument reference and calibration bundle reference. This is a structural prerequisite only: a reference string is not proof of calibration validity. The verifier must resolve the certificate, instrument, range, date, traceability, placement, uncertainty and raw record.

Keep raw data unchanged. Any interpolation, filtering, unit conversion or downsampling must produce a transformation record with source and output digests. Missing intervals remain explicit. No sample may be silently sorted, imputed, clamped or discarded.

The fixture that is syntactically valid but has decreasing elapsed time must be rejected by the stateful verifier. JSON Schema cannot enforce array ordering or per-probe continuity.
