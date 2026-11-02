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
