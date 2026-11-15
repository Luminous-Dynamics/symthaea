# Comfort Trajectory Bench v0.1 — machine-readable contract

This directory contains the v0.1 session schema and synthetic fixtures for the protocol in [the parent research brief](../comfort-trajectory-bench-v0.1.md).

## Files

- `session.schema.json` — JSON Schema Draft 2020-12 contract.
- `fixtures/session-ao01-complete.json` — a complete 120-minute synthetic AO-01 session.
- `fixtures/session-cr01-stop.json` — a synthetic CR-01 session ended by the participant at 34 minutes; later scheduled observations are explicitly marked `not_collected_due_to_stop`.
- `fixtures/session-explicit-missing.json` — a scheduled observation with a dimension explicitly marked missing rather than dropped.
- `fixtures/invalid-observation-order.json` — negative fixture with the 30- and 60-minute observations transposed.
- `fixtures/invalid-rating-range.json` — negative fixture with a rating outside the declared 1–7 range.
- `fixtures/invalid-no-consent.json` — negative fixture where consent is not confirmed.

## Contract guarantees

The schema fixes the protocol and record versions, requires participant pseudonyms and affirmative consent, binds each session to an exact garment/pattern/BOM/assembly/configuration identity and material lot, limits comfort ratings to integers 1–7, distinguishes observed values from missing / not-applicable / unable-to-judge / not-collected-after-stop states, fixes the planned observation schedule at 0, 15, 30, 60, 90 and 120 minutes, and requires calibration plus a non-interference assessment when microclimate sensors are used.

All fixtures are synthetic. Do not place real names, contact information, body measurements, photographs or private preferences in this public corpus.

## Semantic checks required beyond JSON Schema

A JSON Schema validator checks record shape and local constraints. The benchmark implementation must additionally reject:

1. duplicate event IDs;
2. events outside the actual session interval or out-of-order event timestamps;
3. early-stop records whose later scheduled observations are not all marked `not_collected_due_to_stop`;
4. completed records with an actual duration other than 120 minutes;
5. non-completed records with no corresponding stop/withdrawal/adverse-event/protocol event;
6. participant-stop, withdrawal or adverse-event events inconsistent with the session termination reason;
7. a scheduled observation marked as not-collected-due-to-stop when it occurred before or at session end;
8. inconsistent style, variant, configuration, pattern, BOM, assembly-plan or material-lot identity across a session;
9. a comparator arm without a declared comparator identity;
10. sensor data without a named profile, calibration digest and non-interference assessment;
11. raw participant identifiers or direct personal data in public records.

Unknown or stale identity must fail closed. Do not infer a garment configuration from a filename or a human-readable style name. The exact configuration digest and material-lot identity govern evidence reuse.

## Validation status

The schema and seven fixtures were read back from the exact development branch. A deterministic read-back harness accepted the three positive fixtures and rejected the three negative fixtures for the expected conditions. This is a structural fixture check, not yet a committed Rust validator, a full JSON Schema meta-schema certification, CI evidence, human-subject approval, or physical qualification.
