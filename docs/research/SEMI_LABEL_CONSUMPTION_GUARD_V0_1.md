# SEMI-LABEL-CONSUMPTION-001 — calibration-qualified label consumption guard v0.1

**Related issue:** #13411  
**Implementation:** `symthaea-semiconductor::benchmark::label_consumption_guard`  
**Status:** Rust reference component with authored tests; exact-head validation pending.

## Purpose

Prevent a process-event metrology label from being consumed downstream merely because its label status says `validated`.

A usable label must be bound to:
- the exact measurement event;
- the exact instrument, measurand and unit;
- the exact calibration certificate reference;
- the measurement qualification report reproduced from the measurement, certificate and policy;
- the selected temporal qualification frontier (historical K0 or current K1).

The guard composes the existing process-event graph label, calibration qualification gate (#13382), and bitemporal calibration requalification adapter (#13412). It does not create a new event store, dependency graph, signature system or label-promotion service.

## Contract

Input:
- one `MetrologyLabelV1`;
- the frozen `CalibrationRequalificationInputV1`, containing the physical measurement, calibration certificate, policy, original qualification report, owner event inventory and K0/K1 materialization.

Output:
- label and measurement event identities;
- exact calibration certificate ID;
- selected frontier (`AsKnownAtK0` or `CurrentAtK1`);
- `EligibleUnderDeclaredPolicy`, `Unknown` or `Blocked`;
- whether the evidence contract permits downstream consumption at that frontier;
- input/output evidence digests and explicit reasons;
- explicit false values for source-authenticity proof, training-snapshot mutation and equipment-control authority.

The function independently reruns the measurement qualification and bitemporal calibration requalification from their inputs. It rejects a forged or stale original qualification receipt.

## Fail-closed mapping

- A validated label with matching event, instrument, measurand, unit, certificate and qualified calibration may be eligible under the declared policy.
- Provisional, censored or unknown labels remain UNKNOWN; this reference guard does not silently turn a censored interval into a point target.
- Invalid labels are BLOCKED.
- Missing instrument or certificate references are UNKNOWN.
- Conflicting measurement event, instrument, measurand, unit or certificate identities are BLOCKED.
- An unqualified original measurement is not eligible.
- A qualified retroactive revocation observed after K0 preserves the K0 assessment but blocks current K1 consumption.
- A relevant event that exists in the owner inventory but is not materialized to the candidate K1 context makes current eligibility UNKNOWN, not eligible.
- A changed input, report digest or policy requires a new receipt.

## Non-authority boundary

The guard returns an eligibility report only. It does not:
- edit, delete, revise or promote a process-event label;
- change a historical training manifest or model snapshot;
- trigger retraining;
- assert that a certificate issuer is authentic;
- issue a calibration or conformity certificate;
- authorize wafer disposition, sampling, maintenance, recipe changes, APC/R2R, or equipment commands.

`EligibleUnderDeclaredPolicy` means the supplied declared evidence contract passed. It is not independent physical traceability or source authenticity.

## Authored tests

The test module covers:
1. validated label at K0 and K1;
2. provisional and censored labels;
3. missing instrument/calibration reference;
4. measurement/certificate mismatch;
5. instrument, measurand and unit mismatch;
6. invalid versus unknown label states;
7. retroactive revocation preserving K0 and blocking K1;
8. relevant but unmaterialized revocation;
9. tampered original qualification receipt;
10. deterministic eligibility receipt replay.

Tests are authored and must not be described as passing until the exact branch head completes rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 independent reference workflow.

## Next integration step

This component is the bounded consumer-side gate. A separate integration increment must apply it when building training/evaluation snapshots, record exactly which eligible labels were consumed, and requalify or quarantine dependent model snapshots when a later K1 event changes the eligibility of their input labels. That downstream dependency closure remains tracked by #13411.
