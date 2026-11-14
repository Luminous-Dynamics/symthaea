# EDLC Trace Evidence Receipt v0.1

**As of:** 2026-11-14  
**Status:** synthetic multi-artifact receipt and Rust integrity auditor authored; Rust execution and physical-laboratory validation pending.

## 1. Paradigm

The benchmark object should be more than a summary PDF or unreferenced CSV. Every derived result should resolve to:
- immutable raw waveform bytes;
- exact load and readiness-probe profile bytes;
- physical device and lot identity;
- test protocol and application boundary;
- instrument configuration and calibration interval;
- environment and sample-clock metadata;
- analysis version and deviations.

The receipt is a manifest that binds those objects with file paths, byte lengths and SHA-256 digests. It is intended to make missing provenance visible before any physical performance claim is admitted.

## 2. Integrity is not authenticity

A matching SHA-256 digest proves only that the bytes currently inspected match the digest written in the receipt, assuming the receipt itself is trusted. It does **not** prove:
- who created the file;
- whether the named instrument was used;
- whether the calibration certificate is genuine;
- whether the serial number belongs to the physical device;
- whether the operator followed the protocol;
- whether the result is independently replicated.

The first implementation explicitly reports that limitation. Signed lab attestations, trusted identity/key management and independent calibration-provider validation are later gates; no unsigned JSON field should be described as authenticated origin.

## 3. Required evidence package

The first schema requires:
- raw terminal trace with event ID, elapsed time, terminal voltage/current, source voltage/current and temperature;
- per-event load waveform;
- a readiness-probe profile for every event after the first call;
- device identity;
- instrument configuration;
- calibration certificate/record with matching instrument serial and a validity interval covering the full acquisition;
- test protocol with exact event order, sequence profile and standby mode;
- environment record.

Each artifact is identified by unique ID, safe relative path, role, media type, byte length and lowercase SHA-256. Paths must remain inside the workspace root after canonicalization; path traversal and symlinks resolving outside the root are rejected.

The auditor checks raw trace column presence, event identity, row count, monotonic elapsed time within each event and source-side current. An open-circuit receipt may not include non-zero source current. Load-profile and readiness-probe CSVs must have strictly increasing elapsed time and at least two rows.

Bidirectional current is permitted in profiles: negative load current can represent regeneration or charge flow and must not be clipped merely because a discharge-only fixture is common.

## 4. Synthetic known-answer package

The checked-in fixture has nine files:
- one raw terminal trace covering two synthetic calls;
- two load profiles;
- one readiness-probe profile for the second call;
- device identity, instrument configuration, calibration record, protocol and environment JSON.

The Rust command is:

```sh
cargo run -p xtask -- edlc-trace-evidence-audit
cargo test -p xtask edlc_trace_evidence
```

There are 19 authored unit tests covering known-answer integrity, digest mismatch, byte length, duplicate artifact ID/path, required roles, path traversal, timebase consistency, calibration validity, device identity, event order, first/later readiness-profile rules, event ordering, CSV row count and profile role linkage.

These tests are authored, not executed in this environment. The fixture is synthetic and its device/instrument/certificate values are deliberately fictional. It proves receipt arithmetic and integrity rules only.

## 5. External published-data admission

Published raw-data files use a separate lane. Their receipt must carry:
- DOI, repository record ID, version and file-specific persistent ID;
- licence and reuse conditions;
- source citation;
- repository MD5 (if supplied) and locally computed SHA-256;
- exact file size, headers, units, sample identity and protocol metadata;
- parser/normalization version and parent-file digest for derived traces.

The first candidates and their rights/evidence ceilings are in [EDLC_EXTERNAL_RAW_DATA_REPRODUCTION_V0_1.md](EDLC_EXTERNAL_RAW_DATA_REPRODUCTION_V0_1.md). A public electrode or flexible-cell dataset cannot be promoted to complete-module reserve evidence.

## 6. Physical laboratory admission gate

Before a receipt is classified as a physical lab record, obtain:
- actual manufacturer, model, chemistry/build, lot and unique physical serial;
- independent laboratory identity, operator and campaign ID;
- instrument manufacturer/model/serial, channel map, sample rate, timebase and uncertainty;
- calibration certificate ID, issuing provider, validity window and the actual certificate bytes;
- synchronized terminal/source V/I/time channels and temperature channels;
- exact load/probe waveform files, protocol revision, device boundary and charge/standby history;
- complete-device mass/volume and all installed auxiliary loads;
- deviations, aborts, missingness, censoring and every failed device;
- reviewable chain-of-custody and signed lab attestation when the trusted signing system is established.

If any item is unavailable, represent it as unknown or missing with a reason. Do not silently synthesize values or pass a partial package as physical qualification.

## 7. Evidence and benchmark gates

| Gate | Evidence | Permitted claim |
|---|---|---|
| R0 | Synthetic bundle, deterministic hash/path/schema checks | Software receipt integrity only |
| R1 | Version-pinned public dataset, licence, file-level digests, raw header/unit checks | Published-data ingestion/reproduction |
| R2 | Complete physical receipt, calibrated source/terminal channels, unique device/lot identity | Physical measurement under the stated protocol |
| R3 | Repeated-call trace replay and preregistered device-level statistics | Sequence performance in the measured domain |
| R4 | Independent lab replication, signed chain-of-custody and cross-lab uncertainty | Reproducibility within the tested domain |

A higher gate requires all lower evidence contracts plus its own additional evidence. The synthetic fixture does not grant R1–R4 admission.

## 8. Current implementation and limitations

Added:
- `schemas/energy/edlc_trace_evidence_receipt_v0_1.schema.json`;
- `fixtures/energy/electrochemical-capacitor/trace_receipt_v0_1/` synthetic artifact bundle;
- `fixtures/energy/electrochemical-capacitor/trace_evidence_receipt_known_answer_v0_1.json`;
- `xtask/src/edlc_trace_evidence.rs`;
- `xtask/src/main.rs` command `edlc-trace-evidence-audit`.

The implementation verifies artifact byte lengths and SHA-256 values, path confinement, cross-links between manifest and device/instrument/calibration/protocol/environment records, trace event identity, CSV structure and open-circuit source-current semantics. It is not a full generic JSON Schema evaluator, signature verifier, calibration-body cryptographic trust system, electrochemical result calculator, or physical reliability model.

No external raw dataset has been downloaded or ingested into the branch, no lab package has been received, and no physical result is claimed.
