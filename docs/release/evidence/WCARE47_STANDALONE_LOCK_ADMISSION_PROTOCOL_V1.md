# WCARE-47 — Standalone Cargo.lock admission protocol v1

Status: `PREREGISTERED_ADMISSION`
Authority: `MeasurementOnly`
Protocol version: `wcare47-standalone-lock-admission-v1`
Tracks: #2914

## Purpose

WCARE-46 integrates the exact WCARE-42 implementation tree but correctly leaves execution blocked because the standalone verifier has no committed `Cargo.lock`.

WCARE-47 defines how a future lock may be admitted as evidence. The governing distinction is:

`lock exists != lock admitted != WCARE-42 executable-qualified`

## Exact source subject

Admission is scoped to the exact integrated WCARE-42 source:

- `tools/wcare42_builder_attestation_verifier/Cargo.toml` blob `5410040e5616241dd4ba581af8f297675d083830`;
- verifier source blob `1c300a455f054d118e55556aac81b623824629bc`;
- golden-test blob `666242f74fb302f9be2b62fa4b3050f3ee9ffefd`;
- root `rust-toolchain.toml` blob `4f0430eac96d545bcfaa0df23ce475faf4aee96a` with channel `1.96.0`.

A source change requires a new admission subject; an old lock receipt cannot follow changed verifier code.

## Admission evidence

A candidate lock may be classified `LOCK_ADMITTED` only if one exact run establishes all of the following:

1. the exact source/toolchain blobs above are present before execution;
2. evaluated HEAD is a descendant of WCARE-46 exact head `8a3cacb449b923ceee32b6b08e2c811ce532c676`;
3. working tree is clean for qualification;
4. standalone `Cargo.lock` exists as a committed file;
5. its SHA-256 and Git blob identity are recorded;
6. lock format/version and package census are parsed successfully;
7. every non-root package is registry-backed and checksum-bearing in v1; Git/path dependencies are rejected;
8. active `rustc` is exactly 1.96.0 and Cargo identity is recorded;
9. `cargo metadata --locked --manifest-path tools/wcare42_builder_attestation_verifier/Cargo.toml --format-version 1` succeeds;
10. `cargo test --locked --manifest-path tools/wcare42_builder_attestation_verifier/Cargo.toml` succeeds, including the independent Ed25519 golden-vector tests;
11. exact WCARE-42 source and root toolchain blobs remain unchanged after execution.

## Generation provenance boundary

Compatibility admission does not prove how the candidate lock was historically generated. Unless a separate PREPARED→generation→FINAL receipt exists, the result must keep:

`lock_generation_provenance_established = false`.

A lock can therefore be admitted as an exact dependency-resolution artifact usable for qualification without claiming provenance that was never observed.

## Classifications

- `LOCK_ADMITTED` — exact candidate lock passed the admission ceremony.
- `LOCK_REJECTED` — the candidate lock or subject failed a substantive admission invariant.
- `INFRASTRUCTURE_INDETERMINATE` — required tool/runtime/infrastructure prevented a valid conclusion.
- `INVALID_PROTOCOL` — the qualification contract or source binding is malformed/drifted.

## Initial state

At the initial WCARE-47 freeze the standalone lock is absent. The only valid classification is therefore:

`INFRASTRUCTURE_INDETERMINATE: candidate_lock_missing`

No admission or executable WCARE-42 claim is made.

## Promotion boundary

Even `LOCK_ADMITTED` establishes only that the exact dependency graph is pinned and executable tests succeeded under the admission environment. WCARE-42 must still run its own exact verifier qualification afterward.

## Non-claims

WCARE-47 grants no runtime authority and does not establish builder authentication, external temporal preregistration, subject correctness, reviewer independence, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, or solved alignment.