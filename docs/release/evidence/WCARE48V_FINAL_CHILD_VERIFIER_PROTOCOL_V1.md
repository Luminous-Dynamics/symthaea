# WCARE-48V — FINAL-child structure verifier v1

Status: `PREREGISTERED_VERIFIER`
Authority: `MeasurementOnly`
Tracks: #2982
Verifier lineage parent: repaired WCARE-47 exact head `0321a22986e5c6bf1a6b099d30dc75ccc33d9deb`

## Purpose

Verify the eventual WCARE-48 FINAL child against a contract that exists before that child does.

The generator receipt cannot contain the FINAL commit SHA without creating a circular commitment. WCARE-48V therefore verifies the FINAL commit externally while remaining on a sibling lineage that cannot alter the WCARE-48 PREPARED ancestry.

The governing distinction is:

`generation receipt internally consistent != FINAL child structurally valid != lock admitted`

## Exact PREPARED subject

The only admissible FINAL parent is:

`52d1d9fb741250ab8bcab205113689a8cc9431bb`

Its parent must remain repaired WCARE-47:

`0321a22986e5c6bf1a6b099d30dc75ccc33d9deb`

## FINAL-child theorem

A candidate FINAL commit is `FINAL_CHILD_VALID` only when all of the following hold:

1. the candidate exists as a commit and has exactly the frozen PREPARED commit as its parent;
2. the candidate changes exactly two paths, both additions:
   - `tools/wcare42_builder_attestation_verifier/Cargo.lock`;
   - `docs/release/evidence/WCARE48_LOCK_GENERATION_RECEIPT_V1.json`;
3. the receipt is duplicate-key-free JSON with exactly the frozen v1 field set;
4. receipt repository, branch, protocol, PREPARED head/parent/tree, workflow/script/protocol blobs, WCARE-42 source blobs and root toolchain blob match the frozen Git subject;
5. the committed lock blob equals the receipt candidate Git blob, while its SHA-256 equals both receipt lock hashes;
6. the committed lock is Cargo.lock v4, contains exactly one standalone root package, has the receipt package count, and every non-root package is registry-backed with a 64-hex checksum;
7. the receipt records locked metadata/test success, same-runner repeat-generation equality, unchanged source postflight, and repository-host-observed generation lineage;
8. every stronger provenance, executable-qualification and runtime-authority field remains false;
9. the FINAL commit message binds the exact PREPARED head and GitHub run id, with the run id equal to the receipt;
10. all frozen WCARE-42/source/toolchain/workflow/script/protocol blobs remain unchanged in the FINAL commit.

## Classifications

- `FINAL_CHILD_VALID` — structural and receipt consistency theorem passed.
- `FINAL_CHILD_INVALID` — candidate exists but violates the theorem.
- `TARGET_MISSING` — requested FINAL target does not resolve to a commit.
- `INVALID_VERIFIER` — verifier prerequisites or frozen PREPARED lineage are unavailable/drifted.

## Promotion boundary

`FINAL_CHILD_VALID` does not admit the lock. The exact FINAL child must still pass WCARE-47's lock-admission theorem, followed by the later execution-input closure and WCARE-42 executable qualification.

This verifier does not establish builder authentication, external preregistration, independent-host reproducibility, subject correctness, runtime authority, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, or solved alignment.
