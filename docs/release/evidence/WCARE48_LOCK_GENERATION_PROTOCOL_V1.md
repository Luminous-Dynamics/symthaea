# WCARE-48 — observed standalone Cargo.lock generation protocol v1

Status: `EXECUTABLE_GENERATION_PREREG`
Authority: `MeasurementOnly`
Tracks: #2942
Parent: repaired WCARE-47 exact head `0321a22986e5c6bf1a6b099d30dc75ccc33d9deb`

## Purpose

Produce the first `tools/wcare42_builder_attestation_verifier/Cargo.lock` under an observed repository-hosted PREPARED → GENERATION → FINAL lineage, then submit the generated child commit to WCARE-47 unchanged for lock admission.

The governing distinction is:

`generation observed != lock admitted != WCARE-42 executable-qualified`

## PREPARED subject

The PREPARED commit must:
- be a direct child of repaired WCARE-47 head `0321a22986e5c6bf1a6b099d30dc75ccc33d9deb`;
- contain this protocol, the frozen generation script, and the branch-scoped workflow;
- contain no standalone WCARE-42 `Cargo.lock`;
- preserve the exact WCARE-42 manifest/source/golden-test blobs and root Rust toolchain blob already bound by WCARE-47.

The workflow records the PREPARED commit, tree, workflow blob, script blob, protocol blob, GitHub run identity, runner identity, and exact Rust/Cargo identities before generation.

## Generation

The generator must:
1. refuse an existing committed or working-tree standalone lock;
2. require a clean PREPARED checkout;
3. verify Rust 1.96.0 and the exact WCARE-42/source/toolchain Git blobs;
4. run `cargo generate-lockfile --manifest-path tools/wcare42_builder_attestation_verifier/Cargo.toml`;
5. require Cargo.lock format 4, the standalone root package, and registry/checksum-backed non-root packages;
6. run `cargo metadata --locked` and `cargo test --locked` before any evidence commit;
7. regenerate once from the same PREPARED Git subject in a fresh detached worktree and require byte-identical lock output;
8. re-check all exact source/toolchain blobs after execution.

The repeated generation is only a same-runner repeatability check. It does not establish independent-host or cross-time reproducibility.

## FINAL evidence commit

After generation succeeds, the workflow may commit exactly two new files:
- `tools/wcare42_builder_attestation_verifier/Cargo.lock`;
- `docs/release/evidence/WCARE48_LOCK_GENERATION_RECEIPT_V1.json`.

That child commit is the FINAL event. Its parent must be the PREPARED commit. The commit message must bind the PREPARED head and GitHub run id.

No source, protocol, workflow, or verifier byte may be modified in the FINAL evidence commit.

## Promotion boundary

The generated child commit is not admitted merely because it exists. It must independently pass the frozen WCARE-47 admission verifier on the committed tree. WCARE-47 remains authoritative for `LOCK_ADMITTED`.

WCARE-48 may establish only `github_host_observed_generation_lineage = true`.

It must keep the following false unless separately proven:
- `external_generation_provenance_established`;
- `builder_authentication_established`;
- `external_preregistration_established`;
- `independent_host_reproducibility_established`;
- `wcare42_executable_qualification_established`;
- `runtime_authority_granted`.

## Non-claims

This protocol does not establish subject correctness, reviewer independence, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, runtime authority, or solved alignment.
