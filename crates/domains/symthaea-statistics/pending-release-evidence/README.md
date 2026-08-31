# Pending statistics release evidence

This directory preserves verification sources that were synced into the repository before their corresponding statistics implementation was integrated into the active Cargo package.

## Active package boundary

`crates/domains/symthaea-statistics/Cargo.toml` currently declares `symthaea-statistics` version `0.1.0`, and the active `src/lib.rs` exports the foundational descriptive, distribution, inference, regression, Bayesian-diagnostic, and special-function surface for that package.

On 2026-07-24, monorepo sync commit `997746177b0af2672f55510fd1b773efcbc3bb18` imported later-release migration/patch artifacts and versioned integration tests into the live crate directory without integrating the corresponding later-release source/API into the active package. Because Cargo automatically compiles every Rust file under `tests/`, those future-release tests made an otherwise unrelated `cargo test -p symthaea-statistics` fail on unresolved imports.

## Preserved files

The files under `tests/` are moved here byte-for-byte from the former active Cargo integration-test directory:

- `v0_4_models.rs`
- `v0_5_adversarial.rs`
- `v0_6_adversarial.rs`
- `v0_6_reference_vectors.rs`
- `v0_7_adversarial.rs`
- `v0_7_reference_vectors.rs`
- `v0_8_adversarial.rs`
- `v0_8_reference_vectors.rs`

They are **not evidence that the active 0.1.0 crate implements those later APIs**. They are pending release/integration evidence retained for the future statistics-release restoration work.

## Promotion rule

Do not copy these tests back into Cargo's active `tests/` directory merely to claim a later version.

For each release tranche, first integrate the exact corresponding implementation/API, migration contract, and package-version transition. Then move only that tranche's relevant verification files back into the active test surface and require them to pass under an exact-head qualification workflow.

A promotion should preserve provenance to the original synced test blobs and should not weaken or rewrite reference tolerances just to obtain a green result.

## Qualification meaning

After this separation, a green `cargo test -p symthaea-statistics` means the **currently declared active package** is coherent. It does not qualify the pending v0.4-v0.8 release artifacts stored here.

Conversely, failures in these pending files must not be hidden or deleted; they remain explicit evidence debt until the corresponding implementation lineage is restored and qualified.
