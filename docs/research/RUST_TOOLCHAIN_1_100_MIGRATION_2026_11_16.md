# Rust 1.100.0 / Cargo toolchain migration

**Date:** 2026-11-16  
**Status:** staged toolchain migration; exact-head validation and Nix lock refresh pending

## Scope

- Move the workspace's canonical `rust-toolchain.toml` from Rust 1.96.0 to 1.100.0.
- Keep CI installs and explicit `cargo +VERSION` commands synchronized with the canonical version.
- Install the components/target already expected by the root Nix development shell: rustfmt, Clippy, rust-analyzer, rust-src, llvm-tools-preview, and `wasm32-unknown-unknown`.
- Add Cargo dependency cooldown and MSRV-aware resolution defaults.
- Add project-scoped dependency, mutation, feature-matrix and coverage tools to the Rust devShell.

Rust 1.100.0 was released on 2026-11-12. Cargo is shipped with the Rust toolchain; this migration intentionally does not install a separate, potentially mismatched Cargo binary.

## Why these Cargo settings

`.cargo/config.toml` sets a three-day crates.io publication cooldown for future dependency resolution and asks Cargo to prefer MSRV-compatible dependency versions where possible. These settings do not rewrite existing `Cargo.lock` entries. The publication-age check depends on registry publication-time metadata and is not a complete defense against malicious packages.

Cargo 1.100 also introduces native unused-dependency reporting and a more consistent build-directory layout intended to reduce collisions and unnecessary rebuilds, especially across feature sets and worktrees. The migration does not delete target directories or change the dependency lockfile as part of this PR.

## Tooling added to the Symthaea Nix devShell

- `cargo-deny`: advisory, license, source and dependency-policy checks (requires a reviewed `deny.toml` policy before enforcement).
- `cargo-audit`: independent RustSec advisory scan of the lockfile.
- `cargo-mutants`: mutation testing of safety/evidence validators and negative controls; use narrow package/module scopes.
- `cargo-hack`: repeatable feature-combination testing.
- `cargo-llvm-cov`: coverage reporting, backed by the pinned `llvm-tools-preview` component.
- Existing `cargo-nextest` and `cargo-machete` remain available.

`cargo-fuzz` is deliberately not placed in the stable default shell: it requires a separate nightly toolchain lane and should be added with a dated nightly pin and dedicated fuzz targets.

## MSRV policy

The workspace CI toolchain is a tested development baseline, not automatically every crate's public MSRV. Package-specific lower `rust-version` declarations and standalone subcrate toolchains (for example, the Wisdom component's separately pinned 1.94.1 lane) are not raised by this change. We should retain a separate MSRV audit rather than claiming all crates now require Rust 1.100.

## Nix lock refresh blocker

The current repository lockfiles pin old toolchain/package metadata:

- Root `flake.lock`: nixpkgs last updated 2026-05-15; rust-overlay last updated 2026-06-13.
- `crates/domains/symthaea-hal/flake.lock` and `crates/core/nixward/flake.lock`: rust-overlay last updated 2026-07-10.
- Android subflake locks under `symthaea-soma` and `symthaea-spore`: rust-overlay last updated 2026-03-17.

These pinned rust-overlay snapshots predate Rust 1.100. The Rust version bump must not be considered Nix-devShell-qualified until the relevant flake locks are refreshed with Nix, then each devShell is evaluated. Refresh one input at a time; do not use a broad `nix flake update` as part of this compiler migration. The workstation's NixOS branch already contains a November 2026 nixpkgs snapshot with rustup 1.29.3, but that does not update Symthaea's independent flake locks.

## Validation required before merge

1. Exact-head format and compile/test checks under Rust/Cargo 1.100.0.
2. Focused Clippy, nextest and feature-matrix workflows; inspect newly surfaced unused-dependency warnings rather than mass-suppressing them.
3. `cargo deny check`, `cargo audit`, and targeted `cargo mutants` runs; establish a reviewed deny policy separately.
4. Refresh the root and nested Rust-overlay flake locks with Nix and evaluate each corresponding devShell.
5. Record the exact subject commit, toolchain output, Cargo version, lockfile digest, workflow run IDs and test outcomes.

No `Cargo.lock` dependency refresh, host toolchain installation, Nix flake-lock mutation, build, or test pass is claimed by this documentation-only protocol.
