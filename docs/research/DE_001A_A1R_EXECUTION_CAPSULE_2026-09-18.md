# DE-001A1R — exact-head execution capsule

**Date:** 2026-09-18  
**Authority:** fixed-point-reproduction-qualification-only  
**Scientific claim:** NONE

This tranche closes the gap between a correct research binary and evidence that a specific immutable binary actually produced a specific receipt chain.

The qualification workflow checks out the exact pull-request head and records the Git HEAD and TREE before any build or research execution. It then uses Rust 1.96.0 from a commit-pinned `dtolnay/rust-toolchain` action and a commit-pinned Nix installer.

The Rust subject must pass:

1. rustfmt;
2. locked all-target compilation;
3. locked all-target tests;
4. strict all-target Clippy (`-D warnings`);
5. locked release build of every DE-001A stage binary.

The workflow hashes the exact release binaries used for A0, A1P, A1R, and A1N.

## Content-addressed inputs

The workflow realizes `nix#a0-artifacts`, whose eight files are fixed-output derivations. It also builds the independent Nix A0 artifact check, then records the realized store path, NAR hash, and closure size.

## Ordered execution

The reusable `scripts/qualify_a1r.sh` runner executes:

```text
A0 byte integrity
  -> A1P fixed-point provenance
  -> A1R primary (1024 subdivisions)
  -> A1R refined (2048 subdivisions)
  -> A1N convergence
```

A0 or A1P failure is `INVALID` and stops the chain. An A1R exit code of 1 is retained as a valid `NEGATIVE` reproduction result so A1N can still establish numerical convergence. An A1R invalid result stops the chain. A1N non-convergence is `INVALID`, not a cosmological negative.

## Capsule

The retained `qualification-capsule.json` binds:

- exact Git HEAD and TREE;
- base revision;
- Rust version;
- pinned GitHub Action commits;
- pinned nixpkgs revision;
- `Cargo.lock` SHA-256;
- SHA-256 of all four release binaries;
- A0 Nix store path, NAR hash, and closure size;
- SHA-256 of A0, A1P, primary A1R, refined A1R, and A1N receipts;
- workflow run/attempt and each stage outcome;
- postflight immutability outcome.

Evidence is uploaded even when the final qualification verdict fails, so a valid `NEGATIVE` reproduction or infrastructure defect remains inspectable rather than disappearing behind a red CI badge.

A workflow PASS licenses only the statement that the frozen fixed-point reproduction chain executed successfully under the bound software/input identities. It does not establish LambdaCDM, constant dark energy, an anomaly, or a physical mechanism.
