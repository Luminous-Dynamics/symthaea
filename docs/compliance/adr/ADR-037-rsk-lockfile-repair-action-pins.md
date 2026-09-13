# ADR-037: Pin lockfile-repair GitHub Action implementations

**Status**: Proposed
**Change Class**: A
**Scope**: Replicator Safety Kernel (RSK) Cargo.lock repair qualification

## Context

The RSK lockfile-repair diagnostic is intended to produce reviewable, reproducible evidence for Class A blocker #1926. Its Rust toolchain version is pinned, but GitHub Actions referenced by movable tags or branches can change implementation while the workflow commit remains unchanged.

For a supply-chain-sensitive qualification path, workflow source identity should bind the action implementations that perform checkout, toolchain installation, and artifact publication.

## Decision

Pin every third-party action used by `.github/workflows/rsk-lockfile-repair.yml` to an exact upstream commit SHA while retaining the human-readable release/ref in an adjacent comment:

- `actions/checkout`: `11d5960a326750d5838078e36cf38b85af677262` (`v4` at review time)
- `dtolnay/rust-toolchain`: `ebb3d1676050bfd0971c36c1e215b5751473994d` (`1.96.0` branch at review time)
- `actions/upload-artifact`: `ea165f8d65b6e75b540449e92b4886f43607fa02` (`v4` at review time)

Changing any pinned action implementation requires another reviewed workflow change; upstream tag/branch movement cannot silently change the executor code for an unchanged RSK workflow head.

## Non-claims

Commit pinning does not make GitHub-hosted runners trusted hardware, does not prove runner-image reproducibility, and does not make this diagnostic admissible qualification evidence. The generated lock candidate remains non-admissible until separately reviewed, committed, and qualified under the normal `--locked` RSK path.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
