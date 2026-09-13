# ADR-038: Bind lockfile-repair runner provenance

**Status**: Proposed
**Change Class**: A
**Scope**: Replicator Safety Kernel (RSK) Cargo.lock repair qualification

## Context

ADR-037 pins third-party GitHub Action implementations to immutable commit SHAs. The GitHub-hosted runner remains a separate mutable execution dependency: `ubuntu-latest` can change operating-system generation and every hosted image label can receive image revisions over time.

For the non-admissible #1926 repair diagnostic, executor drift should be constrained and recorded rather than silently treated as equivalent.

## Decision

- replace `ubuntu-latest` with the explicit hosted-runner generation `ubuntu-24.04`;
- retain GitHub-provided image identity variables when available (`ImageOS`, `ImageVersion`);
- record `RUNNER_OS`, `RUNNER_ARCH`, `uname`, `/etc/os-release`, Git version, and Python version in the evidence artifact;
- continue binding exact Rust/Cargo identity and immutable action commits separately.

A future change to runner generation is a reviewed Class A workflow change. Hosted-image revision remains observable evidence, not a trust grant.

## Non-claims

This does not make GitHub-hosted infrastructure reproducible, confidential, tamper-proof, or suitable as a production trust root. The workflow remains NON-ADMISSIBLE diagnostic evidence and cannot establish compilation qualification, runtime admission, physical safety, or replication authority.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
