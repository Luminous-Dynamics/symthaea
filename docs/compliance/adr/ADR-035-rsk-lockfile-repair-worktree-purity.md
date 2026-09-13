# ADR-035: RSK Lockfile Repair Worktree Purity

**Status**: Proposed  
**Date**: 2026-09-13  
**Change Class**: A  
**Scope**: Replicator Safety Kernel (RSK) lockfile-repair diagnostics only

## Decision

The RSK lockfile-repair diagnostic must begin from a clean exact-head checkout and must fail if pinned Cargo modifies or creates any repository path other than `Cargo.lock`.

The diagnostic evidence bundle must explicitly bind:

- exact subject Git head;
- pull-request base head when applicable;
- root `Cargo.toml`;
- `rust-toolchain.toml`;
- all three RSK manifests;
- base, committed-head, and post-Cargo lockfiles;
- Rust/Cargo toolchain identity;
- post-Cargo worktree status.

Staged changes or untracked files after Cargo are always rejected. Tracked changes are permitted only for `Cargo.lock`; commit-verification mode separately requires even that file to be byte-idempotent.

## Rationale

The repair diagnostic intentionally permits Cargo to rewrite `Cargo.lock` in generation mode. That narrow exception must not become permission for tool-induced mutation elsewhere in the source tree. Explicit worktree-purity checks preserve the intended evidence boundary.

## Evidence boundary

This remains NON-ADMISSIBLE diagnostic evidence. Worktree purity and a qualified lockfile transition are prerequisites only; they do not establish compilation, runtime admission, or replication authority.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
