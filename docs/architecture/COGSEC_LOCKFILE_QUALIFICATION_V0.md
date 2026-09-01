# CogSec Lockfile Qualification v0

Status: pre-enforcement reproducibility protocol.

This document defines how the stacked CogSec workspace hydrates and qualifies `Cargo.lock` without weakening the existing `--locked` CI boundary.

## Invariant

A lockfile may be either generated or qualified in one worktree state, but a newly generated uncommitted lockfile is never described as qualified.

The focused qualification gate intentionally begins with `cargo metadata --locked` and requires `git diff --exit-code -- Cargo.lock`. That means qualification applies to a committed repository state, not to an operator's uncommitted candidate lockfile.

## Pinned toolchain

Lock hydration and qualification require the repository toolchain pinned by `rust-toolchain.toml`:

- rustc 1.96.0
- cargo 1.96.0

The hydration utility refuses another Rust or Cargo version rather than silently producing a different dependency resolution artifact.

## Canonical command

Run from a clean checkout of the target CogSec stack:

```bash
bash scripts/cogsec-hydrate-lock-and-qualify.sh
```

The utility is intentionally invokable through `bash`; executable-bit transport is not part of the evidence claim.

## Pass 1: hydration

If one or more CogSec workspace packages are absent from `Cargo.lock`, the utility:

1. requires a completely clean tracked and untracked worktree;
2. records the current HEAD and lockfile SHA-256;
3. runs Cargo metadata once without `--locked` using pinned Cargo 1.96.0;
4. permits Cargo to modify only `Cargo.lock`;
5. verifies that all required CogSec package entries are present;
6. immediately re-runs `cargo metadata --locked`;
7. runs `git diff --check -- Cargo.lock`;
8. preserves only the Cargo-generated lockfile diff.

It does **not** run the focused qualification suite while the lockfile is uncommitted.

The operator must review the generated diff and commit it without manually synthesizing or editing dependency entries.

## Pass 2: qualification

After committing the generated `Cargo.lock`, rerun the same command from a clean worktree.

If the lockfile is already complete and stable, the utility runs `scripts/cogsec-focused-qualification.sh`, which performs the canonical focused gates:

1. locked workspace metadata consistency;
2. rustfmt;
3. CogSec package `cargo check`;
4. CogSec package tests;
5. legacy S0/S1/S2 control determinism;
6. documentation tests;
7. Clippy with warnings denied.

A successful second pass is evidence about the exact committed HEAD. It is not evidence that GitHub-hosted CI has passed.

## Failure semantics

Before a successful hydration result is deliberately preserved, any error or interrupt restores the original `Cargo.lock` from a temporary backup.

Unexpected filesystem changes are fail-closed. Cargo metadata may change only `Cargo.lock`, and the qualification pass must leave the clean worktree unchanged.

The utility also verifies that HEAD does not move during qualification.

## Why CI does not hydrate

CI must diagnose stale or inconsistent committed dependency state. It must not repair that state implicitly.

If CI were allowed to regenerate `Cargo.lock`, the repository could claim a passing build against dependency resolution that was never reviewed or committed. That would collapse reproducibility evidence into runner-local state.

Therefore:

> CI verifies the lock. Pinned Cargo produces the lock. A reviewed commit promotes the lock into repository state.

## Required CogSec packages

The current focused stack requires lockfile membership for:

- `symthaea-cogsec`
- `symthaea-cogsec-evidence`
- `symthaea-cogsec-qualification`
- `symthaea-cogsec-shadow-runtime`

The list is deliberately explicit so newly added CogSec packages cannot silently fall outside this qualification path.

## Non-claims

This protocol does not claim:

- that the current stacked lockfile has already been hydrated;
- that the new CogSec crates compile;
- that tests or Clippy pass;
- that GitHub Actions has executed;
- that Cargo registry/network inputs are independently authenticated beyond Cargo's normal mechanisms;
- that a locally successful qualification substitutes for required hosted CI or review policy.

It only defines a deterministic, fail-closed path from an incomplete committed lockfile to a Cargo-generated candidate, and then from a reviewed committed lockfile to focused executable qualification.
