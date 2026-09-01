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

## Protocol self-test

The hydration control flow has an independent local contract test:

```bash
bash scripts/test-cogsec-lock-hydration.sh
```

The self-test creates disposable Git repositories and stubbed `rustc`/`cargo` executables. It verifies:

- shell syntax;
- hydration stops before qualification while `Cargo.lock` is uncommitted;
- the same command enters qualification after the lock candidate is committed;
- non-additive lock churn is rejected and rolled back;
- dirty/untracked worktrees are rejected before mutation;
- wrong pinned toolchains are rejected before mutation.

This is control-flow evidence only. Stubbed Cargo cannot establish that the real Symthaea workspace resolves, compiles, tests, or passes Clippy.

## Pass 1: hydration

If one or more CogSec workspace packages are absent from `Cargo.lock`, the utility:

1. requires a completely clean tracked and untracked worktree;
2. records the current HEAD and lockfile SHA-256;
3. runs Cargo metadata once without `--locked` using pinned Cargo 1.96.0;
4. permits Cargo to modify only `Cargo.lock`;
5. rejects any deletion or rewrite of pre-existing lockfile material, so CogSec hydration is additive-only;
6. verifies that all required CogSec package entries are present;
7. immediately re-runs `cargo metadata --locked`;
8. runs `git diff --check -- Cargo.lock`;
9. preserves only the Cargo-generated lockfile diff.

It does **not** run the focused qualification suite while the lockfile is uncommitted.

The operator must review the generated diff and commit it without manually synthesizing or editing dependency entries.

## Additive-only blast-radius ratchet

CogSec lock hydration is not a general dependency refresh.

After Cargo produces the candidate lock, the utility inspects the `Cargo.lock` diff. Any removed pre-existing line causes hydration to fail and the original lockfile to be restored.

This deliberately rejects cases where resolving the CogSec workspace would also rewrite, replace, downgrade, upgrade, or remove pre-existing dependency state. Those changes may be legitimate, but they require a separately reviewed lock-maintenance change rather than being hidden inside CogSec qualification work.

The intended successful pass-1 shape is therefore monotonic:

> existing lock state + Cargo-required CogSec material

not:

> opportunistic workspace dependency re-resolution.

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

## Assurance levels

Do not collapse these results into one green label:

1. **Protocol self-test** — shell/control-flow and rollback behavior under stubbed tools.
2. **Pinned local qualification** — real Cargo 1.96.0 hydration followed by the seven focused gates on the exact committed HEAD.
3. **Hosted qualification** — GitHub Actions execution/check evidence for that HEAD.

A stronger level cannot be inferred from a weaker one.

## Non-claims

This protocol does not claim:

- that the current stacked lockfile has already been hydrated;
- that the new CogSec crates compile;
- that tests or Clippy pass;
- that GitHub Actions has executed;
- that Cargo registry/network inputs are independently authenticated beyond Cargo's normal mechanisms;
- that an unrelated lockfile rewrite can or should be forced through this narrow hydration path;
- that a locally successful qualification substitutes for required hosted CI or review policy.

It only defines a deterministic, fail-closed path from an incomplete committed lockfile to a narrowly bounded Cargo-generated candidate, and then from a reviewed committed lockfile to focused executable qualification.
