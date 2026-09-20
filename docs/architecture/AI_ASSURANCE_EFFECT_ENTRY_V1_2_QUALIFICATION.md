# AI Assurance Effect-Entry v1.2 Qualification

This document records the qualification boundary for the effect-entry linearization tranche in PR #196.

## Dependency boundary

PR #196 is stacked on PR #186 (`research/ai-assurance-budget-recovery-v1.1`).

The previous failing PR #196 qualification run tested the effect-entry head against lower-stack commit `546049fb4c9cbbab27a22781527d795df1bc2cef`. That snapshot predates the current recovery-branch normalization and therefore is not valid evidence about the current stacked source.

The repaired lower source head used for the next qualification attempt is:

- PR #186 head: `993055b49208ecc6786a382b657ab64722938a92`
- prior normalized source head: `c146f760ff918230c3af747d8e6df8c8fff31dc9`

The lower repair includes the observed `JoinHandle::join` ownership correction, committed assurance-crate lockfile resolution, rustfmt normalization, test-only import cleanup, and a non-mutating AI Assurance qualification workflow.

On 2026-09-01 the live PR #196 base pointer was explicitly refreshed to `993055b49208ecc6786a382b657ab64722938a92` before this qualification-trigger commit. Any earlier run whose pull-request payload still records `546049fb4c9cbbab27a22781527d795df1bc2cef` remains stale evidence and must be ignored for v1.2 qualification.

## Previous red-run diagnosis

The previous PR #196 job had two independent failure classes:

1. A real Rust compile failure in the lower-stack `budget_public_api.rs` concurrency test. `Iterator::filter` passed a shared reference to each `JoinHandle`, while `JoinHandle::join(self)` requires ownership. The current lower branch consumes handles through `map` before filtering the join results.
2. Qualification checkout drift. The workflow ran mutating `cargo fmt` and unlocked `cargo metadata` before checking the diff. That could both rewrite formatting and synthesize a missing `Cargo.lock` entry during qualification. The current lower workflow uses `cargo fmt ... -- --check` and begins dependency resolution with `cargo metadata --locked`.

Neither failure is evidence that the effect-entry state machine itself violated its concurrency contract.

## v1.2 qualification gate

Treat PR #196 as qualified only when a fresh merge ref against the current PR #186 head demonstrates all of the following with Rust 1.96.0:

1. `cargo fmt --package symthaea-ai-assurance -- --check`
2. `cargo metadata --locked --format-version 1`
3. `cargo test --locked -p symthaea-ai-assurance`
4. compile-fail doctests included by the package test run
5. `cargo clippy --locked -p symthaea-ai-assurance --all-targets -- -D warnings`
6. no tracked changes to `Cargo.lock` or `crates/core/symthaea-ai-assurance` after qualification

A run against an older lower-stack SHA does not satisfy this gate.

## Effect-entry invariants under qualification

The v1.2 source must continue to preserve:

- acquisition and revocation share one short linearization lock;
- revocation latches new admission closed until explicit quiescent resume;
- a pre-stop ticket cannot acquire after epoch rotation, including after resume;
- a permit that wins before stop represents exactly one already-admitted effect;
- dropping an unused permit repairs outstanding accounting;
- normal return and unwind repair in-flight accounting;
- arbitrary effect callbacks do not execute while the domain mutex is held;
- tickets and permits bind the exact action, authority-snapshot, and adapter-semantics commitment;
- wrong-domain and commitment substitution fail closed;
- the acquisition/revocation race has only the two documented total-order outcomes.

## Evidence boundary

Passing this gate qualifies the Rust implementation, tests, doctests, formatting, locked dependency resolution, and Clippy for this stack snapshot. It does not by itself prove that a concrete adapter faithfully implements the semantics named by `adapter_semantics_digest`, nor that admission evidence survives a process crash or panic before trusted host code persists it.

The next semantic refinement after qualification remains evidence-before-adapter-entry: split permit consumption into a begin/in-flight guard so trusted host code can persist the admission receipt before arbitrary adapter code runs.

## Non-claims

- no claim of end-to-end MAGI/tool integration;
- no claim of composite multi-domain atomic admission yet;
- no claim that already-admitted external effects are retroactively cancellable;
- no claim of durable crash recovery or durable monotonic time;
- no claim that commitment digests are attestations without trusted derivation;
- no claim that GitHub `action_required` is a passing or failing code verdict.
