# Mycelix Development Workflow

## Purpose

This document defines the default developer workflow for Mycelix work from the
monorepo root.

It standardizes four things:

1. faster Rust test execution with `cargo-nextest`
2. background check loops with `bacon`
3. safer default deployment with `deploy-rs`
4. later-stage fleet orchestration with `Colmena`

## Default Environment

From `/srv/luminous-dynamics`, enter the shared development environment:

```bash
nix develop .
```

The root shell now includes:

- `cargo-nextest`
- `bacon`
- `deploy`
- `colmena`

It also exports convenience functions for common Mycelix targets:

- `lum_check_sensorium`
- `lum_check_personal`
- `lum_check_commons`
- `lum_check_health`
- `lum_check_finance`
- `lum_check_knowledge`
- `lum_check_pulse`

These are intended for fast local iteration when you are already in the repo
root shell.

## Fast Checks With Bacon

Use the repo-level `bacon.toml` from the monorepo root:

```bash
bacon
```

Useful jobs:

- `check-sensorium`
- `check-personal`
- `check-commons`
- `check-health`
- `check-finance`
- `check-knowledge`
- `check-pulse`
- `test-sensorium`
- `test-health`
- `test-finance`
- `test-knowledge`
- `test-pulse`

Recommended usage during frontend work:

- run `bacon` in one terminal
- keep the default job on the app you are actively editing
- use `cargo check --manifest-path ...` or the `lum_check_*` helpers for one-off verification
- use repo-root `nix develop . -c ...` commands for reproducible scripted checks

## Faster Tests With Cargo Nextest

Prefer `cargo-nextest` over raw `cargo test` for routine Rust test runs.

Examples:

```bash
cargo nextest run --manifest-path mycelix-health/Cargo.toml
cargo nextest run --manifest-path mycelix-finance/Cargo.toml
cargo nextest run --manifest-path mycelix-workspace/mycelix-pulse/apps/leptos/Cargo.toml
```

Why this is the default:

- better process isolation
- faster execution on larger suites
- cleaner failure surfaces for multi-crate work

Use plain `cargo test` only when a crate or test harness is not nextest-friendly
yet.

## Deployment Recommendation

### Default: `deploy-rs`

For Mycelix deployments, `deploy-rs` should be the default first choice.

Why:

- simpler rollout model
- good fit for flake-based deployment
- rollback protection is the most valuable early safety feature

Use it first when the priority is:

- safe shell/frontend rollout
- service updates on a small number of nodes
- avoiding bad remote changes that strand the machine

### Later: `Colmena`

Adopt `Colmena` when Mycelix deployment becomes a real fleet-orchestration
problem.

Use it when the priority is:

- many nodes
- parallel host evaluation and rollout
- grouped deployment across classes of machines

`Colmena` is valuable, but it should not replace the simpler deployment path
until the operational shape actually requires it.

## Practical Rule

For current Mycelix work, the preferred sequence is:

1. develop inside `nix develop .`
2. keep `bacon` running during active edits
3. run `cargo nextest run` for routine verification
4. use `deploy-rs` for early real deployments
5. introduce `Colmena` when the deployment surface becomes a fleet

## Sensorium And Frontend Work

For Sensorium and the Leptos frontends, prefer checks from the repo root so the
environment stays consistent:

```bash
lum_check_sensorium
lum_check_personal
lum_check_commons
lum_check_pulse
```

For scripted or CI-like verification, prefer:

```bash
nix develop . -c /run/current-system/sw/bin/zsh -lc 'RUSTC_WRAPPER= SCCACHE_DISABLE=1 cargo check --manifest-path /srv/luminous-dynamics/mycelix-sensorium/Cargo.toml'
```

That pattern keeps local interactive work fast while preserving a reproducible
verification path.
