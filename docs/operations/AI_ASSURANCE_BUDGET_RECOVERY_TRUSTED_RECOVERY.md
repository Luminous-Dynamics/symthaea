# AI Assurance #186 Trusted CPU Recovery

## Purpose

This document defines the queue-neutral correctness/reproducibility recovery path for the frozen AI Assurance budget-recovery tranche in pull request #186 when GitHub-hosted `ubuntu-latest` assignment is unavailable or severely delayed.

This recovery path does **not** weaken the hosted gate, convert queued state into PASS/FAIL evidence, authorize arbitrary branch execution, or make performance claims from a different runner substrate.

The recovery workflow is:

- `.github/workflows/self-hosted-ai-assurance-budget-recovery.yml`
- `workflow_dispatch` only;
- `main` only;
- `permissions: {}`;
- scheduled only by the unique `symthaea-trusted-cpu-v1` capability;
- no target SHA/ref inputs.

The runner itself remains governed by `docs/operations/GITHUB_ACTIONS_NIXOS_RUNNER.md` and issue #75.

## Frozen target identity

The v1 #186 recovery profile is hard-bound to:

- pull request: `#186`
- target commit: `251c47c5a1f09015ab6c29794df7ac5b6efae373`
- target tree: `1451a57e2dcaf50de6b5b43776364a5af8b78f15`
- known lower-stack base: `6ca61356c6ff49b2ba77cae332f006d054d5f84d`
- hosted AI Assurance workflow blob: `35bbddd9d6a7f8ad77073e4ab4faf6722f00deea`

A later #186 source head requires a new reviewed recovery profile. Do not turn this workflow into a generic SHA executor.

## Exact changed-path boundary

Before target code executes, the recovery workflow proves that the complete diff from the frozen lower-stack base is exactly:

- `.github/workflows/ai-assurance.yml`
- `Cargo.lock`
- `crates/core/symthaea-ai-assurance/src/action.rs`
- `crates/core/symthaea-ai-assurance/src/budget.rs`
- `crates/core/symthaea-ai-assurance/src/budget_guard.rs`
- `crates/core/symthaea-ai-assurance/src/budget_purpose.rs`
- `crates/core/symthaea-ai-assurance/src/capability.rs`
- `crates/core/symthaea-ai-assurance/src/effect_guard.rs`
- `crates/core/symthaea-ai-assurance/src/host.rs`
- `crates/core/symthaea-ai-assurance/src/independence.rs`
- `crates/core/symthaea-ai-assurance/src/lib.rs`
- `crates/core/symthaea-ai-assurance/src/policy.rs`
- `crates/core/symthaea-ai-assurance/src/policy_guard.rs`
- `crates/core/symthaea-ai-assurance/src/resolution.rs`
- `crates/core/symthaea-ai-assurance/src/resource.rs`
- `crates/core/symthaea-ai-assurance/src/temporal_policy.rs`
- `crates/core/symthaea-ai-assurance/src/trusted.rs`
- `crates/core/symthaea-ai-assurance/tests/budget_guard_public_api.rs`
- `crates/core/symthaea-ai-assurance/tests/budget_public_api.rs`
- `crates/core/symthaea-ai-assurance/tests/budget_purpose_public_api.rs`
- `crates/core/symthaea-ai-assurance/tests/policy_public_api.rs`
- `crates/core/symthaea-ai-assurance/tests/policy_revocation_public_api.rs`
- `crates/core/symthaea-ai-assurance/tests/trust_domain_attacks.rs`

Therefore this frozen target cannot change root Cargo manifests, `.cargo` configuration, Nix runner/helper code, flake inputs, Rust toolchain declaration, build scripts, or unrelated workspace source while still satisfying the recovery profile.

`Cargo.lock` is part of the frozen target tree and is hashed in the emitted attestation. Dependency resolution is executed with `--locked`.

## Trusted harness separation

The job creates two independent work areas under `RUNNER_TEMP`:

1. **trusted harness** — exact `main` commit defining the recovery workflow and trusted-runner policy;
2. **frozen source** — exact #186 target commit/tree.

Runner policy tests and `nix/ci-rust-shell.nix` come from the trusted harness, not from the unmerged #186 source.

The target's exact source/toolchain/dependency lineage is still preserved: target commit/tree, known base, hosted-workflow blob, `flake.lock`, `rust-toolchain.toml`, and `Cargo.lock` identities are recorded or checked explicitly.

## Reproduced focused gate

Inside the pinned trusted CPU Nix shell, the workflow runs:

```bash
cargo fmt --package symthaea-ai-assurance -- --check
cargo metadata --locked --format-version 1
cargo test --locked -p symthaea-ai-assurance
cargo clippy --locked -p symthaea-ai-assurance --all-targets -- -D warnings
```

This covers the same focused correctness/reproducibility surface as the hosted AI Assurance lane:

- package formatting;
- committed locked dependency resolution;
- unit tests;
- public/adversarial integration tests;
- compile-fail doctests reached by the package test run;
- Clippy with warnings denied.

The trusted harness must also pass both runner-policy and trusted-routing evaluations before target code executes.

## Immutability

Cargo home and target output are placed under `RUNNER_TEMP` outside the source trees.

At completion, both harness and frozen source must retain their exact commits/trees and have no tracked, staged, untracked, or ignored mutations. A run that repairs or rewrites the target checkout is not qualification evidence.

## Recovery attestation

A successful run emits schema:

`symthaea.ai-assurance.budget-recovery.trusted-recovery.v1`

The manifest binds at least:

- GitHub run ID/attempt;
- runner identity, OS, and architecture;
- trusted harness commit;
- target commit/tree/base;
- hosted workflow blob;
- nixpkgs revision;
- Rust channel;
- SHA-256 identities of `flake.lock`, `rust-toolchain.toml`, and `Cargo.lock`;
- exact focused gate;
- locked dependency mode;
- evidence scope.

The manifest and its SHA-256 should be retained with the #186 qualification trail.

## Evidence interpretation

A PASS can support only the narrow claim:

> Frozen #186 head `251c47c5...` passed its focused formatting, locked-resolution, Rust test/doctest, and Clippy gate on the recorded trusted NixOS CPU correctness substrate under the recorded pinned runtime/toolchain context.

It does **not** prove:

- GitHub-hosted run #67 passed;
- full repository CI passed;
- later #196/v1.2 or #271/v1.3 trees passed;
- self-hosted timing/throughput/RSS is comparable to GitHub-hosted execution;
- external independent assurance exists;
- the trusted runner is safe for arbitrary PR/fork code.

If hosted run #67 later executes, preserve its conclusion as independent additional evidence. Never rewrite a queued hosted run into PASS because recovery succeeded.

## Activation sequence

Do not dispatch the #186 recovery workflow until all are true:

1. the trusted-runner support has been reviewed and landed on trusted `main`;
2. a dedicated/disposable CPU host or VM is configured with no unrelated secrets, sensitive mounts, or privileged host-control sockets;
3. the external registration credential is root-owned, outside the Nix store, and scoped as documented;
4. GitHub reports the exact `symthaea-trusted-cpu-v1` capability online;
5. `.github/workflows/self-hosted-runner-smoke.yml` succeeds from `main`;
6. the runner-policy and routing-exclusivity evaluations succeed;
7. only then dispatch `Trusted Recovery - AI Assurance Budget Recovery` from `main`.

If any trust-policy or smoke gate fails, do not execute the frozen assurance target until the runner defect is understood.

## Promotion boundary

A recovery PASS for #186 does not automatically promote #196.

The already-prepared v1.2 composition must still be audited against this exact #186 tree and receive its own focused qualification. Evidence remains exact-tree and exact-claim scoped throughout the stack.
