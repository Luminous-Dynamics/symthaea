# AI Assurance Trusted CPU Recovery

## Purpose

This document defines the queue-neutral correctness/reproducibility recovery path for the frozen AI Assurance foundation when GitHub-hosted `ubuntu-latest` assignment latency prevents the ordinary focused workflow from executing.

It does **not** weaken the hosted merge gate, convert queued state into evidence, or authorize performance claims from a different runner substrate.

The recovery workflow is:

- `.github/workflows/self-hosted-ai-assurance-foundation-recovery.yml`
- `workflow_dispatch` only;
- `main` only;
- scheduled only by the unique `symthaea-trusted-cpu-v1` runner capability;
- `permissions: {}`;
- no third-party Actions.

The trusted CPU runner itself remains governed by `docs/operations/GITHUB_ACTIONS_NIXOS_RUNNER.md` and issue #75.

## Frozen source identity

The v1 recovery harness is deliberately **not generic**. It accepts no SHA/ref input.

It is frozen to:

- target commit: `9fa6c767f9acd5af1aeda38080349840c6cd2dc1`
- target tree: `5889bfe2fefcf4ede72e57cf7d318e5ff1b88940`
- known base: `9748eeba5cca2e62fbdac0d677dee716488f7c9c`
- target hosted AI Assurance workflow blob: `acfae727dd34ab1a4dcc36da6517ec57607a88fa`

This is the normalized, read-only #121 foundation head whose ordinary focused workflow is AI Assurance run #45.

Any later foundation head requires a separately reviewed recovery-harness update. Do not turn this workflow into an arbitrary SHA executor merely for convenience.

## Exact diff allowlist

Before any target code is built or tested, the recovery job proves the frozen target descends from the known base and that its complete changed-file set is exactly:

- `.github/workflows/ai-assurance.yml`
- `Cargo.lock`
- `crates/core/symthaea-ai-assurance/Cargo.toml`
- `crates/core/symthaea-ai-assurance/README.md`
- `crates/core/symthaea-ai-assurance/src/action.rs`
- `crates/core/symthaea-ai-assurance/src/capability.rs`
- `crates/core/symthaea-ai-assurance/src/lib.rs`
- `crates/core/symthaea-ai-assurance/src/trusted.rs`
- `crates/core/symthaea-ai-assurance/tests/trust_domain_attacks.rs`
- `docs/architecture/AI_ASSURANCE_KERNEL_V0_1.md`
- `docs/architecture/AI_ASSURANCE_TRUST_DOMAINS_V0_1.md`

Therefore the unmerged target cannot alter the root Cargo manifests, `.cargo` configuration, Nix harness, runner module, flake pins, Rust toolchain declaration, or unrelated workspace code while still satisfying the recovery harness.

The target workflow blob is checked separately so the gate being reproduced is attributable to the exact focused workflow version on the frozen tree.

## Trusted harness / unmerged source separation

The job creates two independent worktrees under `RUNNER_TEMP`:

### Trusted harness

Checked out from the exact `main` commit that defines the recovery workflow.

The runner policy test and `nix/ci-rust-shell.nix` come from this trusted harness, not from the unmerged target tree.

### Frozen source

Checked out at the exact target commit/tree after ancestry, tree, workflow-blob, and changed-path verification.

Its `flake.lock`, `rust-toolchain.toml`, root Cargo configuration, and other infrastructure are known-base content because the allowlist prohibits target changes to those files.

## Reproduced gate

Inside the pinned trusted CPU Nix shell the workflow records tool versions and executes:

```bash
cargo fmt --package symthaea-ai-assurance -- --check
cargo metadata --locked --format-version 1
cargo test --locked -p symthaea-ai-assurance
cargo clippy --locked -p symthaea-ai-assurance --all-targets -- -D warnings
```

This matches the semantic/reproducibility surface of the frozen hosted AI Assurance workflow:

- package formatting;
- locked workspace resolution;
- unit tests;
- external/adversarial integration tests;
- compile-fail doctests reached by `cargo test`;
- Clippy with warnings denied.

The job also runs the eval-only trusted-runner policy test from trusted `main` before executing the target gate.

## Immutability

Cargo home and target output live under `RUNNER_TEMP`.

After qualification, both trusted harness and frozen source must remain:

- at their exact original commits/trees;
- with no tracked diff;
- with no staged diff;
- with no untracked or ignored workspace mutations.

A mutated source tree is not accepted as recovery evidence.

## Recovery attestation

A successful run emits a textual manifest containing at least:

- schema id;
- GitHub run id/attempt;
- runner identity/OS/architecture;
- trusted harness commit;
- target commit/tree/base;
- hosted workflow blob;
- nixpkgs revision;
- Rust channel;
- `flake.lock`, `rust-toolchain.toml`, and `Cargo.lock` SHA-256 identities;
- exact gate description;
- locked dependency mode;
- evidence scope.

The manifest itself is hashed with SHA-256 and printed into the run log/step summary.

This is local run evidence, not yet externally signed or independently witnessed evidence.

## Evidence interpretation

A PASS may support this narrow statement:

> The frozen #121 source tree passed its focused formatting, locked-resolution, Rust test/doctest, and Clippy gate on the recorded trusted NixOS CPU correctness substrate using the recorded pinned toolchain/runtime context.

It must **not** be promoted into claims that:

- GitHub-hosted CI passed;
- full repository CI passed;
- Showroom Integrity passed on the same recovery run;
- timing/throughput/RSS are comparable to GitHub-hosted runners;
- later stacked assurance PRs were qualified;
- external independent assurance exists;
- the trusted CPU host is safe for arbitrary branch/fork code.

The ordinary hosted #121 run remains valuable independent evidence when runner capacity returns.

## Activation sequence

Do not dispatch the recovery gate until all of the following are true:

1. the runner support has been reviewed and landed on `main`;
2. the NixOS runner module is enabled only on a dedicated/disposable trusted CPU host or VM;
3. the external registration access token is root-owned, outside the Nix store, and scoped exactly as documented;
4. GitHub shows one runner with the exact `symthaea-trusted-cpu-v1` capability;
5. `.github/workflows/self-hosted-runner-smoke.yml` passes from `main`;
6. the trusted-runner eval policy test passes;
7. the operator verifies there are no unrelated secrets, privileged sockets, or sensitive mounts on the host;
8. only then dispatch `Trusted Recovery - AI Assurance Foundation` from `main`.

If the smoke fails, do not run assurance/research recovery jobs until the runner defect is understood.

## Relationship to later assurance work

This v1 workflow is intentionally frozen to #121.

Do not reuse its PASS to qualify #186, #196, v1.3, or later control-plane/recovery work. If hosted capacity remains unavailable and a later exact tree needs trusted-CPU recovery, add a **separately frozen target harness** with its own commit/tree/base/workflow/diff allowlist and evidence interpretation.

This preserves the same rule used throughout the assurance program: evidence applies only to the exact claim and exact tree it actually tested.
