# SYM-ARCH-002A Trusted Exact-Tree Recovery

## Purpose

This is a one-purpose recovery path for pull request #57 while GitHub-hosted runner assignment is severely delayed.

It is **not** a generic pull-request runner and it is **not** a new experiment. It exists only to execute the frozen SYM-ARCH-002A correctness gate against the already-reviewed exact source tree on the hardened trusted CPU runner.

The recovery workflow is:

`.github/workflows/self-hosted-sym-arch-002a-core-recovery.yml`

It is `workflow_dispatch` only, main-defined, grants `permissions: {}`, accepts no target/ref inputs, and can schedule only on `symthaea-trusted-cpu-v1`.

## Frozen target

The v1 recovery profile is hard-bound to:

- PR: `#57`
- target commit: `f61f5ca04700db90a4f33baca5e58cd1daf068c9`
- target tree: `ca6cfd632f614144f3be1362d51b2222d60a664f`
- known base: `e143c61110a70a361111bda91a986caa2924489a`
- hosted workflow Git blob: `c48b655bfb1a233e29d4a1bf478985d9acb37c11`

The allowed target-side diff is exactly:

```text
.github/workflows/sym-arch-002a-core.yml
crates/domains/symthaea-psych-bench/src/experiment/confirmatory.rs
crates/domains/symthaea-psych-bench/src/experiment/mod.rs
crates/domains/symthaea-psych-bench/src/lib.rs
docs/research/SYM_ARCH_002A_EXPERIMENTAL_CORE_V1.md
```

Any commit/tree/workflow-blob/path mismatch fails before target code is executed.

## Trust separation

The workflow maintains two independent temporary Git worktrees:

1. **trusted harness** — fetched from the exact `main` commit that defined the recovery workflow (`GITHUB_SHA`);
2. **frozen source** — fetched from the hard-coded #57 commit.

Runner-policy tests and `nix/ci-rust-shell.nix` come only from the trusted harness. They are never imported from the unmerged target tree.

The Rust shell is parameterized so the shell implementation comes from the trusted harness while the actual nixpkgs, rust-overlay, and `rust-toolchain.toml` inputs come from the frozen target. This preserves #57's dependency/toolchain boundary without trusting its helper infrastructure.

## Executed correctness gate

After provenance checks pass, the frozen source is run through:

```bash
rustfmt --edition 2024 --check \
  crates/domains/symthaea-psych-bench/src/experiment/mod.rs \
  crates/domains/symthaea-psych-bench/src/experiment/confirmatory.rs \
  crates/domains/symthaea-psych-bench/src/lib.rs

cargo test --locked -p symthaea-psych-bench --lib experiment -- --nocapture
cargo check --locked -p symthaea-psych-bench --lib
```

These are the hosted SYM-ARCH-002A correctness commands with the additional fail-closed `--locked` constraint on Cargo dependency resolution.

Cargo home and target state are placed under `RUNNER_TEMP`, outside both Git worktrees.

At completion both trusted-harness and frozen-source worktrees must remain at their expected commit/tree with no tracked, staged, untracked, or ignored workspace mutations.

## Recovery attestation

A PASS emits a canonical manifest with schema:

`symthaea.sym-arch-002a.trusted-recovery.v1`

The manifest binds:

- GitHub run ID and attempt;
- runner name, OS, and architecture;
- trusted harness commit;
- target commit and tree;
- known base;
- hosted-workflow blob;
- target nixpkgs revision and Rust channel;
- SHA-256 of target `flake.lock` and `rust-toolchain.toml`;
- gate profile;
- locked Cargo dependency mode.

The workflow prints the manifest and its SHA-256 and adds the same provenance to `GITHUB_STEP_SUMMARY`. The attestation digest is run-specific operational provenance, not a cryptographic signature and not architecture-performance evidence.

## Evidence interpretation

A successful recovery run can be used as **executor-equivalent correctness evidence for this exact frozen #57 tree** only when all of the following are true:

- the trusted CPU runner's main-only smoke has already passed;
- the recovery workflow itself is running from trusted `main`;
- target commit, tree, base ancestry, hosted-workflow blob, and exact file surface all match the frozen values above;
- the trusted-runner policy evaluation passes;
- rustfmt, the `experiment` unit-test slice, and psych-bench library check all pass;
- both worktrees remain immutable;
- the canonical recovery manifest is emitted with `result=PASS`;
- the recovery run URL, manifest contents, and attestation SHA-256 are posted to #57 before any merge decision.

This recovery evidence is not evidence about architecture performance and must not be used for A6 timing/throughput/RSS claims.

A queued or cancelled GitHub-hosted run remains neither PASS nor FAIL. If the hosted #57 run later executes, preserve its result as additional evidence rather than rewriting the recovery result.

## Merge boundary

A recovery PASS does not authorize changing any SYM-ARCH-002A scientific thresholds, manifest semantics, benchmark definitions, or claim rules. It only answers whether the frozen exact tree passes its correctness gate on the trusted CPU executor.

If the recovery run fails, repair only the observed defect and create a new versioned exact-tree recovery profile for the repaired target. Do not silently retarget this frozen profile.

Once #57 has a durable executable conclusion and the hosted-runner incident is resolved, this one-purpose workflow may be retired; its historical run record and this document should remain as provenance.
