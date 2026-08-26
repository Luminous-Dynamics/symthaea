# SYM-ARCH-002A Offline Exact-Tree Validation

## Why this exists

GitHub-hosted runner assignment for SYM-ARCH-002A PR #57 has experienced extreme queue latency. A repository-scoped self-hosted runner is not the preferred escape hatch because Symthaea is public: a pull request can introduce workflow code that targets a self-hosted runner unless an independent GitHub policy boundary prevents it.

GitHub currently recommends self-hosted runners primarily for private repositories. Organization/enterprise runner groups can reduce this risk when they are configured to allow only selected workflows pinned to an exact trusted branch, tag, or SHA, but that is an organization-level control and must be verified independently before connecting a runner to this public repository.

The v1 fallback therefore does **not** register a machine with GitHub Actions. It is an operator-run correctness validator executed on a disposable NixOS x86_64 host or VM.

## Validator

Run:

```bash
scripts/ci/validate-sym-arch-002a-core.sh
```

Optionally pass a **new, nonexistent** evidence output directory as the only positional argument:

```bash
scripts/ci/validate-sym-arch-002a-core.sh /safe/path/to/new-evidence-directory
```

The validator refuses to reuse an existing evidence directory and refuses to place evidence inside its own source checkout.

The script accepts no target SHA, branch, or ref arguments. Its v1 target is frozen to:

- PR: `#57`
- target commit: `f61f5ca04700db90a4f33baca5e58cd1daf068c9`
- target tree: `ca6cfd632f614144f3be1362d51b2222d60a664f`
- known base: `e143c61110a70a361111bda91a986caa2924489a`
- hosted workflow Git blob: `c48b655bfb1a233e29d4a1bf478985d9acb37c11`
- executor system: `x86_64-linux`

The allowed target-side review surface is exactly:

```text
.github/workflows/sym-arch-002a-core.yml
crates/domains/symthaea-psych-bench/src/experiment/confirmatory.rs
crates/domains/symthaea-psych-bench/src/experiment/mod.rs
crates/domains/symthaea-psych-bench/src/lib.rs
docs/research/SYM_ARCH_002A_EXPERIMENTAL_CORE_V1.md
```

Any commit, tree, ancestry, workflow-blob, path, operating-system, or system-architecture mismatch fails before target tests execute.

## Trusted control-plane files

Before evaluating target Nix or compiling anything, the validator proves that #57 did not change:

- `flake.nix`
- `flake.lock`
- `rust-toolchain.toml`
- root `Cargo.toml`
- `Cargo.lock`

Those blobs must be byte-identical between the frozen base and target.

This matters because the validator uses the target's pinned Nix/Rust/Cargo dependency boundary while taking the minimal shell and sandbox implementation from the reviewed validator checkout.

The harness manifest separately binds SHA-256 values for:

- `nix/ci-rust-shell.nix`
- `scripts/ci/run-sym-arch-002a-sandbox.sh`

so later evidence identifies the exact executor implementation as well as the target tree.

## Operator trust boundary

Run the validator only from a **clean reviewed checkout** of this branch or, once merged, trusted `main`.

The script verifies that its own Git worktree has no tracked, staged, untracked, or ignored mutations and records its harness commit/tree in the evidence manifest.

Use a disposable NixOS x86_64 VM or host with:

- no personal files or browser state;
- no SSH agent, developer keys, cloud credentials, or production secrets;
- no Docker socket or equivalent host-control API;
- no writable mounts from production systems;
- network placement that does not expose sensitive LAN services.

No GitHub access token is required. The validator fetches only public Git objects over HTTPS and disables normal Git credential/config lookup for the frozen-source fetch.

Before that fetch, `HOME`, XDG config/cache/data, Cargo state, and temporary storage are moved into a disposable workspace. GitHub tokens, askpass state, credential-manager interaction, and SSH-agent inheritance are removed.

## Pinned execution sandbox

The target gate uses Bubblewrap from the target-pinned nixpkgs revision. At the current frozen dependency boundary that resolves to Bubblewrap **0.11.2**.

`cargo fetch --locked` occurs before the sandbox solely to populate the isolated dependency cache. The validator then executes formatting, build scripts, tests, and `cargo check` inside a Bubblewrap namespace with:

- all supported namespaces unshared, including the network namespace;
- nested user-namespace creation disabled and asserted disabled;
- all Linux capabilities dropped;
- `/nix/store` mounted read-only;
- the authenticated target mounted read-only at `/workspace`;
- only disposable Cargo home/target, home, tmp, and `/dev/shm` writable;
- a fresh `/proc` and `/dev`;
- no `/root`, `/etc/shadow`, `/run/current-system`, or `/sys` exposure;
- only Nix-store PATH entries retained.

The sandbox self-test fails closed unless:

1. an external TCP connection cannot be opened;
2. `CapEff` in `/proc/self/status` is all zeroes;
3. the forbidden host paths above are absent; and
4. a write probe against `/workspace` fails.

Bubblewrap is unprivileged in the pinned Nix package. If the host kernel cannot provide the required namespace/seccomp behavior, the validator should fail rather than silently downgrade the sandbox.

Bind mounts are established before the new `/proc` mount intentionally; this avoids relying on order-sensitive `/proc` behavior in Bubblewrap setup.

## Executed gate

After identity checks and dependency prefetch, the networkless/read-only sandbox runs:

```bash
rustfmt --edition 2024 --check \
  crates/domains/symthaea-psych-bench/src/experiment/mod.rs \
  crates/domains/symthaea-psych-bench/src/experiment/confirmatory.rs \
  crates/domains/symthaea-psych-bench/src/lib.rs

cargo test --locked --offline -p symthaea-psych-bench --lib experiment -- --nocapture
cargo check --locked --offline -p symthaea-psych-bench --lib
```

`--locked` prevents dependency resolution from rewriting the committed Cargo graph. `--offline` prevents Cargo from resolving or downloading additional dependencies after the explicit prefetch step. Bubblewrap's unshared network namespace separately prevents target build/test code from using the host network even if that code bypasses Cargo.

The read-only source bind means target code cannot modify its own checked-out tree. Cargo build state remains in the disposable external target directory.

## Evidence bundle

A PASS writes private-by-default files under a fresh evidence directory:

- `gate.log`
- `target-paths.txt`
- `provenance.txt`
- `manifest.txt`
- `manifest.sha256`
- `SHA256SUMS`

The manifest schema is:

`symthaea.sym-arch-002a.offline-validator.v1`

It binds:

- PASS result and correctness-only interpretation;
- validation start/completion times;
- validator harness commit/tree;
- CI-shell and sandbox-helper SHA-256 values;
- frozen target commit/tree/base;
- hosted-workflow Git blob and SHA-256;
- NixOS system and pinned nixpkgs revision;
- Rust channel;
- `flake.lock`, `Cargo.lock`, and `rust-toolchain.toml` SHA-256 values;
- exact target path-set hash;
- provenance-record hash;
- gate-log hash;
- locked/offline dependency mode;
- Bubblewrap namespace/capability/source-mount policy;
- exact correctness-gate profile.

The gate log records the Bubblewrap, Cargo, and rustfmt versions actually used. Because the log hash is bound into the manifest, those runtime versions are transitively part of the evidence identity.

The SHA-256 values are integrity identifiers, not signatures.

## Evidence interpretation

A successful run is **correctness evidence for this exact frozen target only**. It is not architecture-performance evidence and must not be used for A6 timing, throughput, RSS, or resource claims.

A queued or cancelled GitHub-hosted workflow remains neither PASS nor FAIL. If the hosted #57 workflow later executes, preserve that result as additional evidence.

Before merging #57 after an offline PASS:

1. post the manifest hash and full provenance to PR #57;
2. preserve the evidence bundle under the normal release/evidence process;
3. verify that GitHub merge policy/branch protection does not independently require the still-pending hosted check;
4. if an administrative policy exception is required, record it separately from the scientific correctness evidence.

Do not silently retarget this validator if #57 changes. Any repaired target requires a new versioned validator profile with new frozen commit/tree/blob values.

## Future GitHub-connected capacity

If a self-hosted GitHub runner is later desired for the public repository, require an organization/enterprise runner group with both:

- repository access restricted to Symthaea; and
- workflow access restricted to an exact allowlist pinned to `refs/heads/main` or full trusted SHAs.

Verify those settings in GitHub organization/enterprise policy **before** registering the runner. A custom label by itself is not a sufficient security boundary for a public repository.

Until that policy boundary is verified, prefer this non-GitHub-connected disposable validator over a repository-scoped self-hosted Actions runner.
