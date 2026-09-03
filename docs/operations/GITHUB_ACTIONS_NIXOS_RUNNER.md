# Symthaea Trusted CPU GitHub Actions Runner

## Purpose

This runner is an execution-capacity fallback for trusted correctness validation when GitHub-hosted assignment latency becomes operationally unacceptable.

It is infrastructure only. Running a workflow here does not change an experimental result, weaken a scientific gate, or turn an unexecuted workflow into evidence.

The first-party fallback is intentionally CPU-only. GPU access is a separate capability and security design because exposing host devices weakens the default NixOS GitHub-runner isolation boundary.

## Fixed trust boundary

Symthaea is public, so the runner must not be a generic `self-hosted` execution target for arbitrary pull-request code.

`nix/modules/github-actions-runner.nix` intentionally exposes only three host-facing options:

- `enable`
- `name`
- `tokenFile`

Everything security-sensitive is fixed in the module rather than configurable:

- repository: `https://github.com/Luminous-Dynamics/symthaea`
- token type: access token
- lifecycle: ephemeral
- runner self-update: disabled by the pinned nixpkgs module
- GitHub default labels: disabled
- scheduling capability: `symthaea-trusted-cpu-v1`
- JavaScript action runtime: Node 24 only
- Symthaea-specific ambient packages: none
- GPU routing: unavailable

Because GitHub default labels are disabled, this host does not advertise generic `self-hosted`, OS, architecture, or `gpu` labels. Existing generic/GPU workflows therefore cannot accidentally schedule onto it.

Do not broaden that routing surface merely for convenience.

## Trusted-root prerequisite

The trusted runner treats repository `main` as its harness root. That assumption is valid only while `main` is protected against ordinary direct mutation.

Before activating `symthaea-trusted-cpu-v1`, satisfy issue #330 and verify from GitHub that an effective branch-protection or repository-ruleset policy protects `main`. At minimum, ordinary direct pushes and force pushes must not be able to rewrite the trusted workflow/Nix harness outside the reviewed promotion path.

A main-only workflow is not a sufficient trust boundary when `main` itself is directly mutable.

## Host isolation

Treat the runner as a disposable build appliance, not as a general workstation.

Prefer a dedicated VM or dedicated host with:

- no personal files or browser/session state;
- no SSH agent, developer keys, cloud credentials, or mounted home directories;
- no writable mounts from production systems;
- no privileged Docker socket or equivalent host-control API;
- a network segment that does not expose sensitive LAN services;
- only the external runner-registration credential required by the service.

The NixOS systemd sandbox protects the local host filesystem/process boundary, but the runner still requires outbound network access. Network placement is therefore part of the threat model.

## Credentials

Never place a GitHub token, registration token, private key, or other credential in this repository, a Nix expression, a flake input, an Actions variable, or shell history.

The module accepts only an external runtime `tokenFile` path and rejects `/nix/store/...` credential paths. For v2, use a root-owned runtime secret file managed by sops-nix, agenix, or an equivalent mechanism that materializes the decrypted token outside the Nix store. Do not assume a service-local `LoadCredential=` path is supported by this wrapper; that integration would need to be designed and tested explicitly.

For the v2 emergency/static-secret deployment, use a **fine-grained personal access token scoped only to `Luminous-Dynamics/symthaea` with repository `Administration: write` and no additional repository permissions**. GitHub currently documents that exact permission as sufficient for the repository self-hosted-runner registration-token endpoint. The registration token generated from that access token expires after one hour; the pinned nixpkgs runner service obtains registration tokens as needed because this module uses `tokenType = "access"`.

The service now enforces runtime credential-file policy before the upstream runner bootstrap is allowed to copy/use the access token. Registration fails closed unless the credential is a regular file owned by root, mode exactly `0400` or `0600`, non-empty, no larger than 4096 bytes, and a single printable non-whitespace token with no newline. The preflight must not print or hash the credential value.

GitHub also supports GitHub App user and installation access tokens for the registration-token endpoint. Those are the preferred longer-term direction, but they require an external mint/refresh mechanism before they can safely back an ephemeral runner that re-registers after each job. Issue #322 tracks that **v3** credential-broker design. Do not place a short-lived App installation token in the v2 static `tokenFile` and assume it will remain valid across runner lifecycles.

The token file must contain exactly the access token with no trailing newline, be owned by root, and have no group/other permission bits. `0400` is preferred; `0600` is acceptable when the secret manager requires it. Verify before enabling the runner:

```bash
sudo stat -c '%U:%G %a %n' /run/secrets/github-runner/symthaea-pat
sudo test "$(stat -c '%U' /run/secrets/github-runner/symthaea-pat)" = root
sudo sh -c 'mode=$(stat -c %a /run/secrets/github-runner/symthaea-pat); test "$mode" = 400 -o "$mode" = 600'
```

Example runtime path:

```nix
/run/secrets/github-runner/symthaea-pat
```

## NixOS configuration

Import the Symthaea module set or the runner module directly:

```nix
{
  imports = [ /path/to/symthaea/nix/modules ];

  services.symthaea-ci-runner = {
    enable = true;
    tokenFile = "/run/secrets/github-runner/symthaea-pat";
  };
}
```

Use a unique `name` when operating more than one trusted CPU host.

## Ephemeral and immutable lifecycle

At Symthaea's pinned nixpkgs revision, the upstream NixOS GitHub-runner module:

- registers with `--disableupdate`;
- runs under a dynamic user by default;
- clears the work directory on service start;
- wipes runner state before each registration in ephemeral mode;
- deregisters after a completed ephemeral job and starts a fresh registration;
- prevents the runner process from reading the original token path;
- defaults to strict filesystem/device/process hardening.

The Symthaea wrapper fixes `ephemeral = true` and `tokenType = "access"`; those knobs are intentionally absent from the public API.

The protected defaults include `DynamicUser`, `PrivateDevices`, `PrivateMounts`, `PrivateUsers`, `PrivateTmp`, `ProtectHome`, `NoNewPrivileges`, `ProtectSystem=strict`, namespace restrictions, and a restrictive umask.

## Minimal validation toolchain

Do not reproduce `ubuntu-latest` by downloading unmanaged Rust binaries or enabling `nix-ld` on this host.

The trusted smoke uses `nix/ci-rust-shell.nix`, a purpose-built validation shell rather than Symthaea's broad general development environment. It reads the Rust version from `rust-toolchain.toml` through the pinned `rust-overlay` input and contains only the tools needed by the CPU validation path:

- pinned Rust toolchain;
- rustfmt;
- Clippy;
- CA certificates;
- GCC/compiler support;
- OpenSSL + headers;
- pkg-config.

The current repository pin is Rust `1.96.0`. Future toolchain updates should flow through `rust-toolchain.toml`, not a runner-specific version string.

The ambient runner service adds **zero** Symthaea-specific packages. The upstream nixpkgs runner already provides the shell/core Git/Nix tooling needed to bootstrap the pinned per-job shell. Node 24 remains packaged by the runner for future explicitly-reviewed JavaScript actions, although the current smoke uses no third-party Actions.

## Eval-only policy test

`nix/tests/eval-github-actions-runner.nix` evaluates the runner module without contacting GitHub or reading a real token.

It verifies:

- repository scope;
- fixed access-token and ephemeral mode;
- suppression of default GitHub labels;
- exact `symthaea-trusted-cpu-v1` routing label;
- Node 24-only action runtime;
- zero Symthaea-specific ambient packages;
- exact three-option public API;
- rejection of Nix-store credential paths and blank runner names;
- root-only runtime credential preflight ordering and content checks;
- credential-value non-disclosure guard for the preflight;
- token-file isolation from the job process;
- pinned upstream systemd hardening;
- restricted address families without raw packet sockets;
- runner self-update remains disabled.

The manual smoke executes the test directly against the flake's pinned nixpkgs input:

```bash
nix build --no-link --no-write-lock-file \
  --impure \
  --expr 'let f = builtins.getFlake (toString ./.); pkgs = import f.inputs.nixpkgs { system = builtins.currentSystem; }; in import ./nix/tests/eval-github-actions-runner.nix { inherit pkgs; }'
```

## Trusted smoke workflow

`.github/workflows/self-hosted-runner-smoke.yml` is deliberately `workflow_dispatch` only and **main-only**.

`workflow_dispatch` normally permits choosing a branch/ref, so the job rejects anything except `refs/heads/main`. This is meaningful only after #330's trusted-root protection is active. The smoke performs an unauthenticated Git fetch of the exact `GITHUB_SHA`, verifies the detached `HEAD`, and only then executes repository code.

Its scheduling contract is:

```yaml
on:
  workflow_dispatch:

permissions: {}

jobs:
  smoke:
    if: github.repository == 'Luminous-Dynamics/symthaea' && github.ref == 'refs/heads/main'
    runs-on: [symthaea-trusted-cpu-v1]
```

The workflow:

1. fetches the exact protected `main` commit without `actions/checkout`; it does not rely on `GITHUB_TOKEN` for checkout and declares `permissions: {}`;
2. verifies `HEAD == GITHUB_SHA`;
3. records source, nixpkgs, Rust, and pin-digest provenance using Git/Nix/coreutils only;
4. executes the eval-only runner trust-policy and trusted-routing tests;
5. places Cargo home/target state under `RUNNER_TEMP`;
6. enters `nix/ci-rust-shell.nix` once and verifies Rust/Cargo/rustfmt/Clippy;
7. runs `cargo check --locked -p symthaea-psych-bench --lib` so dependency resolution cannot update the locked graph;
8. verifies `HEAD` is unchanged and the source workspace has no tracked, staged, untracked, or ignored mutations.

The workflow declares `permissions: {}` and uses no third-party Actions.

Do not add a pull-request trigger. Any future PR fallback needs a separate threat-model review and must not execute untrusted fork or branch code on this host.

## Frozen recovery targets

Every unmerged recovery target is a separate reviewed workflow with **no ref/SHA inputs**. The trusted workflow fixes the target commit/tree/base, exact path allowlist, and other target-specific provenance before executing target code.

The #186 recovery harness additionally pins the exact base/target `Cargo.lock` blobs and rejects any dependency-resolution delta other than its frozen `9 additions / 0 deletions` lockfile change. For #186 that lock change is only the workspace package stanza for `symthaea-ai-assurance`; it does not alter third-party versions, checksums, registry sources, or transitive dependency selection.

A recovery PASS is scoped to the exact frozen tree and correctness/reproducibility gate. It does not imply hosted-runner equivalence, full repository CI, performance equivalence, or qualification of later stacked PRs.

## Host verification

After `nixos-rebuild switch`:

```bash
systemctl status github-runner-symthaea-validation.service
journalctl -u github-runner-symthaea-validation.service -b --no-pager
```

Then verify in repository settings that the runner is online and carries exactly `symthaea-trusted-cpu-v1` as its custom scheduling capability.

Do not route scientific validation to it until the protected-root requirement and manual smoke both succeed.

## GPU runners are separate

Do not reuse this module for GPU workloads.

A GPU runner needs a separate design because access to `/dev/nvidia*`, render devices, CUDA driver state, or related host resources changes the sandbox threat model. GPU capability should use a distinct routing label and be reviewed/tested independently.

## Scientific performance boundary

This trusted CPU runner is initially a **correctness substrate**: formatting, deterministic unit tests, compile checks, schema/validator checks, and similar gates may execute here once the smoke has passed.

Do not treat its timing, throughput, RSS, or resource measurements as interchangeable with GitHub-hosted or another machine. A6 explicitly binds measurements to runtime context. Any claim-bearing performance comparison on this substrate must preregister and bind the exact hardware/runtime context and use appropriately matched conditions. Changing runner substrate is not evidence of a performance change by itself.

## Acceptance criteria

The trusted CPU fallback is operational only when all are true:

- issue #330 is satisfied and GitHub readback proves `main` has effective trusted-root branch/ruleset protection;
- ordinary direct pushes/force pushes cannot silently rewrite the trusted harness outside the reviewed promotion path;
- the eval-only policy test succeeds against the repository-pinned nixpkgs revision;
- the trusted-routing eval proves only the explicitly reviewed manual/main-only/tokenless workflows can match `symthaea-trusted-cpu-v1`;
- a `/nix/store` token path is rejected at Nix evaluation time;
- the v2 access token is a fine-grained PAT restricted to `Luminous-Dynamics/symthaea` with only repository `Administration: write`;
- the token runtime file is root-owned, `0400` or `0600`, non-empty, no larger than 4096 bytes, contains exactly one printable non-whitespace token with no newline, and is outside the Nix store;
- the service runtime preflight independently enforces those credential-file properties before runner registration;
- the runner service is healthy after reboot;
- GitHub shows the runner online with the exact `symthaea-trusted-cpu-v1` capability;
- the trusted manual smoke completes successfully from protected `main`;
- the smoke verifies its checked-out `HEAD` equals `GITHUB_SHA` and leaves the entire source workspace clean;
- the minimal Rust shell resolves from pinned inputs and verifies Rust/Cargo/rustfmt/Clippy before `cargo check --locked` accepts the committed dependency graph;
- the job begins from a clean ephemeral work/state lifecycle;
- the runner binary remains Nix-controlled with self-update disabled;
- no repository/workflow contains the access token and job code cannot read the external token path;
- no generic, fork, untrusted-branch, or GPU workflow can match the trusted CPU runner;
- the host/VM contains no unrelated secrets or privileged host-control sockets;
- the host network segment exposes no sensitive LAN/control-plane services to target code;
- disabling the module removes the host from active capacity without changing scientific evidence.

## Relationship to SYM-ARCH

Runner substrate is operational provenance, not experimental outcome.

A SYM-ARCH PR still needs the same exact-head formatting, tests, compilation, benchmark-validity, statistics, and evidence gates regardless of whether execution occurs on GitHub-hosted capacity or this trusted CPU fallback.
