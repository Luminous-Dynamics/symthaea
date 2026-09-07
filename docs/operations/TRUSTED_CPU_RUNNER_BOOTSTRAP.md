# Trusted CPU Runner Bootstrap Sequence

## Purpose

This document closes the bootstrap boundary between the queue-neutral recovery branch and the first GitHub-executed trusted-runner smoke.

The trusted CPU fallback exists only to recover **correctness and reproducibility execution capacity** when GitHub-hosted runner assignment is severely delayed. It does not weaken scientific gates, create performance evidence, or turn queued workflows into PASS/FAIL evidence.

## Why bootstrap is two-stage

GitHub `workflow_dispatch` only receives events when the workflow file exists on the repository default branch. Therefore a manual smoke workflow staged only on `ci/nixos-ephemeral-runner-v1` cannot itself bootstrap the branch onto `main`.

Do not solve that circular dependency by adding an automatic pull-request trigger or by letting arbitrary branch code schedule onto the trusted host.

The trust transition is intentionally:

```text
queue-neutral recovery branch
        ↓
host-side bootstrap validation
        ↓
reviewed runner infrastructure on main
        ↓
main-only manual GitHub smoke
        ↓
trusted CPU correctness capability
```

## Stage A — host-side bootstrap validation

Perform this stage on the isolated/disposable NixOS host that will become the trusted CPU runner. The host should contain no unrelated credentials, personal data, production mounts, privileged container socket, or sensitive LAN access.

Clone the canonical public repository, detach at the current published recovery branch, and invoke the canonical validator explicitly through Bash:

```bash
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
git fetch origin ci/nixos-ephemeral-runner-v1
git checkout --detach origin/ci/nixos-ephemeral-runner-v1
bash nix/ci/validate-trusted-runner-bootstrap.sh
```

Do not substitute a locally edited copy of the validator. The script requires a pristine checkout whose `HEAD` exactly matches the freshly fetched published recovery head.

The validator fails closed unless all of the following hold in one validation interval:

- `origin` is the canonical public HTTPS Symthaea repository;
- the working tree is pristine, including untracked and ignored files;
- the checked-out commit exactly equals the current published `ci/nixos-ephemeral-runner-v1` head;
- current public `main` is an ancestor of that recovery head;
- the complete `main...recovery` diff is exactly the reviewed trusted-runner/recovery infrastructure allowlist;
- the runner-policy Nix evaluation succeeds;
- the trusted-routing Nix evaluation succeeds and the trusted CPU capability consumer set is unchanged;
- the minimal pinned Rust shell resolves and accepts locked metadata plus `symthaea-psych-bench` library compilation;
- the source commit/tree and repository cleanliness are unchanged after evaluation;
- a second public-ref refresh produces the same `main` and recovery SHAs seen at the beginning of validation.

If either public ref moves during Stage A, the validator refuses PASS. Detach at the new recovery head and rerun rather than carrying stale bootstrap evidence forward.

On success the validator emits a `symthaea.trusted-runner.bootstrap.v2` manifest in a unique `mktemp` path under `/tmp`. The manifest contains no credential. Record both the manifest contents and its printed SHA-256 before proceeding.

The manifest binds, among other provenance:

- exact recovery commit and source tree;
- exact `main` commit observed for the complete validation interval;
- SHA-256 of the exact reviewed diff-path set;
- Git blob identities for the runner module, routing policy, main-only smoke workflow, and bootstrap validator;
- pinned nixpkgs revision and repository Rust channel;
- `flake.lock` and `rust-toolchain.toml` SHA-256 values;
- PASS state for runner policy, routing policy, locked Rust validation, and ref-stability checks.

Stage A is bootstrap evidence for the runner infrastructure only. It is **not** a substitute for any hosted or self-hosted scientific qualification workflow.

## Stage B — provision the isolated runner host

Only after Stage A succeeds:

1. Materialize the repository-scoped access token outside the Nix store in a root-owned runtime file.
2. Verify the token file is `0400` or `0600`, contains exactly the token with no trailing newline, and is not readable by group/other users.
3. Import `nix/modules/github-actions-runner.nix` or the aggregate Symthaea NixOS module set from the exact Stage-A-qualified recovery generation.
4. Enable only `services.symthaea-ci-runner`.
5. Rebuild the isolated host.
6. Verify the runner service is healthy.
7. Verify GitHub shows exactly the custom capability `symthaea-trusted-cpu-v1`; default labels must remain absent.

Do not route any correctness job to the host yet. An online runner is not a qualified runner.

If the recovery branch changes after Stage A, rerun Stage A against the new exact head before using the changed module or workflow content.

## Stage C — land only the reviewed runner infrastructure

The GitHub smoke becomes dispatchable only after its workflow exists on `main`.

The bootstrap merge/review surface must remain restricted to trusted-runner infrastructure:

- the NixOS runner module and aggregate import;
- eval-only runner/routing tests;
- the minimal pinned CPU Rust shell;
- the host-side bootstrap validator;
- operator documentation;
- reviewed `workflow_dispatch`-only trusted recovery workflows.

Do not bundle application, scientific-result, RCA policy, root Cargo/toolchain, build-script, or performance changes into this bootstrap transition.

Before landing, compare the proposed merge surface with the exact diff-path set bound by the Stage A manifest. If the path set or any reviewed artifact blob changed, Stage A is stale and must be rerun.

Also re-check that every `symthaea-trusted-cpu-v1` consumer is explicitly allowlisted by `nix/tests/eval-trusted-runner-routing.nix` and that no consumer has `push`, `pull_request`, or `schedule` triggers.

## Stage D — main-only GitHub smoke

Once the reviewed infrastructure exists on `main`, dispatch:

```text
Self-hosted NixOS Runner Smoke
```

against `main` only.

The smoke must establish all of the following in one GitHub-issued run:

- canonical repository and `refs/heads/main`;
- unique trusted CPU routing capability;
- exact `GITHUB_SHA` detached checkout;
- no `GITHUB_TOKEN` permissions;
- exact runner/routing policy evaluations;
- pinned Nix/Rust provenance;
- locked Rust compilation through the minimal Nix shell;
- clean source tree after execution.

If the smoke fails, do not route recovery workloads. Disable or leave idle the runner, repair only the observed defect, and repeat Stage A if the trusted infrastructure generation changes, then repeat the smoke.

## Stage E — correctness recovery only

After the main-only smoke is green, reviewed manual recovery workflows may use `symthaea-trusted-cpu-v1` for deterministic correctness/reproducibility gates.

Prefer one-purpose exact-tree recovery harnesses that:

- accept no arbitrary shell command;
- use hard-coded target commit/tree/base identities or another explicitly reviewed target authorization mechanism;
- fetch the trusted harness from `main` and the target source into separate directories;
- freeze the target's security-relevant diff surface;
- reproduce the target's static/semantic correctness contract rather than inventing a weaker substitute;
- preserve an explicitly bound toolchain when the original hosted gate fixes one;
- run only fixed formatting, locked metadata, test, Clippy, schema, validator, or similar CPU correctness checks;
- emit source/toolchain/runner provenance;
- verify source and harness immutability after execution.

Do not treat a trusted-CPU PASS as interchangeable performance evidence with GitHub-hosted hardware or any other machine.

For RCA recovery, proceed one prerequisite at a time. The current first recovery target is the exact frozen canonical-lineage generation for PR #578. Do not fan out recovery capacity to downstream RCA prerequisites until the earliest required target actually executes successfully.

## Failure semantics

At every stage:

```text
queued or unexecuted != PASS
cancelled for supersession != PASS or FAIL
bootstrap policy PASS != scientific gate PASS
trusted CPU correctness PASS != performance equivalence
stale bootstrap manifest != current infrastructure authority
```

The fallback exists to restore executable evidence, not to weaken what counts as evidence.
