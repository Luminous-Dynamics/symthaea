# Trusted CPU Runner Bootstrap Sequence

## Purpose

This document closes the bootstrap and promotion boundary between the queue-neutral trusted-runner recovery branch and the first GitHub-executed trusted-runner smoke.

The trusted CPU fallback exists only to recover **correctness and reproducibility execution capacity** when GitHub-hosted runner assignment is severely delayed. It does not weaken scientific gates, create performance evidence, or turn queued workflows into PASS/FAIL evidence.

The trust transition is intentionally:

```text
operator-authorized recovery commit
        ↓
host-side bootstrap validation
        ↓
bootstrap.v5 + independently retained SHA-256
        ↓
exact promotion onto main
        ↓
host-side promotion verification
        ↓
promotion.v1 + independently retained SHA-256
        ↓
main-only manual GitHub smoke
        ↓
trusted CPU correctness capability
```

## Why bootstrap is multi-stage

GitHub `workflow_dispatch` only receives events when the workflow file exists on the repository default branch. A manual smoke staged only on `ci/nixos-ephemeral-runner-v1` therefore cannot bootstrap itself onto `main`.

Do not solve that circular dependency by adding automatic pull-request execution or by allowing arbitrary branch code to schedule onto the trusted host.

Bootstrap, promotion, and smoke are separate authority transitions:

```text
bootstrap correctness
    != promotion correctness
    != runner qualification
    != scientific qualification
```

## Stage A — host-side bootstrap validation

Run Stage A directly on the isolated NixOS host that will become the trusted CPU runner. The host should contain no unrelated credentials, personal data, production mounts, privileged container socket, or sensitive LAN access.

Obtain the exact recovery commit from the reviewed operator record (issue #75 or another out-of-band authorization record). **Do not infer authorization from the recovery branch head.**

```bash
export SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD='<reviewed 40-hex recovery commit>'
test "${#SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD}" -eq 40

git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
git fetch origin ci/nixos-ephemeral-runner-v1
git checkout --detach origin/ci/nixos-ephemeral-runner-v1

test "$(git rev-parse HEAD)" = "$SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD"
bash nix/ci/validate-trusted-runner-bootstrap.sh
```

The explicit `test` is only an operator convenience. The validator independently requires `SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD` and requires:

```text
operator-authorized head
        = checked-out head
        = freshly fetched public recovery head before validation
        = freshly fetched public recovery head after validation
```

If any equality fails, stop. Never update the expected SHA merely to match the branch. Review and explicitly authorize the new recovery generation first.

The validator also fails closed unless:

- `origin` is the canonical public HTTPS Symthaea repository;
- the working tree is pristine, including ignored and untracked files;
- current public `main` is an ancestor of the recovery generation;
- the complete `main...recovery` diff is exactly the reviewed runner/recovery allowlist;
- the allowlist includes both bootstrap and promotion validators plus the persistent-host lifecycle contract;
- runner-policy Nix evaluation passes;
- trusted-routing Nix evaluation passes and the trusted CPU consumer set is unchanged;
- the pinned minimal Rust shell accepts locked metadata and `symthaea-psych-bench` library compilation;
- the source commit/tree remain immutable during validation;
- both public refs remain unchanged for the complete Stage-A interval.

### bootstrap.v5

On success Stage A emits a unique `symthaea.trusted-runner.bootstrap.v5` manifest and prints its SHA-256.

Retain the exact manifest **outside the repository** and independently record its printed SHA-256 before proceeding. The local path is not authority; the independently recorded hash authenticates the retained manifest bytes later.

`bootstrap.v5` binds, among other provenance:

- exact operator-authorized recovery commit;
- exact recovery commit/tree validated;
- exact pre-promotion public `main` commit/tree;
- exact recovery diff-path SHA-256;
- required authorized ancestor for promotion;
- exact expected post-promotion `main` tree;
- runner-module blob;
- routing-policy blob;
- main-only smoke-workflow blob;
- bootstrap-validator blob;
- **promotion-verifier blob**;
- persistent-host lifecycle-contract blob;
- pinned nixpkgs revision;
- repository Rust channel;
- host Nix system and Nix implementation version;
- `flake.lock` and `rust-toolchain.toml` SHA-256 values;
- PASS markers for authorization, runner policy, routing policy, locked Rust validation, and ref stability.

The authorization record plus the manifest hash plus the manifest bytes jointly establish intent and execution. None substitutes for the others.

Stage A remains runner-bootstrap evidence only.

## Stage B — provision the isolated runner host

Only after Stage A passes:

1. Read `docs/operations/TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md` and record that v1 is a **persistent hardened host with an ephemeral one-job runner registration/process**, not a fresh machine per job.
2. Materialize the repository-scoped access token outside the Nix store in a root-owned runtime file.
3. Require token-file mode `0400` or `0600`, exact token bytes with no trailing newline, and no group/other readability.
4. Import the runner module from the exact Stage-A-qualified recovery generation.
5. Enable only `services.symthaea-ci-runner`.
6. Rebuild the isolated host.
7. Configure and test external retention for the lifecycle logs required by the host-lifecycle contract.
8. Verify the runner service is healthy.
9. Verify GitHub advertises exactly the custom `symthaea-trusted-cpu-v1` capability and no default labels.

Do not route correctness work yet. An online runner is not a qualified runner.

If the recovery branch changes after Stage A, Stage A is stale. Review/authorize the new generation and rerun before using changed infrastructure.

## Stage C — land and mechanically verify the exact Stage-A generation

The smoke becomes dispatchable only after its workflow exists on `main`.

The landing surface must remain restricted to:

- trusted-runner Nix module and aggregate import;
- eval-only runner/routing/lifecycle tests;
- minimal pinned CPU Rust shell;
- bootstrap and promotion validators;
- persistent-host lifecycle contract and operator documentation;
- reviewed `workflow_dispatch`-only trusted recovery workflows.

Do not bundle application, scientific-result, RCA policy, root Cargo/toolchain, build-script, or performance changes.

Before landing, current public `main` must still equal the `main_head` recorded by Stage A. If it moved, rerun Stage A.

The landing must preserve the exact authorized recovery commit in ancestry. **Do not squash or cherry-pick.** A fast-forward or merge commit is acceptable only if the authorized recovery commit remains an ancestor and the resulting tree is byte-identical to the Stage-A-qualified tree.

### Promotion verification

After landing, make a fresh pristine detached checkout of current public `main`. Supply both the retained Stage-A manifest path and its independently recorded SHA-256:

```bash
git fetch origin main ci/nixos-ephemeral-runner-v1
git checkout --detach origin/main

git diff --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)"

export SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_PATH='/absolute/path/to/retained/bootstrap.v5'
export SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_SHA256='<independently recorded 64-hex SHA-256>'

bash nix/ci/validate-trusted-runner-promotion.sh
```

The promotion verifier first authenticates the exact manifest bytes against the independently recorded SHA-256. It never `source`s the manifest; required keys are parsed as data and must occur exactly once.

It then requires all of the following:

```text
local HEAD/tree
    = exact current public main HEAD/tree

local promotion-verifier blob
    = Stage-A-qualified promotion-verifier blob

public recovery head
    = Stage-A operator-authorized recovery head

Stage-A main head
    ∈ ancestors(current public main)

authorized recovery head
    ∈ ancestors(current public main)

current public main tree
    = Stage-A promotion_expected_main_tree

SHA256(diff paths: Stage-A main → current main)
    = Stage-A recovery_diff_paths_sha256

promotion-critical blobs on current main
    = exact Stage-A-qualified blobs
```

Promotion-critical blobs include the runner module, routing policy, smoke workflow, bootstrap validator, promotion verifier, and persistent-host lifecycle contract.

The verifier fetches public `main` and recovery refs again at the end and refuses PASS if either moved during Stage C. It also rechecks local HEAD/tree and repository cleanliness.

### promotion.v1

On success the verifier emits a unique `symthaea.trusted-runner.promotion.v1` manifest and its SHA-256.

Retain both before Stage D. The promotion manifest binds:

- exact authenticated `bootstrap.v5` SHA-256;
- authorized recovery head;
- promoted public `main` head/tree;
- exact promotion-verifier blob;
- PASS for ancestry preservation;
- PASS for tree identity;
- PASS for diff-surface identity;
- PASS for critical artifact-blob identity;
- PASS for exact local promoted checkout;
- PASS for ref stability.

This creates the executable promotion theorem:

```text
operator-authorized recovery R
        +
bootstrap.v5 PASS for tree T
        +
independently authenticated bootstrap manifest
        +
landing preserves R in ancestry
        +
public main tree == T
        +
critical blobs == Stage-A blobs
        +
promotion.v1 PASS
        =
eligible to attempt Stage-D smoke
```

Stage-C PASS is still not runner qualification.

## Stage D — main-only GitHub smoke

Only after `promotion.v1` passes, dispatch:

```text
Self-hosted NixOS Runner Smoke
```

against `main` only.

The smoke must establish in one GitHub-issued run:

- canonical repository and `refs/heads/main`;
- unique trusted CPU routing capability;
- exact `GITHUB_SHA` detached checkout;
- no `GITHUB_TOKEN` permissions;
- exact runner/routing policy evaluation;
- pinned Nix/Rust provenance;
- locked Rust compilation through the minimal shell;
- clean source tree after execution.

Retain the external runner/system lifecycle evidence required by `TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md`. GitHub job success by itself does not prove clean persistent-host state for the next job.

If smoke fails, do not route recovery workloads. Repair only the observed defect. If trusted infrastructure changes, restart at Stage A with a newly reviewed authorization.

## Stage E — correctness recovery only

After the main-only smoke is green, reviewed manual recovery workflows may use `symthaea-trusted-cpu-v1` for deterministic correctness/reproducibility gates.

Recovery harnesses should:

- accept no arbitrary shell command or caller-selected SHA;
- bind exact target commit/tree/base identities;
- fetch trusted harness and unmerged target into separate directories;
- freeze the security-relevant target diff surface;
- reproduce the target's existing semantic/static correctness contract;
- preserve its explicitly bound compiler/toolchain;
- run only fixed formatting, locked metadata, tests, Clippy, schema or validator gates;
- emit source/toolchain/runner provenance;
- verify source and harness immutability.

Trusted-CPU correctness evidence is not performance equivalence.

For RCA recovery, continue **one prerequisite at a time**. The current first target is exact PR #578 canonical evidence-lineage generation. Do not route #531 or downstream work until #578 executes successfully.

## Failure semantics

At every stage:

```text
queued or unexecuted != PASS
cancelled for supersession != PASS or FAIL
bootstrap PASS != promotion PASS
promotion PASS != runner smoke PASS
runner smoke PASS != scientific gate PASS
trusted CPU correctness PASS != performance equivalence
stale manifest != current authority
current branch head != operator authorization
manifest contents != authenticated manifest without its recorded hash
runner process ephemeral != host ephemeral
content-similar landing != exact qualified tree
squash/cherry-pick != preserved qualified ancestry
```

The fallback exists to restore executable evidence, not to weaken what counts as evidence.
