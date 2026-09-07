# Trusted CPU Runner Bootstrap Sequence

## Purpose

This document closes the bootstrap boundary between the queue-neutral recovery branch and the first GitHub-executed trusted-runner smoke.

The trusted CPU fallback exists only to recover **correctness and reproducibility execution capacity** when GitHub-hosted runner assignment is severely delayed. It does not weaken scientific gates, create performance evidence, or turn queued workflows into PASS/FAIL evidence.

## Why bootstrap is two-stage

GitHub `workflow_dispatch` only receives events when the workflow file exists on the repository default branch. Therefore a manual smoke workflow staged only on `ci/nixos-ephemeral-runner-v1` cannot itself bootstrap the branch onto `main`.

Do not solve that circular dependency by adding an automatic pull-request trigger or by letting arbitrary branch code schedule onto the trusted host.

The trust transition is intentionally:

```text
operator-authorized recovery commit
        ↓
current published recovery branch
        ↓
host-side bootstrap validation
        ↓
exact promotion onto main
        ↓
main-only manual GitHub smoke
        ↓
trusted CPU correctness capability
```

## Stage A — host-side bootstrap validation

Perform this stage on the isolated/disposable NixOS host that will become the trusted CPU runner. The host should contain no unrelated credentials, personal data, production mounts, privileged container socket, or sensitive LAN access.

Before cloning, obtain the exact recovery commit that the operator intends to authorize from the repository's runner-capacity record (issue #75 or another reviewed out-of-band record). **Do not infer authorization from whatever commit the recovery branch currently points to.** Export that exact reviewed SHA locally:

```bash
export SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD='<reviewed 40-hex recovery commit>'
test "${#SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD}" -eq 40
```

Clone the canonical public repository, detach at the current published recovery branch, prove that the checked-out commit is the separately authorized commit, and invoke the canonical validator explicitly through Bash:

```bash
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
git fetch origin ci/nixos-ephemeral-runner-v1
git checkout --detach origin/ci/nixos-ephemeral-runner-v1

test "$(git rev-parse HEAD)" = "$SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD"
bash nix/ci/validate-trusted-runner-bootstrap.sh
```

The explicit shell check above is an operator convenience, not the trust boundary by itself. The validator independently requires `SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD`, rejects missing or malformed values, and rechecks that exact authorization against the checkout plus both pre- and post-validation public recovery refs.

This gives Stage A three distinct equalities rather than one circular notion of "current":

```text
operator-authorized head
        = checked-out head
        = freshly fetched public recovery head
```

If any equality fails, stop. Do not update the expected value merely to match the branch; review the new recovery generation first and record a new operator authorization.

Do not substitute a locally edited copy of the validator. The script requires a pristine checkout whose `HEAD` exactly matches both the explicit operator authorization and the freshly fetched published recovery head.

The validator fails closed unless all of the following hold in one validation interval:

- `SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD` is a non-empty 40-hex commit SHA;
- `origin` is the canonical public HTTPS Symthaea repository;
- the working tree is pristine, including untracked and ignored files;
- the checked-out commit exactly equals the operator-authorized recovery head;
- the checked-out commit exactly equals the current published `ci/nixos-ephemeral-runner-v1` head;
- current public `main` is an ancestor of that recovery head;
- the complete `main...recovery` diff is exactly the reviewed trusted-runner/recovery infrastructure allowlist, including the host-lifecycle contract;
- the runner-policy Nix evaluation succeeds;
- the trusted-routing Nix evaluation succeeds and the trusted CPU capability consumer set is unchanged;
- the minimal pinned Rust shell resolves and accepts locked metadata plus `symthaea-psych-bench` library compilation;
- the source commit/tree and repository cleanliness are unchanged after evaluation;
- a second public-ref refresh produces the same `main` and recovery SHAs seen at the beginning of validation;
- the final published recovery head still equals the exact operator-authorized SHA.

If either public ref moves during Stage A, the validator refuses PASS. Detach at the new recovery head, obtain a fresh explicit operator authorization for that reviewed generation, and rerun rather than carrying stale bootstrap evidence forward.

On success the validator emits a `symthaea.trusted-runner.bootstrap.v4` manifest in a unique `mktemp` path under `/tmp`. The manifest contains no credential. Record both the manifest contents and its printed SHA-256 before proceeding.

The manifest binds, among other provenance:

- exact operator-authorized recovery commit;
- exact recovery commit and source tree actually validated;
- exact `main` commit and tree observed for the complete validation interval;
- the authorized recovery commit as the required ancestor of the promoted `main` generation;
- the exact Stage-A-validated recovery tree as the only acceptable post-promotion `main` tree;
- SHA-256 of the exact reviewed diff-path set;
- Git blob identities for the runner module, routing policy, main-only smoke workflow, bootstrap validator, and persistent-host lifecycle contract;
- pinned nixpkgs revision and repository Rust channel;
- host Nix system and Nix implementation version used for Stage A;
- `flake.lock` and `rust-toolchain.toml` SHA-256 values;
- PASS state for explicit operator-authorization verification, runner policy, routing policy, locked Rust validation, and ref-stability checks.

The operator authorization record plus this manifest jointly establish what was intended to be validated and what was actually validated. Neither one substitutes for the other.

Stage A is bootstrap evidence for the runner infrastructure only. It is **not** a substitute for any hosted or self-hosted scientific qualification workflow.

## Stage B — provision the isolated runner host

Only after Stage A succeeds:

1. Read `docs/operations/TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md` and record that v1 is a **persistent hardened host with an ephemeral runner registration/process**, not a fresh machine per job.
2. Materialize the repository-scoped access token outside the Nix store in a root-owned runtime file.
3. Verify the token file is `0400` or `0600`, contains exactly the token with no trailing newline, and is not readable by group/other users.
4. Import `nix/modules/github-actions-runner.nix` or the aggregate Symthaea NixOS module set from the exact Stage-A-qualified recovery generation.
5. Enable only `services.symthaea-ci-runner`.
6. Rebuild the isolated host.
7. Configure and test external retention for the runner/system logs required by the host-lifecycle contract before Stage D/E recovery use.
8. Verify the runner service is healthy.
9. Verify GitHub shows exactly the custom capability `symthaea-trusted-cpu-v1`; default labels must remain absent.

Do not route any correctness job to the host yet. An online runner is not a qualified runner, and an ephemeral runner registration is not an ephemeral host.

If the recovery branch changes after Stage A, rerun Stage A against the newly reviewed and explicitly operator-authorized head before using the changed module or workflow content.

## Stage C — land only the exact Stage-A-qualified generation

The GitHub smoke becomes dispatchable only after its workflow exists on `main`.

The bootstrap merge/review surface must remain restricted to trusted-runner infrastructure:

- the NixOS runner module and aggregate import;
- eval-only runner/routing and lifecycle tests;
- the minimal pinned CPU Rust shell;
- the host-side bootstrap validator;
- the persistent-host lifecycle contract and other operator documentation;
- reviewed `workflow_dispatch`-only trusted recovery workflows.

Do not bundle application, scientific-result, RCA policy, root Cargo/toolchain, build-script, or performance changes into this bootstrap transition.

Before landing, the current public `main` head must still equal the `main_head` recorded in the Stage-A manifest. If `main` moved after Stage A, the manifest is stale: rerun Stage A against a freshly reviewed and explicitly authorized recovery generation.

The promotion operation must preserve the exact authorized recovery commit in `main` ancestry. **Do not squash or cherry-pick the recovery tranche.** A fast-forward or merge commit is acceptable only if the authorized recovery commit remains an ancestor of the resulting `main` generation.

After landing, fetch public `main` again and verify both promotion predicates from the Stage-A manifest:

```text
promotion_required_ancestor
        is an ancestor of new main

AND

new main tree
        == promotion_expected_main_tree
```

The second condition is byte-level Git tree identity, not merely the same list of paths or a visually similar diff. It ensures the infrastructure that becomes dispatchable on `main` is exactly the tree Stage A validated.

A review/merge tool may create a new merge commit identity, but it may not alter the promoted tree or discard the authorized recovery commit from ancestry. If either condition fails, do not dispatch the smoke; restore/review the landing and rerun Stage A if the recovery generation changes.

Also compare the proposed merge surface with the exact diff-path set bound by the Stage-A manifest. If the path set or any reviewed artifact blob changed, Stage A is stale and must be rerun after explicit authorization of the changed recovery generation.

Re-check that every `symthaea-trusted-cpu-v1` consumer is explicitly allowlisted by `nix/tests/eval-trusted-runner-routing.nix` and that no consumer has `push`, `pull_request`, or `schedule` triggers.

This creates an explicit promotion theorem:

```text
Stage-A PASS for tree T
        +
landing preserves authorized recovery ancestry
        +
post-landing main tree == T
        =
eligible to attempt Stage-D smoke
```

Stage-C success is still not runner qualification.

## Stage D — main-only GitHub smoke

Once the reviewed infrastructure exists on `main` and the Stage-C ancestry/tree predicates pass, dispatch:

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

The operator must also retain the external runner/system lifecycle evidence required by `TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md`; GitHub job success alone does not establish clean host state for a subsequent job.

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
current branch head != operator authorization
runner process ephemeral != host ephemeral
content-similar landing != exact qualified tree
squash/cherry-pick of qualified recovery != preserved promotion ancestry
```

The fallback exists to restore executable evidence, not to weaken what counts as evidence.
