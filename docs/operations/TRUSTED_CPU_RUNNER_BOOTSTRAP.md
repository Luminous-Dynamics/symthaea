# Trusted CPU Runner Bootstrap and Recovery Eligibility

## Purpose

This document defines the fail-closed trust transition from the queue-neutral recovery branch to a trusted-CPU correctness run. It does **not** create scientific evidence, performance equivalence, or authority from queued/cancelled work.

The complete v1 recovery sequence is:

```text
explicit operator authorization
        ↓
Stage A: bootstrap.v6 PASS
        ↓
Stage B: provision hardened persistent host / ephemeral runner process
        ↓
Stage C: exact promotion + promotion.v2 PASS
        ↓
Stage D: main-only GitHub smoke + smoke.v1 PASS
        ↓
Stage E: promotion/smoke join + recovery-eligibility.v1 PASS
        ↓
Stage F: one exact frozen correctness recovery target (#578 first)
```

Each arrow is a separate evidence transition. None may be inferred from the previous stage.

## Stage A — host-side bootstrap validation

Run on the isolated NixOS host that will become the trusted CPU runner. Obtain the exact reviewed recovery SHA from issue #75 or another out-of-band authorization record. Do not infer authorization from the branch head.

```bash
export SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD='<reviewed 40-hex recovery commit>'

git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
git fetch origin ci/nixos-ephemeral-runner-v1
git checkout --detach origin/ci/nixos-ephemeral-runner-v1

test "$(git rev-parse HEAD)" = "$SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD"
bash nix/ci/validate-trusted-runner-bootstrap.sh
```

The validator independently requires:

- explicit 40-hex operator authorization;
- canonical public HTTPS origin;
- pristine tracked/untracked/ignored source tree;
- authorized head = checked-out head = freshly fetched recovery head;
- current public `main` is an ancestor of recovery;
- exact reviewed recovery path allowlist;
- exact runner/routing/smoke/bootstrap/promotion/recovery-eligibility/lifecycle artifact identities;
- runner-policy and routing-policy Nix evaluation;
- pinned minimal Rust environment with locked Cargo metadata and compilation;
- unchanged source commit/tree;
- unchanged public `main` and recovery refs across the complete validation interval.

A successful run emits:

```text
schema=symthaea.trusted-runner.bootstrap.v6
```

and a SHA-256 of the exact manifest bytes. Retain **both** the manifest and printed hash outside the repository.

`bootstrap.v6` binds:

- operator-authorized recovery head and exact source tree;
- exact pre-promotion `main` head/tree;
- exact promoted-tree requirement and required recovery ancestor;
- reviewed path-set SHA-256;
- runner module, routing policy and smoke workflow blobs;
- bootstrap, promotion and recovery-eligibility verifier blobs;
- host-lifecycle contract blob;
- nixpkgs/Rust/lock/toolchain provenance;
- host Nix system/version;
- PASS states for authorization, policy evaluation, locked Rust validation and ref stability.

If either public ref moves or any reviewed artifact changes, Stage A is stale. Review/authorize the new recovery generation and rerun.

## Stage B — provision the host

Only after `bootstrap.v6 PASS`:

1. Read `TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md`.
2. Treat v1 as a **persistent hardened host with an ephemeral runner registration/process**, not a fresh host per job.
3. Put the repository-scoped runner credential outside the Nix store in a root-owned runtime file (`0400` or `0600`).
4. Import the exact Stage-A-qualified runner module and enable only `services.symthaea-ci-runner`.
5. Rebuild the isolated host.
6. Configure external retention for required runner/system lifecycle logs.
7. Verify GitHub shows only capability `symthaea-trusted-cpu-v1`; default labels remain absent.

An online runner is not a qualified runner. Do not dispatch correctness recovery yet.

## Stage C — exact promotion onto `main`

The manual GitHub smoke is dispatchable only after its workflow exists on the default branch.

Before landing, current public `main` must still equal the `main_head` recorded by Stage A. Promotion must preserve the operator-authorized recovery commit in ancestry. **Do not squash or cherry-pick.** The resulting `main` tree must be byte-identical to `promotion_expected_main_tree` from `bootstrap.v6`.

After landing, detach a pristine checkout at current public `main` and verify promotion using the exact retained Stage-A evidence:

```bash
export SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_PATH='/path/to/bootstrap.v6'
export SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_SHA256='<recorded Stage-A SHA-256>'

bash nix/ci/validate-trusted-runner-promotion.sh
```

The verifier first authenticates the exact bootstrap-manifest bytes. It then requires:

- local pristine checkout = current public `main` head/tree;
- local promotion-verifier blob = Stage-A-bound verifier blob;
- recovery branch still equals the authorized recovery head;
- Stage-A `main` remains in promoted ancestry;
- authorized recovery head remains in promoted ancestry;
- promoted tree = exact Stage-A-qualified recovery tree;
- promoted diff-path digest = Stage-A digest;
- runner/routing/smoke/bootstrap/promotion/recovery-eligibility/lifecycle blobs = Stage-A-qualified blobs;
- public refs stable through promotion verification.

Success emits:

```text
schema=symthaea.trusted-runner.promotion.v2
```

plus its SHA-256. Retain both exact bytes and hash.

`promotion.v2 PASS` means only **Stage-D smoke is eligible**. It does not qualify the runner.

## Stage D — main-only GitHub smoke

Dispatch `Self-hosted NixOS Runner Smoke` manually against `main` only.

The smoke has:

- `workflow_dispatch` only;
- canonical repository and `refs/heads/main` guard;
- unique `symthaea-trusted-cpu-v1` routing capability;
- `permissions: {}`;
- no checkout/action dependency;
- exact `GITHUB_SHA` detached checkout;
- runner/routing Nix evaluation;
- pinned locked Rust correctness check;
- source-tree immutability check.

Only after all checks succeed, the final step emits:

```text
schema=symthaea.trusted-runner.smoke.v1
```

and its SHA-256. Retain the exact manifest bytes and hash from the successful GitHub run. Also record the GitHub run ID/attempt from the manifest and independently inspect that run in GitHub.

`smoke.v1` binds the exact GitHub `main` head/tree, run identity, runner identity, smoke workflow, runner/routing blobs, Nix/Rust provenance and PASS states. It is correctness/reproducibility smoke evidence only.

If the smoke fails, do not route recovery workloads. Repair the observed substrate defect; if trusted infrastructure changes, restart at Stage A.

## Stage E — exact promotion/smoke join

Before any recovery workflow is dispatched, detach a pristine checkout at **current public `main`** and provide the exact retained promotion and smoke manifests plus their independently recorded hashes:

```bash
export SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_PATH='/path/to/promotion.v2'
export SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_SHA256='<recorded promotion SHA-256>'
export SYMTHAEA_TRUSTED_SMOKE_MANIFEST_PATH='/path/to/smoke.v1'
export SYMTHAEA_TRUSTED_SMOKE_MANIFEST_SHA256='<recorded smoke SHA-256>'

bash nix/ci/validate-trusted-runner-recovery-eligibility.sh
```

The Stage-E verifier requires:

- exact SHA-256 match for both evidence files before parsing;
- `promotion.v2 PASS` and `smoke.v1 PASS` schemas;
- same promoted/smoked `main` head and tree;
- same smoke-workflow, runner-module and routing-policy blobs;
- current public `main` still equals the smoke-qualified head/tree;
- recovery branch still equals the authorized recovery head;
- pristine local checkout equals exact current public `main`;
- local eligibility-verifier blob equals the Stage-A-bound blob;
- both public refs remain stable through the join.

Success emits:

```text
schema=symthaea.trusted-runner.recovery-eligibility.v1
```

and a SHA-256. Retain both.

This creates the final bootstrap theorem:

```text
bootstrap.v6 PASS
    + promotion.v2 PASS
    + smoke.v1 PASS
    + exact promotion/smoke identity join
    + current main/recovery ref stability
    = recovery-eligibility.v1 PASS
```

Even this is **not scientific evidence**. It only authorizes attempting a reviewed trusted-CPU correctness recovery workload.

Any movement of public `main` after the smoke invalidates v1 recovery eligibility and requires a new smoke and Stage-E join. This is intentionally conservative.

## Stage F — one exact correctness recovery target

Only after `recovery-eligibility.v1 PASS` may reviewed manual recovery workflows use `symthaea-trusted-cpu-v1`.

Recovery harnesses must:

- accept no arbitrary command/ref/SHA input;
- hard-code or otherwise explicitly authorize the exact target commit/tree/base;
- keep trusted harness and unmerged target source separate;
- freeze the complete target diff and hosted workflow identities;
- reproduce the hosted static/semantic correctness gate rather than weaken it;
- preserve the hosted toolchain when fixed;
- use locked dependency resolution where compatible;
- emit source/toolchain/runner provenance;
- verify source and harness immutability;
- make no performance-equivalence claim.

Proceed one prerequisite at a time. The first RCA recovery target remains exact PR #578 canonical-lineage generation. Do not prepare or dispatch #531/#555/#582/#585/#588 recovery in parallel.

## Failure semantics

At every stage:

```text
queued or unexecuted != PASS
cancelled for supersession != PASS or FAIL
bootstrap PASS != promotion PASS
promotion PASS != smoke PASS
smoke PASS != recovery eligibility
recovery eligibility != scientific qualification
trusted CPU correctness PASS != performance equivalence
current branch head != operator authorization
runner process ephemeral != host ephemeral
content-similar landing != exact qualified tree
squash/cherry-pick != preserved qualified ancestry
changed main after smoke => smoke/recovery eligibility stale
```

The fallback exists to restore executable evidence, never to weaken what counts as evidence.
