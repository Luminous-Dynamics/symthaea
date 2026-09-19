# Trusted CPU Runner Bootstrap and Recovery Eligibility

## Purpose

This document defines the fail-closed trust transition from the queue-neutral recovery branch to a trusted-CPU correctness run. It does **not** create scientific evidence, performance equivalence, or authority from queued/cancelled work.

The complete v1 recovery sequence is:

```text
external content-addressed bootstrap authorization
        ↓
Stage A: bootstrap.v9 PASS
        ↓
Stage B: provision hardened persistent host / ephemeral runner process
        ↓
Stage C: exact promotion + promotion.v5 PASS
        ↓
Stage D: main-only GitHub smoke + smoke.v1 PASS
        ↓
Stage E: promotion/smoke join + recovery-eligibility.v4 PASS
        ↓
separate explicit Stage-F target authorization
        ↓
Stage F: one exact correctness recovery target
```

Each arrow is a separate evidence transition. None may be inferred from the previous stage.

## External bootstrap authorization capsule

The recovery branch must never authorize itself. Before Stage A, an operator creates and retains an authorization capsule **outside the repository checkout** and independently records the SHA-256 of its exact bytes.

The capsule is deliberately bootstrap-only. It does not authorize a Stage-F workload, scientific qualification, or repair.

Exact v1 format:

```text
schema=symthaea.trusted-runner.bootstrap-authorization.v1
decision=AUTHORIZE_BOOTSTRAP
repository=https://github.com/Luminous-Dynamics/symthaea.git
recovery_branch=ci/nixos-ephemeral-runner-v1
authorized_recovery_head=<reviewed 40-hex recovery commit>
authorized_recovery_tree=<reviewed 40-hex recovery tree>
authorized_main_head=<reviewed 40-hex current main commit>
authorized_main_tree=<reviewed 40-hex current main tree>
recovery_diff_paths_sha256=<sha256 of sorted recovery-only path list plus final newline>
authorization_scope=trusted-runner-bootstrap-only
stage_f_authority=NONE
qualification_claim=NONE
repair_authority_claim=NONE
```

Every key must occur exactly once with a non-empty value. Do not add a favorable timestamp, target name, qualification result, or repair permission to this object.

A reproducible way to obtain the identities before making the human authorization decision is:

```bash
git fetch origin main ci/nixos-ephemeral-runner-v1
MAIN_HEAD="$(git rev-parse origin/main)"
MAIN_TREE="$(git rev-parse origin/main^{tree})"
RECOVERY_HEAD="$(git rev-parse origin/ci/nixos-ephemeral-runner-v1)"
RECOVERY_TREE="$(git rev-parse origin/ci/nixos-ephemeral-runner-v1^{tree})"
RECOVERY_DIFF_PATHS_SHA256="$(git diff --name-only "$MAIN_HEAD" "$RECOVERY_HEAD" | LC_ALL=C sort | sha256sum | awk '{print $1}')"

printf 'main=%s tree=%s\nrecovery=%s tree=%s\npaths=%s\n' \
  "$MAIN_HEAD" "$MAIN_TREE" "$RECOVERY_HEAD" "$RECOVERY_TREE" "$RECOVERY_DIFF_PATHS_SHA256"
```

Review those identities and the exact diff before creating the external capsule. Then independently record:

```bash
sha256sum /secure/path/bootstrap-authorization.v1
```

The authorization capsule is an auditable human decision record, not a cryptographic signature. Its independently recorded SHA-256 prevents later byte substitution within this bootstrap procedure; organizational signer/authentication policy, if required, remains a separate trust layer.

## Stage A — host-side bootstrap validation

Run on the isolated NixOS host that will become the trusted CPU runner. Do not infer authorization from the branch head, an issue comment alone, or successful static review.

```bash
export SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD='<reviewed 40-hex recovery commit>'
export SYMTHAEA_TRUSTED_RECOVERY_AUTHORIZATION_PATH='/secure/path/bootstrap-authorization.v1'
export SYMTHAEA_TRUSTED_RECOVERY_AUTHORIZATION_SHA256='<independently recorded 64-hex SHA-256>'

git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
git fetch origin ci/nixos-ephemeral-runner-v1
git checkout --detach origin/ci/nixos-ephemeral-runner-v1

test "$(git rev-parse HEAD)" = "$SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD"
bash nix/ci/validate-trusted-runner-bootstrap.sh
```

The validator independently requires:

- explicit 40-hex expected recovery head;
- exact external authorization-capsule byte hash equals the independently recorded SHA-256;
- authorization capsule resolves outside the repository checkout;
- authorization schema/decision/repository/branch/scope are exact;
- authorization says `stage_f_authority=NONE`, `qualification_claim=NONE`, and `repair_authority_claim=NONE`;
- authorization recovery head/tree equal the checked-out and freshly fetched recovery generation;
- authorization current-`main` head/tree equal freshly fetched public `main`;
- authorization recovery path digest equals the exact reviewed path allowlist digest;
- canonical public HTTPS origin;
- pristine tracked/untracked/ignored source tree;
- current public `main` is an ancestor of recovery;
- exact reviewed recovery path allowlist;
- exact runner/routing/smoke/ARC3-qualifier/SE-001Q-recovery/SE-001Q-helper/CI-shell/bootstrap/promotion/recovery-eligibility/lifecycle artifact identities;
- shell syntax validation of bootstrap/promotion/recovery-eligibility verifiers;
- runner-policy and routing-policy Nix evaluation;
- pinned minimal Rust environment with locked Cargo metadata and compilation;
- parse-compilation of the trusted SE-001Q replay helper under the pinned Python environment;
- unchanged source commit/tree;
- unchanged public `main` and recovery refs across the complete validation interval.

A successful run emits:

```text
schema=symthaea.trusted-runner.bootstrap.v9
```

and a SHA-256 of the exact manifest bytes. Retain **both** the manifest and printed hash outside the repository.

`bootstrap.v9` binds:

- SHA-256, schema, scope, and explicit no-Stage-F-authority state of the external authorization capsule;
- operator-authorized recovery head and exact source tree;
- exact pre-promotion `main` head/tree;
- exact promoted-tree requirement and required recovery ancestor;
- reviewed path-set SHA-256;
- runner module, routing policy, smoke workflow, ARC3 protocol qualifier workflow, SE-001Q recovery workflow, SE-001Q trusted replay helper and pinned CI-shell blobs;
- bootstrap, promotion and recovery-eligibility verifier blobs;
- host-lifecycle contract blob;
- nixpkgs/Rust/lock/toolchain provenance;
- host Nix system/version;
- PASS states for external authorization, policy evaluation, locked Rust validation, helper/verifier syntax validation and ref stability.

If either public ref moves or any reviewed artifact changes, Stage A is stale. Review and create a fresh external authorization capsule for the new recovery generation before rerunning.

## Stage B — provision the host

Only after `bootstrap.v9 PASS`:

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

Before landing, current public `main` must still equal the `main_head` recorded by Stage A. Promotion must preserve the operator-authorized recovery commit in ancestry. **Do not squash or cherry-pick.** The resulting `main` tree must be byte-identical to `promotion_expected_main_tree` from `bootstrap.v9`.

After landing, detach a pristine checkout at current public `main` and verify promotion using the exact retained Stage-A evidence:

```bash
export SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_PATH='/path/to/bootstrap.v9'
export SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_SHA256='<recorded Stage-A SHA-256>'

bash nix/ci/validate-trusted-runner-promotion.sh
```

The verifier first authenticates the exact bootstrap-manifest bytes. It then requires:

- `bootstrap.v9 PASS`;
- external authorization SHA-256/schema/scope are present and `stage_f_authority=NONE`;
- local pristine checkout = current public `main` head/tree;
- local promotion-verifier blob = Stage-A-bound verifier blob;
- recovery branch still equals the authorized recovery head;
- Stage-A `main` remains in promoted ancestry;
- authorized recovery head remains in promoted ancestry;
- promoted tree = exact Stage-A-qualified recovery tree;
- promoted diff-path digest = Stage-A digest;
- runner/routing/smoke/ARC3/SE-001Q/CI-shell/bootstrap/promotion/recovery-eligibility/lifecycle blobs = Stage-A-qualified blobs;
- explicit re-read of `.github/workflows/self-hosted-se001q-evidence-recovery.yml` and `nix/ci/se001q-trusted-replay.py` from promoted `main`;
- public refs stable through promotion verification.

Success emits:

```text
schema=symthaea.trusted-runner.promotion.v5
```

plus its SHA-256. `promotion.v5` carries the same external bootstrap-authorization digest and explicit bootstrap-only/no-Stage-F scope forward. Retain both exact bytes and hash.

`promotion.v5 PASS` means only **Stage-D smoke is eligible**. It does not qualify the runner or authorize a recovery target.

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

If the smoke fails, do not route recovery workloads. Repair the observed substrate defect; if trusted infrastructure changes, restart at Stage A with a fresh external authorization capsule.

## Stage E — exact promotion/smoke join

Before any recovery workflow is dispatched, detach a pristine checkout at **current public `main`** and provide the exact retained promotion and smoke manifests plus their independently recorded hashes:

```bash
export SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_PATH='/path/to/promotion.v5'
export SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_SHA256='<recorded promotion SHA-256>'
export SYMTHAEA_TRUSTED_SMOKE_MANIFEST_PATH='/path/to/smoke.v1'
export SYMTHAEA_TRUSTED_SMOKE_MANIFEST_SHA256='<recorded smoke SHA-256>'

bash nix/ci/validate-trusted-runner-recovery-eligibility.sh
```

The Stage-E verifier requires:

- exact SHA-256 match for both evidence files before parsing;
- `promotion.v5 PASS` and `smoke.v1 PASS` schemas;
- same external bootstrap-authorization SHA-256/schema/scope carried from Stage A through Stage C;
- `operator_authorization_stage_f_authority=NONE` still holds;
- same promoted/smoked `main` head and tree;
- same smoke-workflow, runner-module and routing-policy blobs;
- Stage-A/Stage-C-bound ARC3 protocol qualifier workflow, SE-001Q recovery workflow, SE-001Q replay helper and CI-shell blobs still match current `main`;
- current public `main` still equals the smoke-qualified head/tree;
- recovery branch still equals the authorized recovery head;
- pristine local checkout equals exact current public `main`;
- local eligibility-verifier blob equals the Stage-A-bound blob;
- both public refs remain stable through the join.

Success emits:

```text
schema=symthaea.trusted-runner.recovery-eligibility.v4
```

and a SHA-256. Retain both.

This creates the final bootstrap theorem:

```text
external bootstrap authorization exact bytes/hash
    + authorization scope == bootstrap-only
    + stage_f_authority == NONE
    + bootstrap.v9 PASS
    + promotion.v5 PASS
    + smoke.v1 PASS
    + exact promotion/smoke identity join
    + explicit SE-001Q recovery workflow/helper identity join
    + current main/recovery ref stability
    = recovery-eligibility.v4 PASS
```

Even this is **not scientific evidence and does not authorize Stage F**. It proves that an eligible trusted-runner substrate descended from one exact external bootstrap decision.

Any movement of public `main` after the smoke invalidates v1 recovery eligibility and requires a new smoke and Stage-E join. Movement of the recovery branch invalidates the external bootstrap authorization and requires a fresh Stage-A authorization capsule.

## Stage F — one exact correctness recovery target

Only after `recovery-eligibility.v4 PASS` may an operator separately authorize one reviewed manual recovery workflow to use `symthaea-trusted-cpu-v1`.

The Stage-A authorization capsule is deliberately insufficient for this decision. `stage_f_authority=NONE` is carried all the way through eligibility to make that non-transfer machine-visible.

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

Proceed one prerequisite at a time. Recovery-target priority is an **operator authorization decision**, not something inferred from branch age or queue duration. The historical first RCA target remains exact PR #578 canonical-lineage generation; ARC3 protocol subject `6ea96737aff361920c181891a71f80a9f481ddef` is staged as a separately reviewed exact correctness target through `.github/workflows/arc3-protocol-trusted-cpu-qualify.yml`.

SE-001Q is additionally staged through `.github/workflows/self-hosted-se001q-evidence-recovery.yml`, with trusted harness logic in `nix/ci/se001q-trusted-replay.py`. That target is deliberately narrower than qualification:

```text
trusted CPU SE-001Q observation != hosted EV2.4 observation
trusted CPU SE-001Q observation != SE-001 qualification
trusted CPU SE-001Q observation != RepairGrant
```

The trusted helper authenticates the EV2.4 experiment/classifier as inert data, proves the five gate vectors and negative-control semantics match its trusted implementation, captures all gates, and emits an independent provider observation. The unmerged EV2.4 capture/verifier/attester code is authenticated but never executed on the trusted host.

Do not dispatch RCA, ARC3 and SE-001Q merely because they are all staged. Authorize one exact Stage-F target at a time. Do not prepare or dispatch #531/#555/#582/#585/#588 recovery in parallel.

## Failure semantics

At every stage:

```text
queued or unexecuted != PASS
cancelled for supersession != PASS or FAIL
branch head != operator authorization
bootstrap authorization != Stage-F authorization
bootstrap PASS != promotion PASS
promotion PASS != smoke PASS
smoke PASS != recovery eligibility
recovery eligibility != Stage-F authorization
recovery eligibility != scientific qualification
trusted CPU correctness PASS != performance equivalence
trusted CPU SE-001Q observation != hosted EV2.4 observation
trusted CPU SE-001Q observation != RepairGrant
runner process ephemeral != host ephemeral
content-similar landing != exact qualified tree
squash/cherry-pick != preserved qualified ancestry
changed main after smoke => smoke/recovery eligibility stale
changed recovery generation => bootstrap authorization stale
```

The fallback exists to restore executable evidence, never to weaken what counts as evidence.
