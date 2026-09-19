# Trusted CPU Runner Bootstrap and Recovery Eligibility

## Purpose

This document defines the fail-closed trust transition from the queue-neutral recovery branch to a trusted-CPU correctness run. It does **not** create scientific evidence, performance equivalence, or authority from queued/cancelled work.

The complete persistent-host recovery sequence is:

```text
external content-addressed bootstrap authorization
        ↓
Stage A: bootstrap.v9 PASS
        ↓
Stage B: provision hardened persistent host / ephemeral runner process
        ↓
Stage C: exact promotion + promotion.v5 PASS
        ↓
Stage D: main-only GitHub smoke + smoke.v2 PASS
        ↓
Stage E: promotion/smoke join + recovery-eligibility.v5 PASS
        ↓
separate exact, one-use Stage-F target authorization.v2
        ↓
root-owned atomic authorization consumption
        ↓
exact durable-ledger STATUS confirmation
        ↓
Stage F: one exact correctness recovery attempt
```

Each arrow is a separate authority/evidence transition. None may be inferred from the previous stage.

## External bootstrap authorization capsule

The recovery branch must never authorize itself. Before Stage A, an operator creates and retains an authorization capsule **outside the repository checkout** and independently records the SHA-256 of its exact bytes.

The capsule is deliberately bootstrap-only. It does not authorize a Stage-F workload, scientific qualification, or repair.

Exact format:

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

The authorization capsule is an auditable human decision record, not a cryptographic signature. Its independently recorded SHA-256 prevents later byte substitution within this procedure; organizational signer/authentication policy, if required, remains a separate trust layer.

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
- runner-policy and routing-policy Nix evaluation, including authorization socket/ledger isolation and query semantics;
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
6. Confirm `/var/lib/symthaea-stage-f-authorizations` is root-owned mode `0700`.
7. Confirm `/run/symthaea-stage-f-authorization.sock` is owned by root and the fixed runner socket group with mode `0660`.
8. Configure external retention for required runner/system/authorization-consumer logs.
9. Verify GitHub shows only capability `symthaea-trusted-cpu-v1`; default labels remain absent.

The authorization ledger must remain outside the ephemeral GitHub-runner state/work directories. Job code must not receive direct write access to the ledger.

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
- successful query of the root-owned Stage-F authorization consumer through the restricted Unix socket;
- exact equality between the consumer-reported boot ID and `/proc/sys/kernel/random/boot_id`;
- source-tree immutability check.

Only after all checks succeed, the final step emits:

```text
schema=symthaea.trusted-runner.smoke.v2
```

and its SHA-256. Retain the exact manifest bytes and hash from the successful GitHub run. Also record the GitHub run ID/attempt and `host_boot_id` from the manifest and independently inspect that run in GitHub.

`smoke.v2` binds the exact GitHub `main` head/tree, run identity, runner identity, **qualified host boot identity**, smoke workflow, runner/routing blobs, Nix/Rust provenance and PASS states. It is correctness/reproducibility smoke evidence only.

A reboot makes this smoke stale by design. If the smoke fails or the host subsequently reboots, do not route recovery workloads; obtain a fresh smoke before Stage E.

## Stage E — exact promotion/smoke join

Before any recovery workflow is dispatched, detach a pristine checkout at **current public `main`** and provide the exact retained promotion and smoke manifests plus their independently recorded hashes:

```bash
export SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_PATH='/path/to/promotion.v5'
export SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_SHA256='<recorded promotion SHA-256>'
export SYMTHAEA_TRUSTED_SMOKE_MANIFEST_PATH='/path/to/smoke.v2'
export SYMTHAEA_TRUSTED_SMOKE_MANIFEST_SHA256='<recorded smoke SHA-256>'

bash nix/ci/validate-trusted-runner-recovery-eligibility.sh
```

The Stage-E verifier requires:

- exact SHA-256 match for both evidence files before parsing;
- `promotion.v5 PASS` and `smoke.v2 PASS` schemas;
- same external bootstrap-authorization SHA-256/schema/scope carried from Stage A through Stage C;
- `operator_authorization_stage_f_authority=NONE` still holds;
- same promoted/smoked `main` head and tree;
- smoke carries a syntactically valid host boot ID and `stage_f_authorization_consumer=PASS`;
- same smoke-workflow, runner-module and routing-policy blobs;
- Stage-A/Stage-C-bound ARC3 protocol qualifier workflow, SE-001Q recovery workflow, SE-001Q replay helper and CI-shell blobs still match current `main`;
- current public `main` still equals the smoke-qualified head/tree;
- recovery branch still equals the authorized recovery head;
- pristine local checkout equals exact current public `main`;
- local eligibility-verifier blob equals the Stage-A-bound blob;
- both public refs remain stable through the join.

Success emits:

```text
schema=symthaea.trusted-runner.recovery-eligibility.v5
```

and a SHA-256. Retain both.

`recovery-eligibility.v5` carries:

```text
qualified_main_head
qualified_main_tree
qualified_host_boot_id
authorized_recovery_head
SE-001Q workflow/helper blobs
bootstrap-authorization continuity
```

The resulting theorem is:

```text
external bootstrap authorization exact bytes/hash
    + authorization scope == bootstrap-only
    + stage_f_authority == NONE
    + bootstrap.v9 PASS
    + promotion.v5 PASS
    + smoke.v2 PASS on host boot B
    + exact promotion/smoke identity join
    + explicit SE-001Q recovery workflow/helper identity join
    + current main/recovery ref stability
    = recovery-eligibility.v5 PASS for host boot B
```

Even this is **not scientific evidence and does not authorize Stage F**.

Any movement of public `main` after the smoke invalidates eligibility and requires a new smoke and Stage-E join. Movement of the recovery branch invalidates the external bootstrap authorization and requires a fresh Stage-A authorization capsule. A host reboot invalidates the smoke/eligibility boot epoch and requires a fresh Stage-D/E sequence.

## Stage F — one exact correctness recovery attempt

Only after `recovery-eligibility.v5 PASS` may an operator separately authorize one reviewed manual recovery attempt to use `symthaea-trusted-cpu-v1`.

The Stage-A authorization capsule is deliberately insufficient for this decision. `stage_f_authority=NONE` is carried all the way through eligibility to make that non-transfer machine-visible.

### SE-001Q Stage-F authorization capsule v2

Create the SE-001Q authorization only after retaining a valid `recovery-eligibility.v5` manifest and independently recording its SHA-256.

Generate a fresh 256-bit nonce for **this one authorization only**. For example on Linux:

```bash
AUTHORIZATION_NONCE="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
[[ "$AUTHORIZATION_NONCE" =~ ^[0-9a-f]{64}$ ]]
printf '%s\n' "$AUTHORIZATION_NONCE"
```

Never reuse a nonce, even for a failed or cancelled attempt after consumption.

Exact authorization format:

```text
schema=symthaea.trusted-runner.stage-f-authorization.v2
decision=AUTHORIZE_SE001Q_REOBSERVATION
repository=https://github.com/Luminous-Dynamics/symthaea.git
provider=trusted-cpu-v1
target_id=SE-001Q
target_commit=47de7f2a306cffb66b5505786220590aa5f42e90
target_tree=7abdf2ede579b2b729b057ca64470d11eb551031
hosted_verifier_commit=3dd33625162ec7137ef06c8079794a8d2aecc95d
recovery_eligibility_manifest_sha256=<exact retained recovery-eligibility.v5 SHA-256>
qualified_main_head=<qualified_main_head from recovery-eligibility.v5>
qualified_main_tree=<qualified_main_tree from recovery-eligibility.v5>
qualified_host_boot_id=<qualified_host_boot_id from recovery-eligibility.v5>
authorized_recovery_head=<authorized_recovery_head from recovery-eligibility.v5>
se001q_recovery_workflow_blob=<se001q_recovery_workflow_blob from recovery-eligibility.v5>
se001q_replay_helper_blob=<se001q_replay_helper_blob from recovery-eligibility.v5>
authorization_nonce=<fresh 64-hex 256-bit nonce>
max_uses=1
consumption_mode=root-owned-host-ledger-v2
authorization_scope=se001q-independent-provider-reobservation-only
qualification_claim=NONE
repair_authority_claim=NONE
```

Independently record its SHA-256. The capsule authorizes one execution attempt only. It is not a qualification, classification, DiagnosticWitness, or RepairGrant.

The workflow requires four explicit `workflow_dispatch` inputs:

```text
recovery_eligibility_manifest_base64
recovery_eligibility_manifest_sha256
stage_f_authorization_base64
stage_f_authorization_sha256
```

Prepare exact byte-preserving base64 strings with:

```bash
ELIGIBILITY_B64="$(base64 -w 0 /secure/path/recovery-eligibility.v5)"
STAGE_F_AUTH_B64="$(base64 -w 0 /secure/path/se001q-stage-f-authorization.v2)"
```

These capsules contain no repository credential and must never contain a PAT, runner token, or other secret.

### Machine preflight

Before consuming authority or executing a Rust gate, the workflow fails closed unless it can prove:

- exact byte hashes of both supplied capsules;
- `recovery-eligibility.v5 PASS` and all required Stage-E PASS predicates;
- bootstrap authorization remains bootstrap-only with `stage_f_authority=NONE`;
- `GITHUB_SHA` still equals current public `main`, not merely the main commit at dispatch time;
- current public `main` tree equals the eligibility-qualified tree;
- current public recovery branch still equals the eligibility-authorized recovery head;
- current workflow/helper blobs equal the Stage-E-bound blobs;
- Stage-F capsule binds exactly the eligibility hash, main identity, recovery identity, workflow/helper identity, frozen SE-001 subject and EV2.4 reference verifier;
- Stage-F capsule boot identity equals the eligibility-qualified boot;
- authorization nonce is exactly 64 lowercase hex characters;
- `max_uses=1`;
- `consumption_mode=root-owned-host-ledger-v2`;
- authorization scope is `se001q-independent-provider-reobservation-only`;
- qualification and repair-authority claims remain `NONE`;
- runner/routing eval tests still pass.

### Atomic one-time consumption and status confirmation

Only after all preflight/policy checks pass does the workflow approach the local root-owned authorization consumer.

It first sends:

```text
BOOT_ID
```

and requires the returned boot ID to equal `qualified_host_boot_id` from both eligibility and Stage-F authorization. A reboot therefore fails **before consuming the nonce**.

It then sends exactly:

```text
CONSUME <authorization_nonce> <stage_f_authorization_sha256>
```

The root consumer atomically reserves the nonce marker in the persistent ledger, writes the authorization digest and boot identity to a temporary file, then atomically renames that file to the durable consumption record. Exactly one concurrent/repeated consumer can reserve the nonce.

A `CONSUMED` response is not sufficient by itself. The workflow immediately sends:

```text
STATUS <authorization_nonce> <stage_f_authorization_sha256>
```

and requires the exact response:

```text
CONSUMED_STATUS <authorization_nonce> <stage_f_authorization_sha256> <qualified_host_boot_id>
```

Only after that durable-ledger confirmation does the workflow emit a content-addressed:

```text
symthaea.trusted-runner.stage-f-consumption.v2
```

receipt. The receipt is bound to the authorization hash/nonce, qualified boot, GitHub run/attempt, qualified main and authorized recovery head and records `ledger_status_checked=PASS`.

Only after this receipt exists may the scientific replay step begin.

### Consumption semantics

One-use means **one execution attempt**, not one successful observation.

If authorization consumption succeeds and the later Rust replay, evidence capture, runner process, network, disk, or finalization fails, that authorization remains spent.

```text
CONSUMED
    + later failure/cancellation
    => authorization spent
    => no retry/re-run with same nonce
```

For another attempt, create a **new Stage-F authorization.v2 with a new 256-bit nonce**. The same `recovery-eligibility.v5` may be reused only while all of its identities remain current and the qualified host has not rebooted.

A GitHub job re-run after consumption is expected to fail at the one-time consumer. Do not weaken this behavior to make re-runs convenient.

If a run fails before or around consumption, do not infer authority state from the GitHub job conclusion or from absence of a local receipt. Query the root ledger with the exact nonce/authorization pair when possible:

```text
UNUSED                 => no marker exists on this qualified boot
CONSUMED_STATUS        => exact authorization is spent
CONSUMED_DIFFERENT     => nonce is spent by a different authorization identity
INCOMPLETE             => nonce reservation occurred but complete record is unavailable
```

`INCOMPLETE` is fail-closed: treat the authorization as unavailable and issue a fresh Stage-F authorization with a fresh nonce for any later attempt. The rejection finalizer records the corresponding state as `UNUSED_CONFIRMED`, `CONSUMED_CONFIRMED`, `NONCE_CONSUMED_DIFFERENT_AUTHORIZATION`, or `CONSUMPTION_UNCERTAIN` where the query is available.

## SE-001Q provider observation and execution binding

After successful authorization consumption and durable status confirmation, the trusted helper:

- authenticates the EV2.4 experiment/classifier as inert data;
- proves the five gate vectors and negative-control semantics match the trusted implementation;
- captures all gates even after nonzero outcomes;
- emits content-addressed gate observations/classifications and evidence manifest;
- executes no unmerged EV2.4 Python implementation;
- grants no qualification or repair authority.

Before constructing the successful execution binding, the finalizer independently re-queries `STATUS` and again requires the exact consumed nonce/authorization/boot tuple.

A successful replay additionally emits:

```text
symthaea.se001q.trusted-cpu-execution-binding.v3
```

which binds the provider observation to:

- exact eligibility SHA-256;
- exact Stage-F authorization SHA-256 and nonce;
- `max_uses=1` / `root-owned-host-ledger-v2` consumption mode;
- exact Stage-F consumption.v2 receipt;
- independently reconfirmed durable ledger status;
- qualified host boot;
- GitHub run/attempt;
- current main/recovery/workflow/helper identities;
- frozen SE-001 subject and EV2.4 reference;
- observation manifest and evidence-bundle identities.

Keep the objects distinct:

```text
Stage-F Authorization
    != Authorization Consumption
    != Durable Ledger Status
    != Provider Observation
    != Execution Authorization Binding
    != Classification
    != DiagnosticWitness
    != RepairGrant
```

A failed/incomplete execution still emits reconstructible rejection evidence where possible. Partial rejection v5 records the best independently queryable ledger state but cannot satisfy the successful replay contract.

## Recovery-target priority

Do not dispatch RCA, ARC3, SE-001Q, or other staged recovery targets merely because they exist. Authorize **one exact Stage-F target/attempt at a time**.

The historical first RCA target and ARC3 protocol subject remain separately reviewed targets. SE-001Q is additionally staged through `.github/workflows/self-hosted-se001q-evidence-recovery.yml` with trusted harness logic in `nix/ci/se001q-trusted-replay.py`.

The SE-001Q target remains deliberately narrower than qualification:

```text
trusted CPU SE-001Q observation != hosted EV2.4 observation
trusted CPU SE-001Q observation != SE-001 qualification
trusted CPU SE-001Q observation != RepairGrant
```

## Failure semantics

At every stage:

```text
queued or unexecuted != PASS
cancelled for supersession != PASS or FAIL
bootstrap authorization != Stage-F authorization
bootstrap PASS != promotion PASS
promotion PASS != smoke PASS
smoke PASS != recovery eligibility
recovery eligibility != Stage-F authority
Stage-F authority != Stage-F consumption
Stage-F consumption != durable-ledger confirmation
Stage-F consumption != scientific qualification
consumed authorization != reusable authorization
INCOMPLETE consumption != reusable authorization
host reboot => prior smoke/eligibility/Stage-F authorization stale
trusted CPU correctness PASS != performance equivalence
trusted CPU SE-001Q observation != hosted EV2.4 observation
trusted CPU SE-001Q observation != RepairGrant
current branch head != operator authorization
runner process ephemeral != host ephemeral
content-similar landing != exact qualified tree
squash/cherry-pick != preserved qualified ancestry
changed main after smoke => smoke/recovery eligibility stale
changed recovery branch after Stage A => bootstrap authorization stale
```

The fallback exists to restore executable evidence, never to weaken what counts as evidence.
