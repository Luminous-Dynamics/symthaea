# Merge Admission Receipt v1

## Purpose

Symthaea needs a hard semantic boundary between **evidence that a candidate was tested** and **authority to integrate that candidate**.

The central theorem is:

```text
candidate-owned CI result
    !=
merge authority
```

A successful workflow run is evidence. It is not, by itself, proof that:

- the run belongs to the exact candidate head now under review;
- it was evaluated against the current target base;
- the expected workflow generation executed;
- every required job actually ran rather than being skipped;
- the candidate did not alter the CI/governance control plane that interprets its own result;
- repository policy admits the result as sufficient for integration.

`MergeAdmissionReceiptV1` makes those distinctions explicit.

## Scope of v1

V1 is a **pure policy core and receipt format**.

It does not:

- call GitHub;
- merge a pull request;
- create a GitHub status/check;
- sign a receipt;
- claim that caller-supplied observations are authentic;
- run candidate code;
- replace code review;
- weaken full integration qualification.

The evaluator is intentionally useful before enforcement exists: it freezes the semantics that a future trusted collector/check must satisfy.

## Five dispositions

V1 does not reduce merge state to green/red.

```text
ADMITTED
    all v1 predicates are satisfied

INCOMPLETE
    required evidence is absent, pending, cancelled, skipped, or not known complete

STALE
    evidence belongs to a previous candidate head or previous target base

BOOTSTRAP_REQUIRED
    the candidate changes the merge-authority control plane itself

REJECTED
    evidence identity is wrong, malformed, contradictory, or contains an explicit failed required gate
```

These states are intentionally non-equivalent.

```text
not yet proven != stale proof != changed authority machinery != failed proof
```

## Exact identity boundary

Every observation binds:

```text
repository
 target branch
 current target-base SHA
 candidate head SHA
 candidate tree SHA
 policy content digest
 control-plane blob identities
 full-integration workflow blob identity
 workflow run ID
 workflow event
 qualification head SHA
 qualification base SHA
 required-job observation set
```

The output receipt also binds the full evidence observation through:

```text
evidence_binding_sha256
```

and binds the complete receipt body through:

```text
receipt_sha256
```

These hashes are content addresses, not signatures.

## Head currentness

A successful run on head `H1` does not qualify head `H2`.

```text
run.head_sha != candidate_head_sha
    -> STALE
```

No branch-name match, textual lineage claim, or "latest successful run" substitution may override exact head identity.

## Base currentness and TOCTOU

A candidate qualified against target base `B1` is not automatically admitted after the target branch advances to `B2`.

```text
run.base_sha != current_target_base_sha
    -> STALE
```

This deliberately chooses correctness over convenience for v1.

A future policy may admit a separately qualified equivalence/rebase theorem, but v1 contains no such shortcut.

## Control-plane equivalence

For an ordinary candidate, the merge-authority control plane must be byte-identical to the current target-base generation.

Each governed path is observed as:

```text
path
base_blob_sha | null
candidate_blob_sha | null
```

`null == null` means the path is absent from both trees and therefore unchanged.

Any difference means:

```text
candidate changes authority/control plane
    -> BOOTSTRAP_REQUIRED
```

This is stronger than trusting a workflow merely because it has the expected filename or display name.

### Why this matters

Without this rule, a candidate could modify the workflow that determines whether its own expensive tests execute, then present a green result from that modified workflow as proof of merge readiness.

V1 instead says:

```text
normal product change
    + unchanged trusted control plane
    + exact current full integration
        -> eligible for ordinary admission evaluation

control-plane change
        -> independent bootstrap/review path
```

## Bootstrap is explicit, not an error case

A CI/governance change is legitimate work. It simply cannot establish its own authority using the machinery it changes.

The first merge of this v1 policy is itself such a bootstrap event because the target base does not yet contain this policy file.

The same applies to the lifecycle-tiering tranche while its routing files do not yet exist on `main`.

A bootstrap procedure must therefore be independently reviewed/qualified, and only after it lands does its exact target-base generation become ordinary admission policy for later candidates.

## Trusted workflow identity

The full-integration observation must use the exact configured workflow path and the exact workflow blob from the trusted target-base control plane.

```text
observed workflow path != policy workflow path
    -> REJECTED

observed workflow blob != trusted base workflow blob
    -> REJECTED
```

A similarly named workflow cannot substitute.

## Required-job completeness

A top-level workflow conclusion of `success` is insufficient if required jobs were omitted or skipped.

The trusted collector must establish:

```text
job_set_complete == true
```

and provide the required-job observation set.

Every required job must be:

```text
status == completed
conclusion == success
skipped != true
```

A skipped required job produces `INCOMPLETE`, not success.

An explicitly failed required job produces `REJECTED`.

A cancelled full run produces `INCOMPLETE`; cancellation is not evidence that the candidate failed.

## Collector trust boundary

The pure evaluator cannot authenticate its own input.

Therefore:

```text
unsigned local receipt
    !=
repository-enforced merge authority
```

An enforcement deployment needs a **trusted collector** that obtains repository, ref, blob, workflow-run, and job data from GitHub's API (or another independently trusted source), not from a candidate-authored artifact.

The trusted collector should be:

- read-only with respect to candidate contents during evidence collection;
- external/base-owned rather than candidate-selected;
- exact-head and exact-base aware;
- able to enumerate the complete required-job set;
- unable to accept a candidate-provided `job_set_complete=true` assertion without independently establishing it;
- separately versioned and content/provenance identified.

The collector and evaluator can later produce a GitHub App check or equivalent admission signal that a repository ruleset requires.

## Do not use privileged candidate execution as the shortcut

Do not solve the trust problem by broadly running candidate code under `pull_request_target` with elevated base context.

That mixes two different concerns:

```text
trusted observation/admission logic
    vs
untrusted candidate execution
```

The admission collector should inspect existing evidence and metadata. Candidate code should execute in the ordinary unprivileged qualification environment.

## Relationship to lifecycle tiering

CI lifecycle tiering answers:

```text
which verification work should run now?
```

Merge admission answers:

```text
what exact evidence is sufficient to integrate this exact candidate now?
```

They must remain separate.

```text
draft Tier-1 green
    != full integration green
    != merge admission
```

The lifecycle scheduler may reduce expensive work on draft PRs. It cannot mint an admission receipt merely because the cheap tier passed.

## Relationship to focused scientific qualification

Focused theorem gates remain valuable evidence, but v1 explicitly forbids:

```text
focused evidence
    -> substitute for full repository integration
```

unless a future policy revision defines and qualifies such a substitution rule.

This preserves the distinction between:

```text
scientific proposition qualification
repository integration qualification
merge authority
```

## Receipt state machine

Conceptually:

```text
Candidate
   |
   v
Collect exact evidence
   |
   +---- missing/pending/skipped --------> INCOMPLETE
   |
   +---- wrong identity/explicit fail ---> REJECTED
   |
   +---- old head/base ------------------> STALE
   |
   +---- control-plane changed ----------> BOOTSTRAP_REQUIRED
   |
   v
All predicates satisfied
   |
   v
ADMITTED
```

Any head or base movement invalidates the old admission state and requires a new evaluation.

## Evidence monotonicity

Authority may increase only through explicit verified transitions.

```text
run exists
    != run completed
    != run succeeded
    != required jobs complete
    != evidence current
    != control plane trusted
    != admitted
```

No aggregate score may skip these transitions.

## Policy identity

The evaluator computes `policy_sha256` from the exact policy bytes when invoked through the CLI.

This means two semantically similar policies with different bytes are distinct policy artifacts for receipt provenance.

Changing the policy is itself a control-plane event and therefore requires independent bootstrap under the previous trusted generation.

## Current v1 policy

The staged policy is:

```text
scripts/ci/merge_admission_policy_v1.json
```

It currently names the monolithic CI workflow plus the lifecycle/admission policy surfaces as the merge-authority control plane. Paths absent from both target base and candidate are equivalent; introducing/changing one is a bootstrap event.

This is intentionally conservative while lifecycle tiering is still staged separately.

## Required adversarial cases

The v1 test corpus freezes at least these cases:

1. exact current full success -> `ADMITTED`;
2. no full integration -> `INCOMPLETE`;
3. Tier-1/focused green with no full integration -> `INCOMPLETE`;
4. successful old head -> `STALE`;
5. successful old base -> `STALE`;
6. candidate changes CI workflow -> `BOOTSTRAP_REQUIRED`;
7. candidate changes admission policy -> `BOOTSTRAP_REQUIRED`;
8. similarly named wrong workflow -> `REJECTED`;
9. wrong workflow blob -> `REJECTED`;
10. successful workflow with skipped required job -> `INCOMPLETE`;
11. failed required job -> `REJECTED`;
12. cancelled workflow -> `INCOMPLETE`;
13. in-progress workflow -> `INCOMPLETE`;
14. incomplete required-job census -> `INCOMPLETE`;
15. duplicate job/control-plane observations -> `REJECTED`;
16. wrong repository -> `REJECTED`;
17. policy change changes receipt identity;
18. workflow-run identity changes receipt identity;
19. required-job evidence changes receipt identity.

## Enforcement-ready exit gate

V1 should not be called repository-enforced merge authority until all are true:

1. the pure evaluator test corpus passes on its exact head;
2. a trusted collector independently obtains current base/head/tree/blob/run/job facts;
3. the collector proves completeness of the required job set from a trusted base-owned manifest or equivalent source;
4. collector implementation identity is itself governed;
5. an admission result is published as a distinct check/status that cannot be produced by candidate code;
6. repository rules require that admission check before merge;
7. control-plane changes follow an independently reviewed bootstrap process;
8. exact-head and exact-base staleness is enforced;
9. lifecycle-skipped jobs cannot appear as successful required evidence;
10. no `pull_request_target` path executes arbitrary candidate code with privileged credentials/secrets.

Until then, the receipt remains an executable specification of merge semantics rather than an enforcement claim.
