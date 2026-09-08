# Merge Admission Receipt v1

## Purpose

Symthaea needs a hard semantic boundary between **evidence that a candidate was
tested** and **authority to integrate that candidate**.

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
- the collector enumerated every job in the exact run attempt;
- the enumerated jobs satisfy the trusted target-base required-job manifest;
- the candidate did not alter the CI/governance control plane that interprets
  its own result;
- repository policy admits the result as sufficient for integration.

`MergeAdmissionReceiptV1` makes those distinctions explicit.

## Scope of v1

V1 is a **pure policy core and receipt format**.

It does not:

- call GitHub;
- merge a pull request;
- create a GitHub status/check;
- sign a receipt;
- claim caller-supplied observations are authentic;
- run candidate code;
- replace code review;
- weaken full integration qualification.

The evaluator is useful before enforcement exists because it freezes the
semantics that a future trusted collector/check must satisfy.

The policy is permanently:

```text
enforcement_ready = false
```

for this generation. An `ADMITTED` v1 receipt is an unsigned policy-core
disposition, not repository merge authority.

## Five dispositions

V1 does not reduce merge state to green/red.

```text
ADMITTED
    all v1 policy predicates over the supplied trusted observation are satisfied

INCOMPLETE
    required evidence is absent, pending, cancelled, skipped, the GitHub job
    census is not known complete, or the trusted required-job manifest is not
    complete/satisfied

STALE
    evidence belongs to a previous candidate head or previous target base

BOOTSTRAP_REQUIRED
    the candidate changes the merge-authority control plane itself

REJECTED
    evidence identity is wrong, malformed, contradictory, contains an explicit
    failed required gate, or does not match the trusted manifest generation
```

These states are intentionally non-equivalent:

```text
not yet proven
    !=
stale proof
    !=
changed authority machinery
    !=
failed/contradictory proof
```

## Exact run identity

GitHub reruns preserve a workflow `run_id` while incrementing `run_attempt`.
Therefore:

```text
run_id
    !=
exact execution identity

(run_id, run_attempt)
    =
exact workflow execution attempt
```

Both fields are required positive integers and are bound into the receipt
evidence digest.

A rerun attempt is distinct evidence even when every other run attribute is
unchanged.

## Exact candidate/base identity

Every observation binds:

```text
repository
target branch
current target-base SHA
candidate head SHA
candidate tree SHA
```

A successful run on head `H1` does not qualify head `H2`.

```text
run.head_sha != candidate_head_sha
    -> STALE
```

Likewise, a candidate qualified against target base `B1` is not automatically
admitted after the target branch advances to `B2`.

```text
run.base_sha != current_target_base_sha
    -> STALE
```

No branch-name match, textual lineage claim, or "latest successful run"
substitution may override exact head/base identity.

## Control-plane equivalence

For an ordinary candidate, the merge-authority control plane must be
byte-identical to the current target-base generation.

Each governed path is observed as:

```text
path
base_blob_sha | null
candidate_blob_sha | null
```

`null == null` means the path is absent from both trees and unchanged.

Any difference means:

```text
candidate changes authority/control plane
    -> BOOTSTRAP_REQUIRED
```

The governed surface includes the CI workflow, lifecycle scheduler inputs, the
admission policy/evaluator, observation/receipt schemas, required-job
manifest/schema, and normative contracts.

This closes the recursion:

```text
candidate changes judge
candidate's changed judge says PASS
    -> NOT ordinary admission
```

## Policy and manifest bytes are themselves bound

It is insufficient for a trusted observation to merely *name* a base blob while
the evaluator consumes different local bytes.

The evaluator therefore computes the Git blob object identity of the exact
policy and required-job manifest bytes it is given and requires those object IDs
to equal the target-base control-plane blob observations.

Conceptually:

```text
bytes consumed by evaluator
    -> Git blob OID
    == trusted target-base blob OID
```

for both:

```text
scripts/ci/merge_admission_policy_v1.json
scripts/ci/required_ci_job_manifest_v1.json
```

This is in addition to SHA-256 content addresses embedded in the receipt.

## Trusted workflow identity

The full-integration observation must use the exact configured workflow path and
the exact workflow blob from the trusted target-base control plane.

```text
observed workflow path != policy workflow path
    -> REJECTED

observed workflow blob != trusted base workflow blob
    -> REJECTED
```

The required-job manifest independently binds that same workflow path/blob. A
similarly named workflow or manifest for a different workflow generation cannot
substitute.

## Job-census completeness is not manifest satisfaction

This distinction is load-bearing.

```text
complete GitHub API census
    !=
required qualification surface satisfied
```

A collector can truthfully paginate every job GitHub returned from an
accidentally weakened workflow. That would be a **complete census of incomplete
qualification**.

The observation therefore carries:

```text
job_census_complete
job_census[]
```

where each census row binds:

```text
job_id
API-visible job name
status
conclusion
skipped
```

The collector is responsible only for establishing that the census is complete
for the exact `(run_id, run_attempt)`.

The collector does **not** assert `required_job_manifest_match=true`.

## Required-job manifest

Required-job selection is derived by the pure evaluator from the base-owned:

```text
scripts/ci/required_ci_job_manifest_v1.json
```

The manifest binds:

```text
manifest schema
workflow path
workflow blob identity
complete flag
event profiles
top-level job IDs
API-name family regexes
minimum/maximum instance cardinalities
required disposition:
    success
    allowed_skip
```

For a complete manifest, each top-level job ID must be represented exactly once
by a family in every profile.

At runtime the evaluator:

1. chooses the profile for the exact workflow event;
2. matches API-visible job names using `fullmatch`, not substring search;
3. verifies each family cardinality;
4. rejects jobs matching multiple families;
5. rejects jobs absent from the complete profile;
6. applies each family's required disposition;
7. derives the exact required-job subset and its digest.

Therefore:

```text
collector says census complete
        +
trusted manifest bytes
        +
pure evaluator
        ->
manifest satisfaction
```

—not:

```text
collector says "manifest passed"
        ->
authority
```

## Current manifest is deliberately incomplete

The staged real manifest is bound to the audited `ci.yml` blob:

```text
a48366076b30eb8e12d22c927a3b8bf333181409
```

but currently carries:

```text
complete = false
```

with empty event profiles.

That is intentional.

Until an exact structural/job-family census is derived and separately validated
against that workflow generation:

```text
workflow success
    + complete GitHub job census
    + incomplete manifest
        -> INCOMPLETE
```

The bootstrap tranche therefore cannot accidentally become merge authority
before its own required surface is known.

## Job disposition

For a family requiring `success`, every matched instance must be:

```text
status == completed
conclusion == success
skipped != true
```

A skipped or pending required instance produces `INCOMPLETE`.

An explicitly failed required instance produces `REJECTED`.

For a family declared `allowed_skip`, an explicitly skipped instance is allowed,
but an explicit failure remains `REJECTED`.

Too few family instances produce `INCOMPLETE`.

Too many, overlapping, or unmanifested instances produce `REJECTED`, because
they indicate manifest/workflow shape drift rather than merely unfinished work.

## Receipt evidence binding

The receipt content-addresses:

```text
policy SHA-256
required-job manifest SHA-256
candidate/base identities
control-plane blob identities
workflow path/blob
run ID
run attempt
event
status/conclusion
head/base
job-census completeness
job-census count + digest
derived manifest satisfaction
decision
reasons
```

The derived satisfaction summary binds:

```text
profile
family count
required-job count
required-job digest
family-summary digest
```

Two workflow attempts, two job censuses, or two manifest generations therefore
cannot silently collapse into one receipt identity.

These hashes are content addresses, not signatures.

## Bootstrap is explicit

A CI/governance change is legitimate work. It simply cannot establish its own
authority using the machinery it changes.

The first merge of this policy is itself a bootstrap event because `main` does
not yet contain the admission control plane.

The lifecycle-tiering tranche is likewise bootstrap work while its routing files
are absent from `main`.

A bootstrap procedure must be independently reviewed/qualified. Only after that
exact generation lands on the target base can it become ordinary admission
policy for later candidates.

## Collector trust boundary

The pure evaluator cannot authenticate GitHub facts by itself.

A future trusted collector must independently obtain, from GitHub's API or an
equivalent trusted source:

- current target-base SHA;
- candidate head/tree identities;
- control-plane blob identities;
- exact workflow path/blob;
- workflow `run_id` and `run_attempt`;
- exact run head/base/event/status/conclusion;
- **all pages** of the job census for that attempt.

It must not accept candidate-provided census-completeness assertions.

The collector should remain read-only and base-owned. It should inspect existing
unprivileged candidate execution evidence rather than execute arbitrary
candidate code in a privileged context.

Do not solve this with a broad `pull_request_target` path that checks out and
runs candidate code with elevated credentials.

## Relationship to lifecycle tiering

CI lifecycle tiering answers:

```text
which verification work should run now?
```

Merge admission answers:

```text
what exact evidence is sufficient to integrate this exact candidate now?
```

They remain separate.

```text
draft Tier-1 green
    !=
full integration green
    !=
merge admission
```

Lifecycle scheduling may reduce expensive work on draft PRs. It cannot mint
merge authority.

## Relationship to focused scientific qualification

Focused theorem gates remain useful evidence, but v1 forbids:

```text
focused theorem evidence
    -> substitute for full repository integration
```

unless a later policy explicitly defines and qualifies such a substitution.

This preserves:

```text
scientific proposition qualification
    !=
repository integration qualification
    !=
merge authority
```

## Required adversarial corpus

The current pure-core test corpus includes at least:

1. exact current full success + complete manifest -> `ADMITTED`;
2. no full integration -> `INCOMPLETE`;
3. successful old head -> `STALE`;
4. successful old base -> `STALE`;
5. candidate changes workflow -> `BOOTSTRAP_REQUIRED`;
6. candidate changes policy -> `BOOTSTRAP_REQUIRED`;
7. candidate changes required-job manifest -> `BOOTSTRAP_REQUIRED`;
8. policy bytes differ from observed target-base policy blob -> `REJECTED`;
9. manifest bytes differ from observed target-base manifest blob -> `REJECTED`;
10. manifest binds wrong workflow generation -> `REJECTED`;
11. cancelled/in-progress workflow -> `INCOMPLETE`;
12. incomplete GitHub job census -> `INCOMPLETE`;
13. complete census missing a required family -> `INCOMPLETE`;
14. explicit required-job failure -> `REJECTED`;
15. required success job skipped -> `INCOMPLETE`;
16. declared `allowed_skip` family may skip;
17. complete census contains unmanifested job -> `REJECTED`;
18. census job matches multiple families -> `REJECTED`;
19. incomplete trusted manifest -> `INCOMPLETE`;
20. duplicate job IDs -> refuse malformed observation;
21. wrong repository -> `REJECTED`;
22. unknown observation fields -> refuse;
23. v1 cannot claim `enforcement_ready=true`;
24. different `run_attempt` -> different evidence/receipt identity;
25. changed job census -> different evidence identity;
26. changed manifest -> different manifest/receipt identity.

## Enforcement-ready exit gate

V1 must not be called repository-enforced merge authority until all are true:

1. the pure evaluator and schema corpus pass on its exact head;
2. a trusted collector independently obtains current base/head/tree/blob/run/job
   facts;
3. collector pagination proves the complete job census for exact
   `(run_id, run_attempt)`;
4. the required-job manifest is structurally derived/validated against the exact
   trusted workflow blob and flips to `complete=true`;
5. manifest matching/cardinality/disposition logic is independently qualified;
6. collector implementation identity is itself governed;
7. an admission result is published as a distinct check/status candidate code
   cannot mint;
8. repository rules require that trusted admission check before merge;
9. control-plane changes follow an independently reviewed bootstrap process;
10. exact-head and exact-base staleness is enforced;
11. lifecycle-skipped jobs cannot masquerade as successful required evidence;
12. no privileged workflow executes arbitrary candidate code with secrets.

Until then, receipts remain executable specifications of merge semantics rather
than enforcement claims.
