# Merge Admission Receipt v2

## Status

V2 is a staged, queue-neutral successor to the frozen v1 wire contract.

It remains:

```text
enforcement_ready = false
```

and must not be represented as repository-enforced merge authority.

The active staging branch is:

```text
governance/merge-admission-receipt-v2
```

No pull request or Actions run is required to exercise the pure policy semantics.

## Why v2 exists

V1 correctly separated CI evidence from merge authority, but its required-job boundary contained one overloaded assertion:

```text
job_set_complete = true
```

That phrase can mean two different things:

```text
A. the collector retrieved every job GitHub reported for a run

B. those jobs satisfy the complete base-owned qualification manifest
```

A does not imply B.

A weakened workflow may have a perfectly complete API census while omitting work that policy intended to require.

V2 therefore freezes:

```text
complete GitHub census
    !=
required qualification manifest satisfied
```

and moves manifest satisfaction into the policy core rather than trusting a collector-supplied boolean.

## V1 remains frozen

The v1 observation/receipt contract is not silently rewritten.

```text
v1 bytes + semantics
    remain v1

new wire semantics
    -> v2
```

This is the same exact-contract discipline expected of cryptographic or authority-bearing wire formats.

## V2 observation model

A full-integration observation now carries:

```text
workflow_path
workflow_blob_sha
run_id
run_attempt
event
status
conclusion
head_sha
base_sha
job_census_complete
job_census[]
```

Each job census entry carries:

```text
job_id
name
status
conclusion
skipped
```

### Run attempt is part of evidence identity

GitHub may rerun the same workflow run ID.

Therefore:

```text
run_id
    !=
exact execution attempt
```

V2 binds:

```text
(run_id, run_attempt)
```

into the evidence receipt.

A rerun is new evidence even if every other field is unchanged.

## Complete census is its own proof

The collector must independently establish:

```text
job_census_complete == true
```

This means the external collector has completed pagination/enumeration of the workflow run's job surface.

It does **not** mean those jobs are sufficient for qualification.

If this proof is absent:

```text
-> INCOMPLETE
```

## Required-job manifest

The base-owned manifest is:

```text
scripts/ci/required_ci_job_manifest_v1.json
```

Its schema is:

```text
scripts/ci/required_ci_job_manifest_v1.schema.json
```

The manifest binds itself to one exact workflow generation through:

```text
workflow_path
workflow_blob_sha
```

and defines event-specific profiles.

Each profile contains one family per top-level workflow job:

```text
job_id
api_name_regex
min_instances
max_instances
required_disposition
```

The current staged manifest deliberately has:

```text
complete = false
```

and empty profile bodies.

That means the real repository policy cannot produce an admitted result from this manifest generation.

This is intentional. The manifest must be audited against the exact workflow and its dynamic matrix cardinalities before `complete` can become true.

## Structural manifest validator

`scripts/ci/validate_required_ci_job_manifest.py` validates the manifest against the exact workflow bytes and v2 policy.

For a qualification-complete manifest it requires:

```text
manifest.workflow_blob_sha
    == Git blob SHA(exact workflow bytes)

manifest event profiles
    == policy accepted events

profile top-level job IDs
    == exact workflow top-level job IDs

family job IDs
    == exact workflow top-level job IDs
```

An incomplete manifest fails by default.

The option:

```text
--allow-incomplete
```

exists only so an unfinished scaffold can be structurally audited.

Its output still states:

```text
qualification_eligible = false
```

It is not a qualification bypass.

## Manifest satisfaction belongs to the policy core

V2 does not accept a field such as:

```text
required_job_manifest_match = true
```

from the collector.

Instead:

```text
trusted collector
    -> complete raw job census

base-owned manifest
    -> expected families/cardinalities/dispositions

policy core
    -> independently performs matching
```

This removes an authority-bearing boolean from the collector wire format.

## API-visible job families

Matrix expansion means a top-level YAML job is not necessarily one GitHub job row.

For example, conceptually:

```text
test-all-features
    -> Test CI-safe (core-infra)
    -> Test CI-safe (core-media)
    -> ...
```

The manifest therefore specifies API-visible name regexes and instance bounds rather than pretending every YAML job produces exactly one API row.

All regexes must be explicitly anchored:

```text
^...$
```

and a census job must match exactly one family.

Unknown or ambiguously classified jobs fail closed.

## Cardinality is semantic evidence

For each family:

```text
min_instances <= observed_instances <= max_instances
```

must hold.

A complete census with a missing matrix leg is therefore not qualification-complete.

Cardinality mismatch produces:

```text
INCOMPLETE
```

rather than pretending a test failed.

## Required dispositions

V2 supports exactly two family dispositions:

```text
success
allowed_skip
```

`success` requires each matched instance to be completed successfully and not skipped.

`allowed_skip` exists for jobs whose base-owned event predicate legitimately causes a skipped job on a particular event profile.

An explicit test failure remains:

```text
REJECTED
```

while pending/cancelled/skipped required evidence remains:

```text
INCOMPLETE
```

## Job identity

V1 rejected duplicate required-job names.

That is too strong for a generic matrix-aware collector because display names are not the durable API identity boundary.

V2 uses the positive GitHub job ID as the uniqueness key.

```text
duplicate job_id
    -> REJECTED
```

Display names are inputs to manifest family matching, not primary evidence identity.

## Order-independent census binding

Pagination/API ordering must not alter receipt identity for the same evidence set.

V2 canonicalizes the job census by:

```text
(job_id, name)
```

before hashing it.

The receipt binds:

```text
job_census_count
job_census_sha256
required_job_observation_count
required_jobs_sha256
```

Thus reordering the same census does not create a different evidence identity, while changing a job/result does.

## Policy self-binding

V2 adds:

```text
policy_path
```

The exact policy bytes loaded by the evaluator are Git-blob hashed and compared to the target-base blob observed for that governed path.

Therefore:

```text
loaded policy bytes
    != target-base policy blob
        -> REJECTED
```

The receipt separately binds:

```text
policy_sha256
policy_blob_sha
```

This is provenance hardening, not self-authentication: a candidate-owned evaluator still cannot grant itself trusted repository authority.

## Manifest self-binding

The exact required-job manifest bytes supplied to the policy core are also Git-blob hashed.

V2 requires:

```text
loaded manifest blob
    == target-base manifest blob

manifest.workflow_path
    == policy workflow path

manifest.workflow_blob_sha
    == target-base workflow blob
```

Any mismatch is rejected.

This prevents a permissive local manifest from being substituted for the governed base manifest.

## Control-plane equivalence remains mandatory

Ordinary product admission still requires exact candidate/base equivalence for every governed control-plane path.

Any change produces:

```text
BOOTSTRAP_REQUIRED
```

before ordinary full-integration evidence can authorize the candidate.

V2's governed surface includes the lifecycle scheduler, workflow, v2 policy/evaluator/schemas, required-job manifest/schema/validator, and this normative contract.

## Evidence-binding structure

A v2 receipt content-binds, at minimum:

```text
repository
target branch
current base SHA
candidate head SHA
candidate tree SHA
policy SHA-256
policy Git blob SHA
control-plane base/candidate blobs
workflow path/blob
run ID
run attempt
event
run status/conclusion
qualification head/base
job census completeness
canonical full census digest
required subset digest
required-job manifest SHA-256
required-job manifest Git blob SHA
decision
reasons
```

The resulting receipt is still unsigned.

## Five dispositions remain

V2 preserves the v1 distinction:

```text
ADMITTED
INCOMPLETE
STALE
BOOTSTRAP_REQUIRED
REJECTED
```

In particular:

```text
cancelled != failed
missing matrix leg != failed test
old successful run != current proof
control-plane change != ordinary product failure
```

## Collector boundary after v2

The external collector is now simpler and less authoritative.

It must establish facts such as:

```text
repository/base/head/tree identity
control-plane blob identities
workflow run identity
run attempt
complete job pagination
raw job rows
```

It does not decide whether the job set satisfies merge policy.

That decision belongs to the base-owned manifest + pure evaluator.

## Remaining enforcement gap

V2 still deliberately does not solve repository enforcement.

The future chain remains:

```text
base-owned collector
        -> exact observation
base-owned policy + manifest
        -> pure disposition
trusted external check producer
        -> admission check
repository ruleset / required check
        -> merge authority
```

Until that chain exists and is independently qualified:

```text
ADMITTED receipt
    !=
GitHub-enforced permission to merge
```

## V2 exit gate

Do not mark v2 enforcement-ready until all of the following hold:

1. the exact staged evaluator corpus passes;
2. the exact manifest-validator corpus passes;
3. the required-job manifest is audited and `complete=true`;
4. each accepted event profile exactly covers the trusted workflow generation;
5. matrix cardinalities are derived and frozen from the exact workflow semantics;
6. API-visible job-name patterns are verified against observed GitHub job rows;
7. a trusted external collector proves complete pagination for an exact run attempt;
8. collector implementation identity is governed independently of candidate code;
9. the admission result is published by a principal candidate code cannot impersonate;
10. repository rules require that exact admission check;
11. control-plane updates use an independent bootstrap procedure;
12. no privileged `pull_request_target` path executes arbitrary candidate code.

Until then, v2 is a stronger executable authority specification, not an enforcement claim.
