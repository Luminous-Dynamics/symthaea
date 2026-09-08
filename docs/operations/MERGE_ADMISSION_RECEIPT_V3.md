# Merge Admission Receipt v3

## Purpose

V3 separates four questions that must not be collapsed:

```text
Did GitHub expose the complete run/job evidence?
Did the trusted base-owned manifest require and classify that evidence correctly?
Did the candidate pass those required gates?
Does any external repository rule currently grant merge authority from that disposition?
```

The answer to the fourth question is still **no**. V3 remains an executable policy core with `enforcement_ready=false`.

## Core theorem

```text
candidate-owned CI result
    !=
complete evidence census
    !=
trusted manifest satisfaction
    !=
merge authority
```

The collector reports facts. It does not report `required_jobs_satisfied=true`.

## Seven dispositions

```text
ADMITTED
    all v3 predicates over the supplied trusted observation hold

INCOMPLETE
    required evidence is genuinely absent, pending, cancelled, or not fully paginated

STALE
    evidence belongs to a prior candidate head or prior target base

BOOTSTRAP_REQUIRED
    the candidate changes the authority/control plane itself

COLLECTOR_INVALID
    the supplied observation is internally contradictory or incomplete in a way
    the collector claims is complete

CONTROL_PLANE_INVALID
    the trusted base policy/workflow/manifest are absent, byte-mismatched, or
    internally inconsistent

REJECTED
    candidate/run identity is wrong or an explicit required gate failed
```

This preserves:

```text
not yet known
    != stale proof
    != candidate changed the judge
    != collector contradicted itself
    != judge/configuration is broken
    != candidate failed a required test
```

## Exact run-attempt identity

GitHub reruns may reuse a workflow `run_id` while incrementing `run_attempt`.
Therefore evidence identity is at least:

```text
(run_id, run_attempt)
```

Both values are bound into `evidence_binding_sha256` and the final receipt.

## Job-census proof

V2 used a boolean `job_census_complete`. V3 replaces that with an explicit proof object:

```text
job_census_proof:
    reported_total_count
    pages_fetched
    terminal_page_observed
```

The pure evaluator derives:

```text
terminal page observed
AND
len(job_census) == reported_total_count
```

If the terminal page has not yet been observed, the result is `INCOMPLETE`.
If the collector claims terminal completion but the count disagrees, the result is
`COLLECTOR_INVALID`.

The collector remains a future trust boundary; these fields do not cryptographically
prove GitHub API authenticity. They make the claim explicit and internally checkable.

## Trusted required-job manifest

The required-job manifest is base-owned and content-bound to the exact trusted workflow
blob. The currently staged manifest names:

```text
.github/workflows/ci.yml
blob a48366076b30eb8e12d22c927a3b8bf333181409
```

which is the exact `ci.yml` blob on the reviewed main generation.

The staged production manifest intentionally remains:

```text
complete: false
```

Therefore the staged production policy cannot produce `ADMITTED` yet. This is an
intentional fail-closed state, not missing bookkeeping.

A complete profile must provide exactly one family for every top-level workflow job.
Each family binds:

```text
job_id
API-visible name regex
minimum instances
maximum instances
required disposition
```

The runtime job census is closed-world:

```text
unmanifested job       -> CONTROL_PLANE_INVALID
ambiguous family match -> CONTROL_PLANE_INVALID
missing required family instance -> INCOMPLETE
extra family instances -> CONTROL_PLANE_INVALID
explicit required failure -> REJECTED
```

## Source proof and runtime proof are separate

A complete manifest must satisfy both:

```text
SOURCE PROOF
manifest.workflow_blob_sha == Git blob(workflow bytes)
manifest top-level job IDs == exact workflow top-level job IDs

RUNTIME PROOF
complete GitHub job census
        ↓
closed-world family matching
        ↓
cardinality + disposition checks
```

Neither proof substitutes for the other.

## Control-plane self-governance

For ordinary candidate admission, all governed files must be byte-identical to the
current target-base generation. The governed surface includes the trusted workflow,
lifecycle rules, required-job manifest/schema, V3 evaluator/policy/schemas, and the
normative manifest matcher.

```text
candidate changes any governed byte
        -> BOOTSTRAP_REQUIRED
```

The candidate may not change its judge and then use that changed judge to authorize
itself.

## Trusted workflow bytes are rechecked

V3 loads the trusted workflow bytes and computes the Git blob identity itself. It does
not rely only on a collector-supplied workflow SHA.

The evaluator also validates the required-job manifest against those exact workflow
bytes before considering candidate evidence.

A broken base-owned policy/manifest/workflow relationship yields:

```text
CONTROL_PLANE_INVALID
```

not `REJECTED`, because that is not evidence that the candidate failed.

## Collector consistency

V3 treats these as collector faults rather than candidate failures:

- duplicate GitHub job IDs;
- `skipped` flag contradicts the job conclusion;
- non-completed job already has a conclusion;
- completed job lacks a conclusion;
- claimed terminal census count differs from the observed census length;
- missing/extra control-plane observation rows.

## Evidence binding

The receipt binds at least:

```text
policy bytes
candidate head + tree
current target base
control-plane blob identities
workflow blob
run_id + run_attempt
run status/conclusion
target head/base identities
job-census proof
canonical full job-census digest
canonical required-job subset digest
required-job manifest digest + blob
final disposition + reasons
```

Ordering the same census differently does not change its digest; changing the actual
job evidence does.

## No enforcement claim

V3 still emits an unsigned local disposition with:

```text
enforcement_ready = false
```

An enforceable deployment still needs all of:

1. a trusted collector that obtains GitHub facts independently of candidate code;
2. qualification of the complete required-job manifest against the exact workflow;
3. qualification of the collector/evaluator implementation itself;
4. a distinct base-owned admission check/status;
5. repository rules requiring that check before merge;
6. an independent bootstrap path for changes to this authority machinery.

Until those exist:

```text
ADMITTED receipt
    !=
permission to merge
```

## Relationship to Tier Q and lifecycle tiering

```text
Tier Q / focused theorem
    asks: did this narrow proposition qualify?

CI lifecycle tiering
    asks: which verification should consume runner capacity now?

Merge admission
    asks: what exact current evidence is sufficient for integration?
```

They remain separate authority planes.

## Required V3 adversarial corpus

At minimum:

1. exact current complete success -> `ADMITTED`;
2. no terminal jobs page yet -> `INCOMPLETE`;
3. terminal census count contradiction -> `COLLECTOR_INVALID`;
4. duplicate job ID -> `COLLECTOR_INVALID`;
5. skipped flag/conclusion contradiction -> `COLLECTOR_INVALID`;
6. incomplete trusted manifest -> `CONTROL_PLANE_INVALID`;
7. manifest/workflow source drift -> `CONTROL_PLANE_INVALID`;
8. wrong loaded workflow bytes -> `CONTROL_PLANE_INVALID`;
9. candidate control-plane drift -> `BOOTSTRAP_REQUIRED`;
10. old head/base -> `STALE`;
11. explicit required-job failure -> `REJECTED`;
12. pending required job -> `INCOMPLETE`;
13. unmanifested runtime job -> `CONTROL_PLANE_INVALID`;
14. rerun attempt changes receipt identity;
15. census ordering does not change evidence identity.

## Governing principle

V3 follows the same epistemic discipline as the spatial/world-model work:

```text
observation != evidence != belief != authority

and here:

workflow run != complete census != manifest-qualified evidence != merge authority
```
