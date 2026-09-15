# CI Admission V1 — staged routing, trusted policy, and heavy-job admission

Status: staged only; no active workflow and no PR.

## Authority boundary

`CI Admission` is a routing decision, not a correctness or merge-qualification decision.
A green routing job may mean that a draft was correctly *not admitted* to heavyweight CI.
Therefore branch protection MUST NOT treat `CI Admission` alone as integration authority.

The stable merge authority remains an exact-head integration qualification receipt/check.

## V1-A — deterministic routing

The router emits routing-only dispositions:

- draft PR without `qualification-requested` -> `DraftNotQualificationEligible`;
- draft PR with the explicit label -> `QualificationPending`;
- ready/non-draft PR -> `QualificationPending`;
- `ready_for_review` -> `QualificationPending`;
- `converted_to_draft` or label removal on an unpromoted draft -> `DraftNotQualificationEligible`;
- `workflow_dispatch`, merge-group candidate, and push to `main` -> `QualificationPending`;
- weekly `schedule` -> `ScheduledMaintenanceOnly`;
- candidate modification of the CI/admission control plane -> `ControlPlaneReviewRequired`;
- unsupported PR actions fail closed as `UnsupportedEvent`;
- unrelated branch push / closed PR -> `QualificationNotRequested`.

`QualificationPending` means only “eligible to request heavyweight execution.” It is not PASS.
`ScheduledMaintenanceOnly` is a separate execution class and cannot satisfy qualification.
The router never emits `QualificationPassed` and always emits `merge_authority=false`.

Every disposition that can cause heavyweight or scheduled-maintenance work requires an exact Git
subject SHA and an exact trusted policy commit SHA. Invalid or absent object IDs fail closed.

## V1-B — trusted control-plane routing

A PR or merge-group candidate MUST NOT execute its own copy of the admission policy.
The rendered admission job therefore:

1. checks out the event subject with full history but without persisted credentials;
2. selects the PR/merge-group base SHA as `ADMISSION_POLICY_SHA`;
3. materializes `scripts/ci-admission-v1.py` with `git show` from that trusted base commit;
4. executes only that materialized policy script;
5. compares exact base and head Git trees with NUL-delimited `git diff --name-only --no-renames`;
6. emits `ControlPlaneReviewRequired` when the candidate changes `.github/workflows/**` or the
   admission-policy/census/generator scripts.

A control-plane-changing candidate therefore cannot self-admit the heavyweight matrix. It needs an
independent bootstrap/review path. This is an operational trust boundary, not a claim that the base
policy is correct.

## V1-C — exact source-bound job census and dependency cut

The census is bound to one exact `.github/workflows/ci.yml` Git blob and partitions all 34 current
top-level jobs into four disjoint classes:

- `cheap_static_candidate`: 4 jobs;
- `heavy_admitted`: 26 jobs;
- `heavy_or_maintenance`: 2 jobs (`test`, `psych-bench`);
- `event_special`: 2 jobs (`sbom`, `stress-tests`).

Any missing, renamed, duplicated, newly introduced, or multiply classified job fails closed.
The verifier also derives every top-level `needs:` edge and rejects dependency cuts that would make
an admitted job depend on a job unavailable in the same execution mode. Event-special jobs are
required to remain dependency-independent in V1.

This matters for scheduled execution: `psych-bench` currently needs `test`, so both are classified
`heavy_or_maintenance`. Weekly schedule can therefore run the default test + psych-bench chain
without admitting the other 26 heavyweight jobs. `stress-tests` remains independently schedule/
manual scoped. The prior design incorrectly made `schedule` heavyweight-eligible and would have
preserved a weekly full-matrix fan-out.

## V1-D — fail-closed renderer

The renderer accepts only the census-bound source workflow. It:

1. expands pull-request event types to include promotion/demotion/label transitions;
2. inserts one `CI Admission` job;
3. gates every `heavy_admitted` job on `heavy_eligible`;
4. gates every `heavy_or_maintenance` job on `heavy_eligible || maintenance_eligible`;
5. preserves existing scalar, inline-list, or simple block-list `needs:` dependencies while adding
   `admission`;
6. composes existing job-level `if:` predicates with the appropriate admission predicate;
7. leaves `event_special` jobs independent;
8. proves rendered jobs equal the exact source census plus the new admission job;
9. proves gate-expression counts equal the classified job counts;
10. fails closed on source-blob drift, unsupported `needs:`/`if:` syntax, duplicate jobs, or census
    drift.

The rendered candidate is not active until it receives GitHub workflow-parser validation and
focused live event tests.

## Two-phase activation

Activation is intentionally split so the candidate cannot bootstrap its own trust root.

### Phase 1 — land policy bytes only

Land the routing policy, census, verifier, renderer, and this contract without modifying any active
workflow. This makes the trusted admission script available on `main` first.

### Phase 2 — activate `ci.yml`

Render the exact reviewed `ci.yml` source only after Phase 1 is on `main`. The activation PR changes
the workflow control plane, so the already-trusted base policy must classify that PR as
`ControlPlaneReviewRequired`; it must not self-admit the heavyweight matrix. Validate that exact
activation head through an independent workflow-syntax/bootstrap path before merge.

Only later non-control-plane PRs may use normal V1 self-routing from trusted base policy.

## Why V1 supersedes V0

V0's draft-global concurrency group was an emergency rate limiter: drafts still instantiated the
heavy workflow and were merely serialized. V1 removes heavyweight admission itself.

Keeping the repository's existing per-ref cancellation then becomes preferable:

- a new commit supersedes older work on the same PR;
- `converted_to_draft` can cancel an older promoted run and replace it with cheap-only work;
- removing `qualification-requested` from a draft can do the same;
- unrelated draft PRs no longer contend through one artificial global FIFO because they only run
  the cheap routing/static surface.

V0 remains historical/fallback design evidence rather than something to layer on top of V1.

## Governance

After stable final integration checks exist, #2485 may bind branch protection to exact integration
qualification. Do not use routing-only `CI Admission` as the sole required merge context.

The V1 control plane currently governs `.github/workflows/ci.yml`; other workflow families remain
outside this admission theorem and can still contribute queue pressure. Their admission treatment
must be separately censused rather than assumed.

## Claims

This staging branch establishes deterministic local policy, exact-source job census, dependency-cut
checks, trusted-base policy materialization design, and a fail-closed renderer with self-tests. It
does not establish GitHub workflow syntax, live event behavior, queue reduction, branch protection,
source correctness, scientific validity, or qualification PASS.
