# HAK-011 — Real Provider Evidence Integration v1

Status: audit-only integration candidate. No runtime authority changes.

## Purpose

HAK-007 through HAK-010 define evidence subjects, terminal receipts, bounded interpretation, provider-bound per-check evidence, and precommitted check selectors. HAK-011 tests whether those semantics survive contact with a real provider execution rather than synthetic fixtures.

The first real subject is GitHub Actions run `34225891059`, attempt 1, for exact subject `227442e681effa9a0bb08bace7f17b76ca910493`. The run completed with conclusion `cancelled`. Its sole job, `102059802264 / HAK Evidence Linter`, also completed `cancelled`, and GitHub reported zero step records.

```text
TerminalFailure != MissingEvidence
PlanNotSatisfied != ClaimDisproven
NoProviderSteps != PermissionToInferPassedChecks
```

## Provider timing scopes

HAK-011 preserves run and job timing separately because they are not the same provider fact.

```text
RunTerminalObservation != JobCompletionTimestamp
```

For the historical run, the materialized provider snapshot records:

```text
run_created_at  = 2026-09-08T12:24:24Z
run_started_at  = 2026-09-08T12:24:24Z
run_updated_at  = 2026-09-08T12:25:21Z

job.created_at   = 2026-09-08T12:24:26Z
job.started_at   = 2026-09-08T12:24:26Z
job.completed_at = 2026-09-08T12:25:20Z
```

The HAK-007 receipt's `provider_started_at` and `provider_completed_at` bind the run-level observation (`run_started_at` and `run_updated_at`). Exact job timing remains separately preserved in the real provider capsule.

The integration validator requires:

```text
job.created_at <= job.started_at <= job.completed_at <= run_updated_at
```

Changing one timing scope does not silently rewrite another.

## Real evidence chain

The materialized artifacts bind:

```text
Provider snapshot
        ↓
TerminalQualificationReceipt(cancelled)
        ↓
PlanConformance(NotSatisfied)
        ↓
EvidenceInterpretation(InsufficientEvidence)
```

This is intentionally a negative execution. The workflow did not complete its qualification plan. That fact is real evidence; it does not establish that HAK-007's semantic claims are false.

## No-step theorem

GitHub reported the terminal job as cancelled and returned `steps = []`.

Therefore:

```text
ProviderSteps = ∅
        =>
ProviderBoundCheckEvidence = ∅
AND every required check = Missing
AND every required negative case = Missing
```

The system must not reconstruct `Passed` checks from workflow source, expected step names, hypothetical execution, or missing logs.

## Materialized artifacts

`docs/architecture/hak/evidence/real/` contains:

```text
hak007-run-34225891059.receipt.json
hak007-run-34225891059.conformance.json
hak007-run-34225891059.interpretation.json
hak007-run-34225891059.capsule.json
```

The capsule records provider run/job metadata observed through the connected GitHub integration and binds the canonical digests of the other HAK artifacts.

```text
MaterializedProviderSnapshot != CryptographicAttestation
```

## Conformance versus claim interpretation

The plan-conformance record is `NotSatisfied` because the qualification receipt is non-success.

The claim interpretation is `InsufficientEvidence`, not semantic `NotSatisfied`:

```text
execution failed to complete
!=
underlying proposition falsified
```

This is important for cancellation, provider outage, runner eviction, infrastructure failure, and other conditions that prevent a planned test from being executed.

## Composition with HAK-009

The historical capsule contains no per-check ProviderBound evidence because the provider reported no steps. HAK-009 strict conformance is still applied with an empty provider-evidence set. That is valid only because every obligation remains `Missing`.

```text
MissingCheck -> no fabricated provider evidence
PassedCheck  -> exact provider evidence required
```

## HAK-011 self-qualification

HAK-011 has its own E5-target qualification plan and HAK-010 precommitted binding policy. Its focused `HAK Real Provider Evidence` workflow binds exact-head checkout, compilation, schema validation, plan lint, policy lint, real-capsule validation, and regressions to fixed job/step selectors.

A green HAK-011 workflow qualifies only the HAK-011 integration-tooling claims. It does not retroactively qualify the cancelled HAK-007 execution.

## Non-claims

HAK-011 does not:

- claim the historical cancelled run passed,
- treat cancellation as semantic falsification,
- fabricate provider step evidence,
- authenticate GitHub metadata cryptographically,
- grant runtime authority,
- certify governance, consent, rescue, or actuator behavior,
- promote provider success into truth,
- convert model output into evidence authority.

## Next boundary

After a successful exact-head HAK-011 run exists, materialize it using the same evidence model and compare it against the cancelled run:

```text
same evidence semantics
different provider conclusion
```

If successful and cancelled executions require different provenance rules, the evidence model is wrong. Only observed outcomes, conformance, and interpretation should differ.
