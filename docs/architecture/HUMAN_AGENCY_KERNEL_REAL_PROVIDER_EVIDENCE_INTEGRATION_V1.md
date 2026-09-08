# HAK-011 — Real Provider Evidence Integration v1

Status: audit-only integration candidate. No runtime authority changes.

## Purpose

HAK-007 through HAK-010 define evidence subjects, terminal receipts, bounded interpretation, provider-bound per-check evidence, and precommitted check selectors. HAK-011 asks whether those semantics survive contact with a real provider execution rather than synthetic fixtures.

The first real subject is the completed GitHub Actions run:

```text
repository   = Luminous-Dynamics/symthaea
workflow     = HAK Evidence
run          = 34225891059
attempt      = 1
subject      = 227442e681effa9a0bb08bace7f17b76ca910493
conclusion   = cancelled
job          = 102059802264 / HAK Evidence Linter
provider steps reported = 0
```

This is intentionally a negative execution. It is useful because it tests whether HAK preserves failure without fabricating missing evidence.

## Core semantic result

```text
TerminalFailure != MissingEvidence
```

but also:

```text
PlanNotSatisfied != ClaimDisproven
```

For this real run:

```text
TerminalQualificationReceipt.conclusion = cancelled
PlanConformance.status                  = NotSatisfied
PerCheckProviderEvidence                = []
ClaimInterpretation.status              = InsufficientEvidence
```

The run did not complete the qualification plan. That fact is real evidence. It does not establish that HAK-007's semantic claims are false.

## No-step theorem

GitHub reported the terminal job as cancelled and returned no step records for the job.

Therefore HAK-011 requires:

```text
ProviderSteps = ∅
        =>
ProviderBoundCheckEvidence = ∅
AND every required check = Missing
AND every required negative case = Missing
```

The evidence system must not reconstruct a plausible-looking Passed check from workflow source, expected step names, logs that do not exist, or knowledge of what the job would have done.

## Materialized artifacts

The real evidence directory contains:

```text
hak007-run-34225891059.receipt.json
hak007-run-34225891059.conformance.json
hak007-run-34225891059.interpretation.json
hak007-run-34225891059.capsule.json
```

The capsule records the provider run/job snapshot observed through the connected GitHub integration and binds the canonical digests of the three HAK artifacts.

It is not a cryptographic GitHub attestation.

```text
MaterializedProviderSnapshot != CryptographicAttestation
```

## Claim semantics

The plan-conformance record is `NotSatisfied` because the terminal receipt is non-success.

The evidence interpretation is deliberately `InsufficientEvidence`, not `NotSatisfied`:

```text
execution failed to complete
!=
semantic claim falsified
```

This distinction matters for CI cancellation, infrastructure failure, provider outage, runner eviction, and other execution failures that do not test the underlying proposition.

## Composition with HAK-009

The real capsule contains no per-check ProviderBound evidence records because GitHub reported zero steps.

HAK-009 strict conformance is still exercised. It accepts the honest absence because no conformance obligation is marked `Passed`.

Thus:

```text
MissingCheck -> no check evidence required
PassedCheck  -> exact provider evidence required
```

This prevents both evidence fabrication and evidence laundering.

## HAK-011 self-qualification

HAK-011 has its own E5-target qualification plan and HAK-010 binding policy. The binding policy precommits the exact workflow job/step that may satisfy each HAK-011 required check and negative case.

The focused `HAK Real Provider Evidence` workflow validates:

- exact-head checkout,
- inherited HAK tooling compilation,
- capsule schema,
- HAK-011 plan,
- HAK-011 precommitted binding policy,
- the real cancelled-run capsule,
- focused adversarial regressions.

A green HAK-011 workflow qualifies only these integration-tooling claims. It does not retroactively qualify HAK-007, whose historical run was cancelled.

## Non-claims

HAK-011 does not:

- claim the cancelled HAK-007 run passed,
- treat cancellation as semantic falsification,
- fabricate provider step evidence,
- authenticate GitHub metadata cryptographically,
- grant runtime authority,
- certify governance, consent, robotics, or safety behavior,
- promote provider success into truth,
- convert model output into evidence authority.

## Next boundary

Once HAK-011 has a successful exact-head run, the next useful exercise is to materialize that successful run using the same evidence model.

Then the comparison is controlled:

```text
same evidence semantics
different provider conclusion
```

If successful and cancelled runs require different evidence rules, the model is wrong. Only their observed outcomes and resulting conformance/interpretation should differ.
