# Human Agency Kernel — Provider-Bound Check Evidence v1

Status: HAK-009 architecture candidate / documentation only.

Parent stack:

- HAK-007 separates plan, subject, execution, receipt and interpretation;
- HAK-008 separates terminal receipt, plan conformance and bounded claim interpretation;
- HAK-009 closes the next gap: a conformance record must not treat a free-form `status: Passed` plus an arbitrary `evidence_ref` as verified check evidence.

## 1. Core theorem

```text
CheckAssertion != ProviderBoundCheckEvidence
ProviderBoundCheckEvidence != CryptographicAttestation
CryptographicAttestation != SemanticClaimTruth
```

HAK-009 therefore does not introduce one universal `VerifiedCheck` boolean.

It distinguishes the provenance class of the evidence.

## 2. Why HAK-009 exists

HAK-008 can establish that a conformance record contains every required plan check and every required negative case.

But a record may still say:

```text
check_id = "regressions"
status = Passed
evidence_refs = ["some:string"]
```

without proving that the referenced execution actually ran the check.

Therefore:

```text
CompleteConformanceBookkeeping
!=
ProviderEstablishedCheckExecution
```

A `Satisfied` conformance used for stronger hosted qualification should eventually consume exact check evidence rather than self-asserted statuses.

## 3. Provider-bound check evidence

`ProviderBoundCheckEvidenceV1` binds one qualification-plan obligation to one exact hosted execution lineage.

Conceptually:

```text
ProviderBoundCheckEvidenceV1 {
    evidence_id
    assurance_class = ProviderBound

    subject {
        repository
        commit_sha
    }

    plan {
        plan_ref
        plan_digest
    }

    qualification_receipt {
        receipt_id
        receipt_digest
    }

    execution {
        provider
        run_id
        run_attempt
        workflow_id
        workflow_path
    }

    obligation {
        kind        // RequiredCheck | NegativeCase
        id
    }

    provider_binding {
        job_id
        job_name
        job_status
        job_conclusion

        step_number
        step_name
        step_status
        step_conclusion

        provider_job_ref
    }

    collected_by
    observed_at
    evidence_digest
}
```

The record must bind the same subject, plan, receipt, provider run, run attempt and workflow as the terminal qualification receipt.

## 4. Obligation identity is precommitted

A provider step is not evidence for an arbitrary claim merely because its name looks relevant.

HAK-009 requires:

```text
obligation.id exists in exact QualificationPlan
```

and the obligation kind must agree with the plan section:

```text
RequiredCheck -> plan.required_checks[].check_id
NegativeCase  -> plan.required_negative_cases[]
```

A later reviewer may not silently relabel an unrelated successful step as evidence for a missing plan obligation.

## 5. Step identity and plan binding

The strongest future form should precommit provider selectors before execution, for example:

```text
check_id
-> workflow_path
-> job identity
-> step identity
-> allowed terminal conclusions
```

HAK-009 v1 treats this as a separate `CheckEvidenceBindingPolicy` rather than silently extending HAK-007's plan semantics after the fact.

Candidate theorem:

```text
PostHocStepSelection != PrecommittedCheckBinding
```

A provider-bound record created without a precommitted selector may still be useful execution evidence, but cannot be upgraded retroactively into the stronger precommitted class.

## 6. Provider-bound does not mean cryptographically attested

Ordinary GitHub Actions run/job/step API data is provider-linked evidence.

HAK-009 calls this:

```text
ProviderBound
```

not:

```text
CryptographicallyAttested
```

The distinction matters because a materialized API record is only as trustworthy as its provider provenance, collection process and preservation semantics.

A static HAK linter can verify internal joins and digests; it cannot manufacture a GitHub signature over historical API metadata.

## 7. Optional attested qualification bundle

For stronger preservation/release evidence, HAK-009 defines a separate optional concept:

```text
AttestedQualificationBundle
```

A qualification workflow may emit a bundle containing, for example:

- exact subject identity;
- plan identity/digest;
- terminal qualification receipt;
- provider-bound check evidence records;
- plan-conformance record;
- interpretation record;
- hashes of relevant logs/reports/artifacts.

The bundle receives its own artifact digest and may be signed using an external artifact-attestation system.

For GitHub Actions, current GitHub artifact attestations use Sigstore/OIDC provenance and can bind an artifact to repository, workflow, commit and triggering event.

HAK's interpretation remains:

```text
VerifiedArtifactAttestation
-> stronger provenance for bundle bytes/origin
-/-> bundle's semantic claims are automatically true
```

Attestation verification is required before an attested bundle is treated as attested.

## 8. Assurance classes

HAK-009 v1 proposes at least:

```text
Asserted
ProviderBound
AttestedBundle
```

### Asserted

The conformance record contains a status/evidence reference but no provider-established join.

### ProviderBound

The obligation is bound to exact hosted provider execution metadata and the exact HAK-007 receipt lineage.

### AttestedBundle

The preserved qualification bundle has cryptographically verified provenance in addition to its internal HAK joins.

These classes describe provenance strength, not safety truth.

## 9. Conformance consumption rule

A future strict conformance profile should require every `Passed` obligation used for E5 hosted qualification to reference a matching provider-bound check-evidence record.

Conceptually:

```text
Passed(obligation)
+ target >= E5
-> ProviderBoundCheckEvidence(obligation)
```

For a release profile requiring stronger preservation:

```text
ProviderBoundChecks
+ exact bundle
+ verified artifact attestation
-> AttestedQualificationBundle
```

The exact tier policy remains domain-owned.

## 10. Failure evidence

A failed provider step remains evidence.

```text
ProviderStepConclusion = failure
```

must not be rewritten into missing evidence.

It may establish:

```text
NotSatisfied
```

or an infrastructure-specific state depending on the precommitted plan.

The provider record and interpretation remain separate.

## 11. Collection and preservation

HAK-009 distinguishes:

```text
LiveProviderRecord
MaterializedProviderRecord
AttestedBundle
```

A live provider API can disappear or change retention state.

A materialized record preserves what was observed but requires collector provenance.

An attested bundle strengthens tamper/provenance evidence for the preserved bytes but still does not establish semantic adequacy.

## 12. No authority creation

Check evidence is evidence about qualification.

It is not operational authority.

```text
ProviderBoundCheckEvidence
!= RuntimePermission

AttestedQualificationBundle
!= RuntimePermission
```

Any later runtime consumer must still apply the HAK authority/rights/domain-policy layers.

## 13. HAK-009 implementation plan

A narrow implementation tranche should add:

1. `provider-bound-check-evidence-v1.schema.json`;
2. an audit-only check-evidence linter;
3. exact plan/subject/receipt/execution joins;
4. obligation membership checks;
5. evidence-record canonical digest;
6. negative tests for run/attempt/subject/plan/receipt/step mismatch;
7. a precommitted binding-policy format before claiming strong check-to-step qualification;
8. later optional qualification-bundle generation and artifact attestation for release evidence.

Do not modify HAK-008's semantics retroactively to pretend its earlier free-form `evidence_refs` were provider-bound.

## 14. Non-claims

HAK-009 does not claim:

- GitHub API metadata is cryptographically signed step evidence;
- a GitHub artifact attestation proves an artifact is safe;
- every CI run should produce artifact attestations;
- provider-bound evidence is independent certification;
- a successful step establishes the adequacy of the qualification plan;
- an evidence artifact grants runtime authority;
- Symthaea or an AI model may certify its own evidence simply by interpreting it.
