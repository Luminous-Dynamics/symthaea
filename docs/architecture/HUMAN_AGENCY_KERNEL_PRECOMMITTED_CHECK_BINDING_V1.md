# HAK-010 — Precommitted Check Evidence Binding v1

Status: architecture + audit-only tooling candidate. No runtime authority changes.

## Purpose

HAK-009 binds a plan obligation to exact provider job/step evidence after execution. HAK-010 closes the remaining selection boundary: the provider source allowed to satisfy each obligation must itself be fixed before the execution whose result will be interpreted.

```text
PostHocStepSelection != PrecommittedCheckBinding
```

The target chain becomes:

```text
QualificationPlan
        +
CheckEvidenceBindingPolicy
        ↓
Exact Hosted Execution
        ↓
Terminal Qualification Receipt
        ↓
ProviderBoundCheckEvidence
        ↓
Precommit-Validated Check Evidence
        ↓
Plan Conformance
        ↓
Bounded Interpretation
```

## Binding policy

`CheckEvidenceBindingPolicyV1` is bound to the canonical qualification-plan content and workflow. It must cover exactly every required check and required negative case in the plan.

Each obligation selector fixes:

- obligation kind and identity,
- provider job name,
- provider step name,
- provider step number,
- allowed terminal job conclusions,
- allowed terminal step conclusions.

The policy has a canonical domain-separated digest over its content excluding the digest field itself.

## Core invariants

### Exact obligation coverage

```text
Bindings = RequiredChecks ∪ RequiredNegativeCases
MissingBindings = ∅
UnknownBindings = ∅
DuplicateBindings = ∅
```

A policy with incomplete or surplus semantic coverage is invalid even if every listed selector is individually well formed.

### Exact plan binding

```text
Policy.plan_id == QualificationPlan.plan_id
Policy.plan_digest == Digest(exact QualificationPlan content)
Policy.workflow_path == QualificationPlan.scope.workflow_path
```

This avoids a Git-commit self-reference while still fixing the exact plan semantics before execution.

### Precommitted provider selector

For provider-bound evidence `E` to satisfy obligation `O`:

```text
E.obligation == O
E.job_name == Binding(O).job_name
E.step_name == Binding(O).step_name
E.step_number == Binding(O).step_number
E.job_conclusion ∈ Binding(O).accepted_job_conclusions
E.step_conclusion ∈ Binding(O).accepted_step_conclusions
```

A different successful job or step cannot be substituted after seeing the run result.

### Composition with HAK-009

HAK-010 does not replace HAK-009. Provider evidence must first satisfy HAK-009 subject, plan, receipt, execution, terminal-job and digest joins. HAK-010 then adds the precommit selector constraint.

```text
ProviderBound != PrecommitValidated
```

## Evidence strength

HAK-010 is an evidence-selection discipline, not a semantic oracle.

A precommitted step may still be a poor test. Therefore:

```text
PrecommittedSelector != AdequateExperiment
PrecommitValidatedEvidence != ClaimTruth
```

Reviewers may challenge whether a selector is sufficiently discriminating before accepting a qualification plan. Changing the selector after execution requires a new binding-policy lineage and new qualification execution; it must not retroactively reinterpret an existing run.

## Current HAK-010 qualification subject

The self-declared E5-target plan is:

`docs/architecture/hak/plans/hak010-precommitted-check-binding-e5-v1.plan.json`

The precommitted binding policy is:

`docs/architecture/hak/policies/hak010-precommitted-check-binding-policy-v1.json`

The policy covers six required checks and fourteen required negative cases. The focused workflow is `.github/workflows/hak-check-binding.yml`.

The selector policy intentionally binds all regression-based negative cases to the focused regression step. That establishes which provider step is evidence that the named negative cases were exercised; it does not make an individual pytest case independently provider-attested.

## Qualification lane

The `HAK Precommitted Check Binding` workflow:

1. resolves the exact pull-request head,
2. checks out and asserts that exact head,
3. uses Python 3.12,
4. compiles inherited HAK evidence tooling and HAK-010,
5. validates schema syntax,
6. lints the precommitted qualification plan,
7. lints the binding policy against that exact plan,
8. runs focused linter and JSON-schema regressions.

A green run is provider execution evidence for the HAK-010 tooling claims only. Receipt, check evidence, conformance and interpretation remain separate artifacts.

## Non-claims

HAK-010 does not:

- grant runtime authority,
- authenticate GitHub metadata cryptographically,
- make CI success equivalent to semantic truth,
- prove that a precommitted test is scientifically adequate,
- permit post-hoc changes to selectors to qualify prior runs,
- convert model output into evidence authority,
- certify governance, consent, rescue or actuator behavior.

## Next boundary

The next useful step is not another generic authority object. It is to materialize real provider-bound evidence from an exact completed HAK run and test whether the HAK-007→010 chain can reconstruct a claim without synthetic fixtures.

That integration exercise should preserve:

```text
ProviderAPIObservation
!= MaterializedReceipt
!= ProviderBoundCheckEvidence
!= PrecommitValidatedEvidence
!= PlanConformance
!= EvidenceInterpretation
```

If that real-evidence exercise reveals gaps, those gaps should be repaired before expanding the evidence stack further.
