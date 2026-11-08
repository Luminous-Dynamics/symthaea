# Qualification Convergence Map V1

Status: architecture/convergence note only. This document grants no qualification authority and establishes no PASS.

Parent program: #3742
First convergence tranche: #3743
Normative-framing guard: #2469

## Purpose

Symthaea already has multiple strong qualification/evidence components. The next step is not to introduce another subject/profile/environment/receipt identity protocol. It is to converge existing primitives into one explicit semantic chain while preserving historical identities and claim ceilings.

The governing rule is:

```text
one normative qualification identity waist
+ adapters / compositions
!= parallel qualification universes
```

In particular, QUAL-001 must not introduce framing magic, collection semantics, subject IDs, profile IDs, environment IDs, attempt IDs, or positive-receipt IDs that compete with the existing generic qualification tooling unless a versioned migration theorem explicitly replaces them.

## Existing ownership map

### Normative qualification framing and semantic identities

Existing `tools/qualification-*` lineages already define or prototype:

- `qualification_framing_v1.py` — language-independent, domain-separated binary framing;
- qualification subject/profile/recipe identities;
- semantic-ID convergence;
- input-closure identity/convergence;
- tool requirements and qualification-environment realization;
- qualification attempt records/sets/integrity;
- PASS selection and replay;
- receipt-candidate gating;
- provider/retry-neutral positive qualification receipt core;
- admission/support-closure follow-ons.

Issue #2469 explicitly requires one normative framing/identity protocol. QUAL-001 inherits that rule.

### Positive qualification receipt core

The existing `symthaea-qualification-receipt-core` candidate defines a portable semantic receipt core only *after* PASS selection / receipt-candidate eligibility.

Its identity binds qualification subject, profile, input closure, environment, and required recipe theorems while deliberately excluding occurrence-specific provider/retry evidence.

Its explicit non-claims include no provider authenticity, no trusted PASS, no scientific validity, no current admission, and no merge/execution authority.

Therefore:

```text
QualificationReceiptCore
!= provider execution witness
!= attempt history
!= claim interpretation
```

QUAL-001 must consume this boundary rather than recreate it.

### Qualification attempt history

The existing qualification-attempt tooling preserves terminal attempts separately from positive receipt identity and distinguishes source/theorem failures from infrastructure/cancellation/unknown outcomes.

Therefore:

```text
QualificationAttemptHistory
!= PositiveQualificationReceipt
```

A later successful attempt never deletes or rewrites earlier failed/unknown attempts.

### Execution lineage

`symthaea-evidence-plane::ExecutionLineageV1` is the repository-level owner for canonical computational input/environment lineage. It binds source/tree, locks, toolchains, host/target, optional Nix identity, features, working directory, argv, relevant environment, and immutable input artifacts, with field-level drift reporting and a guard against mixed evidence lineages.

QUAL-001 should reference/adapt this identity where it fits rather than create a second generic environment/source lineage object.

An environment *realization/seal* may strengthen how those declared facts are observed and bound to actual executables, but it must not silently redefine `ExecutionLineageV1`.

### HAK evidence semantics

HAK-007/008/009/011 already establish the critical separation:

```text
Subject != Plan != Execution != Receipt != Interpretation
ProviderSuccess != ClaimQualified
TerminalFailure != MissingEvidence
PlanNotSatisfied != ClaimDisproven
```

QUAL-001 adopts these semantic distinctions as cross-program invariants. It does not relabel historical HAK artifacts into a new schema.

### CogSec exact-state verification

The CogSec qualification foundation already demonstrates exact-state receipts binding HEAD/tree, lockfile, workspace manifest, qualification script, package set, Rust/Cargo versions, current gate, and last completed gate, plus an independent verifier that re-resolves historical Git objects and re-hashes committed inputs.

QUAL-001 should generalize the reusable theorem, not copy CogSec field names into every domain.

### CI-QUAL provider/runner plane

CI-QUAL-001 owns provider observation ambiguity and runner-state classification: queued, actually running, cancelled, stale summaries, timeout inconsistencies, and focused-subject versus repository-integration evidence.

QUAL must consume provider classifications as evidence. It must not invent a second GitHub runner-state classifier.

### PHYS scientific intent/execution

PHYS-002 binds physical-experiment intent to exact implementation/configuration/workload/seed/sampling/energy boundaries and validates per-seed execution receipts. That remains a scientific-domain theorem.

QUAL may bind the qualification status of PHYS implementations and analyses; it must not absorb the physical-experiment manifest into generic qualification identity.

## Missing cross-program semantic waist

After preserving the ownership above, the genuinely missing generic surface is small.

QUAL-001 owns the composition relationship among three independent dimensions:

```text
ProviderExecutionOutcome
    != PlanConformance
    != ClaimInterpretation
```

Suggested cross-program vocabulary:

```text
ProviderExecutionOutcome
  Succeeded
  Failed
  Cancelled
  TimedOut
  StartupFailure
  Skipped
  OutcomeUnknown

PlanConformance
  NotEvaluated
  Satisfied
  NotSatisfied
  Indeterminate

ClaimInterpretation
  NotEvaluated
  Supported(tier)
  NotSupported
  Contradicted
  Indeterminate
```

The mapping rules are deliberately non-total:

```text
provider failure
    -/-> claim contradiction

provider success
    -/-> claim support

positive receipt core
    -/-> claim support

plan nonconformance
    -/-> claim contradiction
```

`Contradicted` requires evidence specific to the claim's falsification semantics. It can never be inferred solely from infrastructure, checkout, formatting, compilation, lint, or provider failure.

## Canonical composition graph

The cross-program graph should be represented as explicit references to existing identities rather than copied payloads:

```text
EvidenceSubject / QualificationSubjectId
             |
             v
QualificationProfile + Recipe + InputClosure + Environment
             |
             v
QualificationAttemptRecord(s)
             |
             v
ProviderExecutionOutcome
             |
             +--------------------+
             |                    |
             v                    v
PASS selection /            failure evidence
candidate gate              retained append-only
             |
             v
QualificationReceiptCore
             |
             v
PlanConformance
             |
             v
ClaimInterpretation
```

`ExecutionLineageV1` and provider-bound HAK/CogSec evidence attach to the relevant nodes/edges. They do not replace those semantic distinctions.

## Environment realization and TOCTOU boundary

The existing qualification-environment/tool-requirements line already distinguishes qualification environment from input closure. `ExecutionLineageV1` records declared execution lineage. QUAL-001B may add a stronger observed realization/seal proving which executable bytes were actually resolved.

The intended composition is:

```text
QualificationEnvironmentId
+ ExecutionLineageV1
+ observed executable paths/digests
-> EnvironmentRealizationReceipt
```

not:

```text
new parallel QualificationEnvironmentId
```

Later stages should be able to prove they consumed the same sealed executable identities, closing PATH-resolution TOCTOU without redefining the environment semantics.

The seal remains observed environment evidence, not trusted-runner attestation.

## Capability-scoped qualification

The qualification subject must not be able to expand the policy that judges it.

Keep separate externally granted scopes:

```text
trigger_scope
observation_scope
authority_scope
generated_scope
```

with:

```text
effective_authority = granted_authority ∩ requested_authority
```

A descendant may observe parent qualification artifacts while having zero authority to modify them.

This scope/capability layer is orthogonal to subject/profile/receipt identity and must not require a new framing universe.

## Prepare -> seal -> qualify

Generated-state repair is not PASS.

```text
PREPARE
  deterministic, explicitly authorized generated mutations
  -> mutation receipt(s)

SEAL
  exact source/generated state + execution realization frozen

QUALIFY
  no undeclared mutation except evidence outputs
```

Cargo.lock hydration/repair, rustfmt-derived remediation bytes, or other preparation artifacts remain preparation/remediation evidence until a fresh immutable subject qualifies.

## Qualification eras

QUAL uses a qualification-era label only to state which trusted qualification semantics interpreted an artifact.

```text
QERA-001
-> explicit qualified migration
QERA-002
```

An era identifier must not replace existing subject/profile/environment/receipt IDs. Old artifacts remain historical evidence under their original semantics; adapters do not rewrite their provenance.

## Historical-adapter rule

Adapters may derive a generic view from HAK, CogSec, PHYS, resource qualification, ASSURE, or other historical evidence only when the source artifact's theorem is preserved exactly.

They must retain at least:

- source schema/type;
- exact source artifact/commitment identity;
- source qualification semantics/era when known;
- losses or fields that cannot be represented;
- explicit non-claims.

A compatibility adapter cannot claim an old artifact was natively produced under QUAL/QERA-001.

## PHYS repair consequence

Do not repair PHYS-001..006 by importing broad qualification-history ancestry.

After the shared qualification waist is executable-qualified, construct repaired PHYS subjects from the reviewed PHYS semantic payloads and the minimum exact qualification dependencies required by the new era.

Preserve old PHYS heads and red/indeterminate attempts as historical evidence. Any repaired subject receives a new identity, and semantic-equivalence transfer must be proven rather than assumed.

Until then, PHYS-006's current red run establishes only that its qualification attempt stopped before product qualification completed. It does not establish a fixture-source failure or claim contradiction.

## Integration/landing discipline

The `tools/qualification-*` branches are valuable semantic source lineages, but broad Git ancestry is not a safe integration mechanism.

A future landing tranche should:

1. enumerate the exact generic qualification primitives needed;
2. identify their authoritative source blobs/commits and dependency closure;
3. reject duplicate/competing framing implementations;
4. construct a minimal convergence subject on then-current `main`;
5. preserve historical branch identities as provenance;
6. requalify the combined tree rather than transferring standalone PASS by ancestry.

## QUAL-001A revised boundary

QUAL-001A should now prove the convergence map and the missing three-axis algebra, not recreate generic subject/profile/recipe/environment/attempt/receipt identities.

Before adding a new generic type, its implementation must answer:

> Which existing qualification primitive cannot express this theorem, and why is an adapter/composition insufficient?

If that question has no concrete answer, do not add the type.

## Non-claims

This map does not establish that any historical qualification branch is merge-ready, that the existing generic tooling is fully qualified on the current combined tree, that GitHub metadata is authenticated, that execution chronology is trusted, that scientific claims are true, or that PHYS/AURUM is qualified.
