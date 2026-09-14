# Human Agency Kernel Audit v1 — Summary

**Status:** architecture audit only; non-authorizing; non-qualifying.

## Core theorem

HAK-001 freezes the following distinctions before product implementation:

```text
BaselineStanding
    != ScopedQualification
    != Recommendation
    != Consent
    != DelegatedAuthority
    != Execution
    != Outcome
    != HumanBenefit
    != LongitudinalCapabilityChange
```

The central prohibition is:

```text
ModelAssessmentOfPerson != AuthorityOverPerson
```

## Constitutional asymmetry

HAK proposes an intentional asymmetry:

```text
human baseline protections  -> durable floor
human scoped power          -> evidence / role constrained
machine delegated power     -> narrower + expiring + revocable
```

This permits strong domain qualification for consequential roles without converting domain competence, reputation, participation, contribution, or AI inference into generalized human worth.

## Main governance finding

Mycelix's eight-dimensional sovereign profile addresses real weaknesses in capital-only and single-axis governance. However, an eight-dimensional private score can still create a legitimacy problem if it gates baseline participation or automatically multiplies civic voice.

The safer decomposition is:

```text
member / democratic legitimacy
affected-party standing
domain expertise
evidence quality
constitutional constraints
```

These may interact procedurally but should not be silently collapsed into one scalar.

The current sovereign-profile work may still be valuable as scoped contribution/role evidence for expertise discovery, voluntary delegation, stewardship roles, and domain-specific qualification.

## Existing architecture to reuse carefully

- Symthaea rescue consent already demonstrates explicit, case-specific, replay-resistant, withdrawable consent semantics.
- Mycelix reciprocal accountability provides a strong pattern for Know / Inspect / Contest / HumanReview / Appeal rights around person-linked inference and access.
- Mycelix SubPassport already models bounded, expiring, revocable human-to-AI delegation, but its current signable transcript does not bind every security-relevant field and must not yet become canonical HAK authority.
- Symthaea psych-bench already contains metacognitive calibration infrastructure that can host later assistance/reliance experiments.
- Symthaea's recent evidence architecture already strongly separates evidence from belief and action authority.

Reuse the theorems and qualified contracts; do not couple unrelated domain crates merely because their vocabulary sounds similar.

## First candidate protocols

Future tranches may explore:

```text
AugmentationContractV1
CapabilityOutcomeVectorV1
AgencyImpactClaimV1
DissentEnvelopeV1
InstitutionalLinterFindingV1
IntentGraphV1          # Mycelix-owned candidate
```

No common crate is implied by the list.

## First benchmark families

HAK proposes research comparisons across:

```text
NoAI
AnswerOnly
Scaffold
Collaborate
Delegate
```

with measurements including:

- immediate performance;
- understanding;
- delayed unaided transfer;
- calibration and appropriate reliance;
- option preservation and reversibility;
- dependency;
- viewpoint diversity;
- coordination quality;
- contestability.

A task getting easier is not sufficient evidence that the human became more capable.

## Dependency-gated sequence

```text
HAK-001  audit
HAK-002  augmentation-contract semantics
HAK-003  capability-outcome vector
HAK-004  agency-impact preregistration
HAK-005  assistance/reliance psych benchmarks
HAK-006  independent elicitation + dissent envelope
HAK-007  collective-intelligence experiment harness
HAK-008  advisory institutional linter

MYC-HAK-001  delegation transcript hardening
MYC-HAK-002  baseline-standing vs scoped-role governance RFC
MYC-HAK-003  IntentGraphV1
MYC-HAK-004  intent -> scoped commitment compiler
MYC-HAK-005  accountability / contestability composition
```

Every later tranche requires independent review and qualification. HAK-001 transfers no authority or evidence status.

## Review boundary

Review HAK-001 only on this question:

> Does the audit define a sufficiently precise common/non-common human-agency map to prevent future assistance, scoring, confidence, expertise, or delegation mechanisms from silently becoming authority over persons?
