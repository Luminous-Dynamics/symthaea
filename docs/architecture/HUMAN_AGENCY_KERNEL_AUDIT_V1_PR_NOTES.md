# Human Agency Kernel Audit v1 — PR Notes

## Suggested title

`docs(agency): audit Human Agency Kernel boundaries`

## Exact base

```text
main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1
```

## Scope

Documentation only. Expected files:

```text
docs/architecture/HUMAN_AGENCY_KERNEL_AUDIT_V1.md
docs/architecture/HUMAN_AGENCY_KERNEL_AUDIT_V1_REVIEW_CHECKLIST.md
docs/architecture/HUMAN_AGENCY_KERNEL_AUDIT_V1_SUMMARY.md
docs/architecture/HUMAN_AGENCY_KERNEL_AUDIT_V1_SCOPE.md
docs/architecture/HUMAN_AGENCY_KERNEL_AUDIT_V1_PR_NOTES.md
```

No product source, manifests, dependency graph, lockfile, workflow, runtime behavior, governance policy, or action authority changes.

## Why this audit now

Symthaea and Mycelix increasingly touch both sides of the same boundary:

```text
understanding / learning / deliberation / coordination

and

scoring / gating / delegation / execution / institutional authority
```

Without a common/non-common map, neighboring domains can independently create reasonable local mechanisms that compose into an unreasonable global authority system.

HAK-001 freezes the distinctions before implementation.

## Core theorem

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

Central prohibition:

```text
ModelAssessmentOfPerson != AuthorityOverPerson
```

## Main finding

The current eight-dimensional sovereign-profile work solves real problems of single-axis capital/reputation governance, but dimensionality and zero-knowledge privacy do not by themselves establish legitimacy for using a behavioral score to gate baseline civic participation.

HAK-001 proposes separating:

```text
member / democratic legitimacy
affected-party standing
domain expertise
evidence quality
constitutional constraints
```

rather than collapsing them into one generalized fitness score.

This is an audit finding, not a migration in this PR.

## Existing work retained

The audit deliberately preserves valuable current work:

- sovereign-profile dimensions remain candidates for scoped contribution/role evidence;
- rescue-consent semantics provide a strong consent theorem without becoming a universal consent type;
- reciprocal accountability provides reusable subject-rights infrastructure;
- SubPassport provides a useful bounded-delegation starting point;
- psych-bench provides a strong experimental host;
- evidence/authority separation remains foundational.

## Concrete gap found

Before SubPassport can serve as canonical HAK delegation authority, its signed transcript needs an independent hardening tranche. The current signable bytes do not bind every security-relevant field, including `ahimsa_enforced` and the stated purpose.

Therefore:

```text
current SubPassport != canonical HAK delegation receipt
```

No SubPassport code changes belong in HAK-001.

## Research direction

HAK-001 incorporates current human-AI research constraints around:

- dependent vs autonomous cognitive offloading;
- explanation-driven over-reliance;
- human reliance calibration;
- human-AI feedback-loop bias;
- AI-driven idea homogenization;
- collective intelligence as a coordination problem;
- weak real-world impact/complaint measurement in public-sector AI adoption.

The resulting benchmark direction measures both immediate assisted performance and delayed unaided capability.

## Proposed next tranches

```text
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

Later PRs should cite the exact HAK theorem they materialize and receive independent qualification.

## Deliberate non-claims

HAK-001 does not establish that:

- current Mycelix governance is illegitimate;
- one-person-one-vote is universally correct;
- expertise should have no institutional role;
- the sovereign profile should be deleted;
- a universal Human Agency Kernel crate should exist;
- any current AI system improves long-term human capability;
- any benchmark result establishes human flourishing;
- any AI system is or is not a moral patient;
- SCI-001 or another PR qualifies this audit;
- this audit grants any runtime or governance authority.

## Review boundary

Review only:

> Does HAK-001 define sufficiently precise common and non-common semantics to prevent future assistance, scoring, confidence, expertise, consent, and delegation mechanisms from silently becoming authority over persons, while preserving useful scoped qualification and coordination?

Keep draft. Do not begin authority-bearing HAK integration until the relevant lower-level semantics are independently frozen and qualified.
