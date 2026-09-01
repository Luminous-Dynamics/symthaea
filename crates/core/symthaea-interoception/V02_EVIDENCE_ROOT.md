# Affective Emergence v0.2 — Observational Evidence Root

Status: **design-only / blocked on Native Interoception v0.1 qualification**

The v0.2 design now has several independently versioned identities: native baseline, study protocol, candidate definitions, scenarios, analysis rules, blind mapping, exclusion decisions, execution traces, and derived artifacts. This document defines the proposed root-of-evidence object that binds them into one auditable lineage.

## Principle

No single SHA-256 can prove scientific validity, but a root manifest can make silent substitution detectable.

The evidence root should answer:

> Exactly which qualified baseline, prospective study, candidate definitions, scenario cohort, analysis rules, and blinding commitment does this result claim to belong to?

If any identity changes, the root identity changes.

## Proposed ObservationalEvidenceRootManifest

Before confirmatory execution, construct and freeze a prospective root containing at minimum:

- root schema/version;
- evidence run class (`Confirmatory` for confirmatory inference);
- exact v0.1 source commit;
- v0.1 model-semantics version;
- v0.1 qualification-receipt SHA-256;
- v0.1 evidence-capsule SHA-256;
- exact v0.2 observatory/design implementation source commit;
- study-preregistration SHA-256;
- study-level preregistration SHA-256;
- ordered primary/secondary candidate-definition digests;
- explicit primary candidate digest;
- ordered nuisance/null baseline candidate digests;
- scenario-cohort manifest SHA-256;
- analysis-plan SHA-256;
- exclusion-criteria definition digest;
- arm-identity-mapping commitment SHA-256;
- information-firewall contract/version;
- temporal-alignment contract/version;
- blinding-custody contract/version;
- blinding-strength declaration;
- no-feedback isolation-gate definition/version;
- semantic-label-canary gate definition/version;
- toolchain/dependency identities required by the v0.2 implementation;
- canonical SHA-256 of the entire validated root.

This prospective root is frozen before confirmatory execution.

## AnalysisPlan identity

The root must bind an independently canonical `AnalysisPlan` rather than relying on prose in a notebook or paper.

It should include:

- primary hypotheses;
- exact candidate/baseline IDs;
- scenario subset/cohort references;
- comparison cut points/windows;
- directional relations;
- minimum-effect thresholds;
- equivalence tolerances;
- parameter/sensitivity region;
- candidate ranking/model-comparison rule;
- treatment of ties;
- treatment of indeterminate results;
- boundary-turnover diagnostic rule;
- required prefix-causality tests;
- required neutrality tests;
- required no-feedback test;
- required semantic-label-canary test;
- deterministic robustness summary rules;
- exclusion handling;
- unblinding procedure/version.

Changing any of these after confirmatory execution starts creates a new analysis identity and a new confirmatory lineage.

## Prospective vs realized evidence

Do not mix prospective identity with realized results.

### Prospective root

Contains what was promised before confirmatory data:

- qualified baseline identity;
- study/protocol identity;
- candidates;
- scenarios;
- analysis;
- blinding commitment;
- exclusions;
- environment/toolchain identity.

### Realized evidence package

After execution, produce a result package that binds the prospective root plus:

- exact study execution digest(s);
- exact exclusion decision receipt digest(s);
- forecast trajectory artifact digests;
- candidate time-series artifact digests;
- prefix-causality validation report digest;
- no-feedback isolation report digest;
- semantic-label-canary report digest;
- blinded candidate-comparison report digest;
- sensitivity/robustness report digest;
- null-control report digest;
- unblinding receipt digest;
- semantic hypothesis evaluation digest;
- all excluded/indeterminate scenario artifact digests;
- complete artifact hash list.

The realized package has its own canonical SHA-256.

## v0.1 qualification dependency

A confirmatory v0.2 prospective root is invalid unless the referenced v0.1 qualification receipt says the exact native baseline is qualified and the referenced v0.1 evidence capsule validates.

The root may be drafted before v0.1 qualifies, but it must carry a status such as `BlockedOnBaselineQualification` and cannot be promoted to `LockedConfirmatory` until the dependency is satisfied.

This prevents a later v0.1 baseline substitution from being hidden beneath the same v0.2 study name.

## Candidate-set closure

The prospective root contains the complete candidate/baseline set eligible for confirmatory interpretation.

After execution begins:

- no new primary candidate may be added;
- no baseline may be removed because it explains the result too well;
- no failed candidate may be deleted from the package;
- a new candidate may be computed only as explicitly exploratory post-hoc analysis and must not alter the locked confirmatory root.

## Scenario-set closure

The root binds the complete confirmatory holdout cohort before candidate outputs are inspected.

Every scenario digest in the cohort must appear in final accounting as included, excluded, or indeterminate.

The realized evidence package must report:

`locked_count == included_count + excluded_count + indeterminate_count`

A mismatch is a hard integrity failure.

## Artifact dependency graph

Recommended dependency direction:

`v0.1 qualification/evidence`

→ `v0.2 prospective evidence root`

→ `study execution`

→ `exclusion decisions`

→ `forecast trajectories / candidate time series`

→ `prefix-causality + isolation + semantic-leak validation`

→ `frozen blinded comparison`

→ `unblinding receipt`

→ `semantic hypothesis evaluation`

→ `realized evidence package`

No downstream artifact may silently alter an upstream identity.

## Fail-closed promotion state

A later typed status should distinguish at least:

- `DraftDesign`;
- `BlockedOnBaselineQualification`;
- `LockedExploratory`;
- `LockedConfirmatory`;
- `ExecutedAwaitingValidation`;
- `Excluded`;
- `Indeterminate`;
- `QualifiedNegativeOrNullResult`;
- `QualifiedSupportedResult`;
- `IntegrityFailure`.

There should be no generic `Success` state.

A failed primary hypothesis on an otherwise valid confirmatory study is a **qualified negative/null result**, not an integrity failure.

An integrity failure means the evidence chain itself cannot support inference.

Examples of integrity failure include future-information leakage, observational feedback into native execution, semantic-label leakage into blinded primary artifacts, execution replay mismatch, missing locked scenarios, or artifact/hash substitution.

## Reproduction contract

An independent reproducer should be able to start from the prospective root and verify:

1. exact source/dependency/toolchain identities;
2. v0.1 baseline qualification;
3. candidate definitions;
4. scenario cohort materialization;
5. study execution replay;
6. exclusion decisions;
7. derived forecast/candidate artifacts;
8. prefix-causality, no-feedback, and semantic-canary gates;
9. blinded comparison artifact;
10. unblinding transformation;
11. semantic hypothesis outcomes;
12. final realized evidence digest.

Any discrepancy must identify the first dependency edge that fails instead of merely reporting a different final number.

## Human-readable receipt

In addition to machine-readable JSON, generate a concise deterministic text receipt summarizing:

- root digest;
- source commits;
- v0.1 qualification state;
- run class;
- primary candidate;
- candidate/baseline counts;
- scenario counts and disposition accounting;
- blinding strength;
- prefix-causality/no-feedback/semantic-canary status;
- primary hypothesis outcomes;
- integrity-gate status;
- realized evidence package digest.

The human-readable receipt is a view of machine-readable evidence, not an independently editable source of truth.

## Claim boundary

A valid evidence root can show that one fixed prospective research plan led reproducibly to one fixed set of results without silent substitution. It does not by itself establish that any regulatory candidate is emotion, subjective valence, sentience, or consciousness.