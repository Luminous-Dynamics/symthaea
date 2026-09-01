# Affective Emergence v0.2 — Blinding Custody and Unblinding Gate

Status: **design-only / blocked on Native Interoception v0.1 qualification**

Native Interoception v0.1 already removes semantic arm IDs from primary execution and blinded metric artifacts. This document strengthens that boundary by separating the arm-identity mapping itself from the primary-analysis artifact flow.

## Important limitation

Artifact-level blinding is not the same thing as guaranteed cognitive blinding of a human investigator.

If one person designs the scenarios and knows which blind code belongs to which semantic arm, no software claim should pretend that person was psychologically blinded.

The goal here is narrower and auditable:

- primary analysis artifacts do not require semantic arm identity;
- the arm mapping is a separately committed object;
- automated primary analysis can run without access to the mapping;
- unblinding is a distinct, recorded transition after the blinded artifact is frozen.

Where an independent analyst/custodian is available, this structure supports stronger practical blinding.

## Proposed ArmIdentityMapping artifact

A later study package should create a separate mapping artifact containing at minimum:

- mapping schema/version;
- study-preregistration SHA-256;
- one unique opaque blind code for every semantic arm ID;
- mapping-generation method/version;
- optional externally supplied randomization identifier/salt commitment;
- canonical mapping SHA-256.

The study preregistration/evidence capsule should bind the mapping digest without requiring the mapping contents to be present in the primary-analysis workspace.

A digest proves commitment, not secrecy. If secrecy is required, storage/encryption/access control must be handled outside this crate or by a separately qualified mechanism.

## Opaque-code requirements

Blind codes should not reveal semantic condition meaning.

Avoid codes such as:

- `control`;
- `threat`;
- `recovery`;
- `load-high`;
- deterministic unsalted hashes of obvious semantic IDs that an analyst can trivially reproduce.

Prefer externally generated opaque identifiers whose generation mechanism is recorded prospectively.

The same blind code must not be reused for different semantic arms inside one study.

## Artifact-flow separation

Recommended confirmatory flow:

1. define semantic arm specifications;
2. generate and commit `ArmIdentityMapping`;
3. bind the mapping SHA-256 into the locked study package;
4. execute the study using only blind codes in primary trace outputs;
5. extract observational candidate artifacts using blind codes only;
6. produce and freeze the blinded metric/comparison artifact;
7. record its SHA-256;
8. only then make the mapping available to the unblinding/evaluation process;
9. emit an explicit unblinding receipt that binds both artifacts.

A primary-analysis process should not need to deserialize the mapping artifact at all.

## Proposed UnblindingReceipt

A later `UnblindingReceipt` should bind:

- schema/version;
- study-preregistration SHA-256;
- arm-identity-mapping SHA-256;
- study-execution SHA-256;
- exclusion-decision SHA-256;
- candidate-definition/cohort/analysis digests;
- frozen blinded metric/comparison artifact SHA-256;
- resulting semantic hypothesis-evaluation SHA-256;
- exact unblinding operation/version.

The receipt should validate that the blinded artifact already existed and is unchanged relative to the digest supplied to unblinding.

## No pre-unblinding semantic joins

Qualified primary-analysis code must not expose an API that accepts both:

- blinded candidate values; and
- `arm_id -> blind_code` semantic mapping.

That join belongs only in the unblinding layer.

A code search/gate should fail if online/primary observatory modules import the semantic mapping type or expose semantic arm IDs.

## Candidate selection must happen before semantic unblinding for confirmatory runs

For a confirmatory study, the following must already be fixed before the arm mapping is provided to the semantic evaluator:

- primary candidate definition;
- secondary candidate definitions;
- baseline set;
- cut points/windows;
- equivalence/effect thresholds;
- scenario cohort;
- exclusion decisions;
- deterministic comparison/ranking rule;
- treatment of ties and indeterminate results.

If semantic-arm inspection leads to a changed candidate or threshold, the altered analysis is exploratory and requires a new confirmatory lineage.

## Automated blind analysis

Where practical, generate a machine-readable `BlindedCandidateComparisonReport` containing only:

- blind codes;
- candidate IDs/digests;
- scenario IDs or blinded scenario codes as appropriate;
- values/margins;
- structural validation outcomes;
- no semantic condition labels.

The report should be canonical/hashable and freeze before unblinding.

This makes it possible for an independent reviewer to verify that the semantic report is a deterministic transformation of a previously frozen blinded report plus the committed mapping.

## Single-investigator mode

When the same investigator necessarily designs, runs, and interprets the study, the evidence package should say so explicitly.

Suggested evidence field:

`blinding_strength` with values such as:

- `ArtifactOnly`;
- `IndependentPrimaryAnalyst`;
- `IndependentMappingCustodian`;
- `IndependentAnalystAndCustodian`.

Do not claim stronger blinding than was actually used.

`ArtifactOnly` is still useful because it preserves post-hoc auditability, but it does not eliminate expectation bias in a person who knows the mapping.

## Mapping tamper tests

The eventual implementation should test that:

- duplicate blind codes are rejected;
- missing semantic arms are rejected;
- unknown semantic arms are rejected;
- mapping digest changes when any pair changes;
- a mapping from another study is rejected;
- a tampered mapping cannot validate against the locked mapping digest;
- a frozen blinded artifact cannot be replaced after unblinding without breaking the receipt chain.

## Information-firewall interaction

The arm mapping is forbidden online information under `V02_INFORMATION_FIREWALL.md`.

A candidate value at time `t` must not depend on semantic arm identity. Therefore future-mutation/prefix tests should also run with alternate semantic mapping artifacts and require identical numerical candidate values for the same blinded execution data.

Semantic meaning belongs to inference, not candidate computation.

## Reviewability

The final evidence lineage should allow a reviewer to inspect, in order:

1. prospective study/candidate/scenario identities;
2. exact execution artifacts;
3. exclusion decisions;
4. frozen blinded candidate report;
5. committed arm-mapping digest;
6. mapping revealed at unblinding;
7. deterministic semantic hypothesis report;
8. immutable links/hashes between all stages.

This makes the moment at which semantic condition identity enters the analysis explicit rather than implicit.

## Claim boundary

This contract strengthens auditability and may strengthen practical blinding when independent people or systems are used. It does not guarantee absence of investigator bias and it has no bearing on whether any regulatory observable constitutes emotion or subjective experience.