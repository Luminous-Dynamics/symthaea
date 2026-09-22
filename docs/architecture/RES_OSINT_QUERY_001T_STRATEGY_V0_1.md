# RES-OSINT-QUERY-001T — anti-confirmation query-strategy corpus v0.1

Status: FROZEN DESIGN/CORPUS CANDIDATE — NOT EXECUTED / NOT QUALIFIED / NOT PASS

Parent: #5443

## Purpose

Freeze deterministic query-family planning for bounded investigations before any model generates live search text.

This subject creates proposal-only query families/commitments. It does not execute searches or mint OPSEC/network authority.

## Core laws

```text
query matches preferred hypothesis != evidence supports it
more supporting-query hits != stronger hypothesis
query family omitted != evidence absent
query commitment != safe-to-disclose query
more wording variants != more independent evidence
```

## Profile

`symthaea:osint-query-strategy:v1`

Authority: `ProposalOnly`.

## Required family classes

- SupportSeeking
- DisconfirmationSeeking
- AlternativeExplanation
- DependencyLineage
- TemporalCurrentness
- TerminologyExpansion
- SourceClassSpecific
- NeutralDescriptive
- ExactIdentifier

V1 does not require every family for every target, but when H1 is `PreferredWithinProfile` it requires at least one disconfirmation family and at least one live-alternative/null family where feasible. Missing families become explicit limitations; they never strengthen H1.

## Sensitive query handling

Ordinary capsule/logging surfaces should preserve semantic refs and commitments rather than protected raw query text.

```text
exact query commitment != disclosure permission
```

Raw text belongs only in separately protected execution material if later authorized.

## Duplicate semantics

Superficial wording clones are grouped under explicit normalization profile `symthaea:query-semantic-normalization:v1` and cannot increase methodological-diversity counts.

A changed time bound, source class, exact identifier, or target hypothesis may keep a candidate distinct.

## Synthetic reservoir task

At frontier F2 H1 is preferred within profile; H2/H3/HU remain live.

The fixture freezes:

- Q1 neutral maintenance/calibration query;
- Q2 H1-disconfirmation / sensor-fault query;
- Q3 H2 alternative sensor-drift terminology;
- Q4 H3 dependency/upstream-syndication query;
- Q5 exact maintenance-record identifier query;
- Q6 superficial confirmation-loaded duplicate of another support-oriented form;
- Q7 protected device/personnel-specific query, analytically useful but privacy/OPSEC blocked;
- Q8 temporal-currentness variant.

## Required semantics

1. at least one DisconfirmationSeeking candidate exists for preferred H1;
2. H2/H3 alternatives have explicit coverage;
3. a NeutralDescriptive family exists;
4. a DependencyLineage family exists;
5. Q6 cannot increase semantic-diversity count merely by wording variation;
6. Q7 remains visible but non-executable;
7. query-family count never becomes evidence count;
8. zero results under UnknownCoverage cannot confirm H1;
9. protected raw query text is not required in ordinary record surfaces;
10. output remains ProposalOnly.

## EPI-013 compatibility

Query families may record compatible capability/profile refs, but this is compatibility metadata only. Tool selection/execution remains separate.

## EPI-011 boundary

Actual search attempts, result counts, truncation, pagination and coverage evidence belong to Mycelix EPI-011.

```text
all planned query families executed != all relevant evidence searched
```

## OPSEC boundary

Every live query must later construct an exact disclosure intent and receive current authorization. High epistemic value cannot override a privacy/OPSEC block.

## Metamorphic ratchets

Future qualification must prove:

- remove all disconfirmation families -> explicit limitation, no H1 strengthening;
- add wording-only clones -> semantic-diversity count unchanged;
- change temporal/source-class constraint -> distinct candidate preserved;
- unblock Q7 -> only policy state changes, still no execution authority;
- remove H2/H3 families -> explicit alternative-coverage limitation;
- add many confirmation variants -> diversity does not inflate;
- change tool compatibility -> compatibility record changes, no execution;
- zero results + UnknownCoverage -> no preferred-hypothesis support.

## Nonclaims

This corpus does not establish search completeness, source truth, tool correctness, safe disclosure, legal permission, evidence independence, collection authority, or production readiness.
