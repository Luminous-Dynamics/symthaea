# Financial Time Frontier v0.1

A small, dependency-free Rust reference kernel for a shared information-frontier contract
used by financial research and quantitative evaluation.

## Semantics

- Valid time answers when the represented fact applies.
- Published time answers when a source published a version, if public publication exists.
- Known/available time answers when the exact version became available to the evaluated
  principal under the declared rights profile.
- Retrieved time answers when this evaluation environment ingested the bytes.
- Decision frontier is the inclusive upper bound for information admission.

Validity intervals use the half-open convention [valid_from, valid_until). A known but
future-effective commitment can be admitted as information while being explicitly labelled
NotYetEffectiveAtTarget; it must not be described as a realized current fact.

## Evaluation modes

- HistoricalReconstruction: later retrieval is permitted only when a versioned archive,
  release record or equivalent evidence proves the exact version's historical availability
  for the evaluated principal and rights profile. Missing or unverifiable proof is
  Indeterminate.
- ProspectiveStreaming: both source availability and actual local retrieval must be at or
  before the decision frontier.

The implementation rejects future-known evidence, identity mismatches, impossible retrieval
ordering, wrong rights profiles, denied rights and malformed revision ancestry. Superseding
versions must have a strictly later known-at instant than their predecessors. It retains
older versions while identifying which admitted version supersedes them at the chosen
frontier. It counts unique provenance-family identifiers rather than documents; that count is not proof that the underlying sources are actually independent.

`Disposition::Pass` means only that the supplied version passes the declared temporal,
identity, rights and proof-input checks. It does not mean the fact is true, relevant to the
requested period, independently corroborated, or sufficient to support an analytical claim.
Inspect `ValidityState` and reason codes separately; claim support and source-family
independence belong to higher-level verifiers.

## Rights and entitlement proof

A matching `rights_profile_id` and `RightsDisposition::Authorized` value are not sufficient
for qualification. Each evidence version also needs a separately verified `RightsProof`
binding the exact evidence ID, evaluated principal, rights profile, authorization disposition,
validity interval, verifier identity and verifier-build SHA-256. The rights proof must be valid
at the decision frontier. Missing or unverifiable rights proof is Indeterminate; identity,
profile, disposition, verifier or validity mismatches fail closed.

The crate consumes the typed result from an independent Mycelix/provenance rights verifier.
It does not authenticate signatures or licences itself. Do not construct `RightsProof::Verified`
from an unverified JSON field; only the trusted adapter/verifier boundary may produce it.

## Timestamp precision

Version 0.1 accepts only exact, offset-aware availability instants. Some source APIs expose only calendar dates or vintage dates. Do not convert a date-only vintage to midnight and claim an exact instant. For an intraday decision frontier, that source is Indeterminate unless another retained record proves the exact availability time. A later version should support interval-bounded availability (earliest possible / latest possible) and return Indeterminate whenever the interval straddles the frontier.

## Trust boundary

The crate does not retrieve or authenticate source material, verify signatures, validate a
source publisher, check licence text, verify artifact bytes, or establish that a source
supports a financial claim. The caller must independently establish availability and rights
proofs and must use the existing content-addressed artifact verifier for exact-byte integrity.
This crate only applies deterministic temporal and revision rules to those already-qualified
inputs. Its output is not legal advice, investment advice, a market-data entitlement, or a
trading authorization.

## Run

From this directory, run cargo fmt --check and cargo test.

No external Rust crates are required.
