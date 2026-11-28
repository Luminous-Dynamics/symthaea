# MON-044 — Capture Capability and Negative-Space Contract v1

Profile: mon044-capture-capability-input-v1. Status: standalone offline evaluator, version/configuration-bound capability schema, synthetic fixtures and deterministic negative-control tests.

## Required scope

The required-scope inventory describes the complete recovery claim boundary. Each item binds:
- stable item ID;
- state domain;
- object identity;
- operation/observation class;
- required versus optional status;
- exact scope digest.

The inventory has its own source identity and provenance digest. MON-044 verifies the supplied bytes and scope digest but does not authenticate the source. Runtime use requires MON-041 custody.

## Capability identity

The identity binds:
- connector ID, version and distribution digest;
- database engine/version and logical-decoding plugin;
- broker/client versions;
- connector configuration digest;
- publication and schema-history digests;
- snapshot mode;
- transaction metadata enabled state and activation epoch;
- operation filter, column policy and topic-routing digests.

A conformance report is bound to the exact capability identity digest. Reusing a conformance report after configuration/version/source-plugin drift is blocked.

The identity is intentionally strict: configurations without stable digests do not qualify. When a field is genuinely not applicable, the adapter must supply a stable canonical representation and explain it in the versioned capability profile rather than omit the field.

## Capability statuses

- supported: requires a passing conformance profile/report tied to the exact identity;
- conditional: requires passing conformance plus explicit evidence for every preregistered prerequisite;
- unsupported: remains visible and blocks if required;
- excluded: remains visible and blocks if required;
- not_applicable: requires a justification digest;
- unknown: required unknown is indeterminate; optional unknown is reported as negative space without invalidating mandatory coverage.

A required item must have exactly one matching capability entry with exactly the same scope digest. Missing entries hold indeterminate; duplicates or mismatched identities block. Optional declarations outside the required scope are reported separately and never expand the required denominator.

## Verdict semantics

- capability_verdict: whether capability declarations and their supporting evidence are internally coherent.
- coverage_verdict: whether every required scope item is covered or explicitly justified as not applicable.
- campaign_verdict: always indeterminate in v1; custody/source authenticity and full MON composition are external.
- production_action_permitted: always false.

An optional unresolved item may leave capability_verdict indeterminate while coverage_verdict is qualified for the mandatory scope. This is intentional and preserves the distinction between optional capability quality and required recovery coverage.

## Canonicalization

- Connector identity is SHA-256 over the Go JSON encoding of the versioned identity struct.
- Required scope is sorted by item ID, then SHA-256 encoded.
- Capability manifest items are sorted by item ID; each item's prerequisite IDs and evidence are sorted by prerequisite ID; the entire manifest is SHA-256 encoded.
- Empty arrays remain JSON arrays, not null.
- Schema/profile versions are part of the outer input contract; incompatible changes require a new profile version.

The canonicalization fixtures pin empty and non-empty manifest hashes. Cross-language implementations must reproduce the same canonical byte sequence and digest test vectors before interoperability is claimed.

## Security and assurance limits

Hashes do not authenticate:
- connector distribution or runtime binary;
- required-scope inventory source;
- conformance test runner or report issuer;
- publication, filters or live connector configuration;
- schema-history completeness;
- live source state or capture completeness;
- exception authority.

A passing synthetic fixture proves only internal consistency of the supplied synthetic bytes. It does not certify a deployed connector or live recovery.
