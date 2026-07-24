# Symthaea Music Theory Patch Series 22 Plan

**Date:** 2026-07-21  
**Base:** verified Patch Series 21 final tree  
**Theme:** Language-neutral canonical contracts, differential verification, and reproducible release packaging

## Objective

The evidence stack is useful only when an independent implementation can reject the same malformed artifacts and derive the same canonical identities. Series 22 converts the accumulated Rust contracts into an executable conformance boundary.

It does not add publication authority. It makes existing claims easier to verify independently.

## Deliverables

1. A versioned language-neutral canonical encoding specification.
2. Exact canonical byte and SHA-256 vectors for public Series 16–21 models.
3. Positive, boundary, and adversarial JSON fixtures.
4. A verifier-result taxonomy separating parse, structural, identity, policy, signature, lineage, freshness, conflict, and authorization failures.
5. A differential harness that invokes multiple external verifier implementations without a shell.
6. Architecture-independence audits for integer widths, ordering, enum roles, and unknown-field handling.
7. Deterministic archive and manifest tooling.
8. A portable offline verification kit containing no private study or governance data.
9. Consumer compatibility and schema evolution reports.
10. Reproducible release attestations over exact source, vectors, tools, and documentation.

## Canonicalization rules

- Every encoded field is explicitly ordered and versioned.
- Strings are length-delimited UTF-8; no locale normalization is implicit.
- Integers are fixed-width and range checked.
- Collections declare whether order is semantic, canonicalized, or rejected when unsorted.
- Duplicate map keys, duplicate signer identities, duplicate record IDs, and duplicate enum roles are rejected.
- Unknown fields fail closed for persisted evidence models.
- Floating-point values are forbidden in authorization and identity contracts unless represented through a specified canonical integer/fixed-point encoding.
- Rust debug names and platform-sized types never enter canonical bytes.

## Differential acceptance rule

A release candidate passes only when:

- the Rust reference verifier;
- at least one independent verifier implementation; and
- the frozen expected-result corpus

agree on every fixture's acceptance class, failure code, canonical bytes, and identity digest.

Disagreement is a release blocker, not a majority vote.

## Landing order

1. Inventory every public persisted contract and canonical function.
2. Define stable conformance envelope and failure taxonomy.
3. Export exact positive canonical vectors.
4. Export one-field mutation vectors and expected failure stages.
5. Add ordering, duplication, integer-boundary, and unknown-field vectors.
6. Add publication/recovery/resumption lineage scenario vectors.
7. Add a no-shell differential verifier harness.
8. Add a minimal independent reference verifier interface and sample implementation guide.
9. Audit architecture-independent persistence across Series 16–21.
10. Add deterministic archive and manifest builder.
11. Add offline verification-kit exporter and verifier.
12. Add schema compatibility matrix and consumer report.
13. Add fuzz seed corpus and property-test replay.
14. Add end-to-end release reproducibility test.
15. Register conformance persistence roles and freeze ordinals.
16. Document and package the independent-verification release.

## Non-claims

Conformance does not prove signer legitimacy, key security, witness independence, legal authority, universal availability, absence of withheld forks, or artistic validity of the underlying musical evidence.
