# Symthaea Music Theory Patch Series 17

**Date:** 2026-07-20
**Baseline:** Patch Series 16 / Git tree `f9f0c8252ca819be2994ee16823c2e14a16c1968`

## Purpose

Series 17 adds a governed publication layer above the selective-disclosure and
public-governance exports introduced in Series 16. It defines publication
policy, externally authenticated delegation, an append-only catalog with
supersession and revocation, head-bound status proofs, and a reproducible
third-party audit package.

The series also repairs two inherited schema-registry compile defects found by
compile-oriented static inspection. These defects were syntactically valid
Rust token trees inside `vec![]`, so ordinary delimiter and tree-sitter parsing
did not detect their invalid function arity.

## Landing order

### 1. Repair the inherited persistence registry

- Correct the malformed governance-export registry entry.
- Correct the malformed study-acceptance-decision registry entry.
- Register every publication and audit-package persistence contract.
- Add publication roles to the schema-registry coverage regression.
- Give schema roles stable numeric representation for canonical package bytes.

### 2. Publication policy

- Add release, auditor, and archive channels.
- Bind allowed disclosure profiles and retention states.
- Express requirements for release eligibility, receipt-chain evidence, source
  revision, source tree, and unresolved legacy attachment epochs.
- Produce a complete machine-readable gate report.
- Use architecture-independent persisted numeric limits.

### 3. Externally authenticated delegation

- Bind exact policy, channel, delegator, delegate, source restrictions,
  logical validity interval, publication allowance, and nonce.
- Create canonical length-prefixed payload bytes.
- Wrap externally generated signatures with signer metadata.
- Require canonical lowercase signature hexadecimal.
- Delegate authentication to a caller-supplied verifier.
- Bind authorization decisions to the exact export, ordinal, and evaluation
  epoch.

### 4. Append-only catalog and status proofs

- Record immutable publication identities.
- Add globally ordered publish, supersede, and revoke events.
- Enforce terminal transitions, event-chain continuity, logical-epoch order,
  unique channel/export identity, contiguous delegation ordinals, and total
  publication allowance.
- Re-run the external delegation verifier at catalog publication time.
- Build compact status proofs bound to one exact catalog head and event count.

### 5. Reproducible third-party audit packages

- Package the governance export, canonical attestation payload, policy, signed
  delegation, authorization, catalog, status proof, and schema registry.
- Bind schema roles through stable numeric identifiers rather than debug text.
- Audit every embedded relationship and identity.
- Require caller-supplied delegation authentication for a verified package.
- Persist explicit limitations around freshness, external authority, legal
  compliance, external-copy deletion, and transparency policy.

### 6. Operator tooling and documentation

New examples:

- `evidence_publication_policy`
- `evidence_publication_delegation`
- `evidence_publication_catalog`
- `evidence_third_party_audit_package`

New release contract:

- `EVIDENCE_PUBLICATION_RELEASE.md`

## Compatibility

- No changes to `Score`, `compose()`, `compose_styled()`, or musical output.
- No private calibration-bundle schema change.
- Publication artifacts are new external persistence contracts.
- New public trust envelopes reject unknown fields where private-field
  injection would be dangerous.
- External signature algorithms and trust roots remain outside the crate.

## Verification gates

Before packaging, the series must pass:

1. `git diff --check` for every patch.
2. Tree-sitter parsing for every Rust target.
3. Duplicate enum-variant, struct-field, test-attribute, and crate-root export
   checks.
4. Direct free-function arity checks across the publication modules and tools.
5. Macro-token arity checks for every schema-registry `entry()` and
   `legacy_entry()` call.
6. Example-import to crate-root export resolution.
7. Exact `git am --3way` application to the Series-16 baseline.
8. Exact final Git tree equality with the authored repository.
9. Byte-for-byte source archive reproduction.
10. Gzip and SHA-256 verification of every deliverable.

## Required canonical-shell verification

Cargo, rustc, rustfmt, Clippy, and Nix are unavailable in the construction
environment. Before merge, run:

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Run the repository's normal Nix verification lane and exercise publication
against the real external verifier used by the release authority.

## Trust boundary

Series 17 proves deterministic relationships among the supplied artifacts. It
does not prove:

- that the catalog authority or delegation signer should be trusted;
- that a signature algorithm or key-management system is secure;
- that a packaged status proof reflects the latest catalog head;
- transparency-log inclusion or freshness;
- legal or regulatory compliance;
- deletion of copies outside the governed evidence system;
- correctness of caller-supplied logical epochs.
