# Symthaea Music Theory Patch Series 16

**Date:** 2026-07-20
**Baseline:** Patch Series 15 / Git tree `5d2c8164aab3718aa64265e9e42377d5cec546f5`

## Purpose

Series 16 makes governance evidence portable without exporting listener-level records or overstating what deterministic software can prove. It adds exact before/after governance receipts, identity-free retention compliance, selective-disclosure projections, a combined public governance envelope, and canonical bytes for an external signer or transparency system.

The series does **not** claim that withdrawal erases uncontrolled copies, that a retention snapshot establishes legal compliance, that a SHA-256 authenticates its publisher, or that externally verified privacy mechanisms are implemented by the music-theory crate.

## Landing order

### 1. Exact governance transitions

- Add versioned receipts for participant withdrawal, retention enforcement, and externally verified privacy releases.
- Bind source identity, engine identity, logical epoch, before-bundle SHA-256, after-bundle SHA-256, and the complete operation evidence.
- Reconstruct the expected post-operation bundle independently and require exact equality.
- Add contiguous receipt chains with sequence, epoch, source, engine, and bundle-continuity checks.
- Make existing governance mutation tools optionally emit receipts.

### 2. Selective disclosure

- Add public-release and auditor-minimal disclosure policies.
- Project only release assessment, already-suppressed public study statistics, aggregate governance totals, aggregate privacy-budget totals, and selected source identity.
- Encode an explicit mandatory omission list for private corpus cases, judgments, pseudonyms, response identities, links, tombstones, proofs, reveals, assignment registries, and credential presentations.
- Reject unknown fields in public envelopes rather than ignoring injected data.
- Support both self-audit and exact reconstruction against the private evidence bundle.

### 3. Retention compliance snapshots

- Derive identity-free retention state from the private bundle and a versioned logical-epoch policy.
- Distinguish `compliant`, `review_required`, and `noncompliant`.
- Preserve unknown legacy attachment epochs as unresolved evidence.
- Honor fail-closed unknown-epoch policy.
- Export only aggregate counts and age boundaries, never response identities.

### 4. Public governance export

- Combine selective disclosure, retention compliance, and an optional receipt-chain summary.
- Require receipt-chain source and engine identity to match the disclosed bundle.
- Require the chain to terminate at the exact disclosed bundle SHA-256.
- Bind mandatory machine-readable limitations into the export identity.

### 5. External attestation handoff

- Produce stable length-prefixed bytes over the governance export identity, source bundle, engine, revision, and optional tree.
- Leave signature algorithms, keys, signer authority, freshness, timestamping, and transparency-log policy outside the theory crate.

### 6. Tools and documentation

New tools:

- `evidence_governance_receipt`
- `evidence_governance_receipt_chain`
- `evidence_selective_disclosure`
- `evidence_retention_snapshot`
- `evidence_governance_export`
- `evidence_governance_attestation_payload`

Existing tools extended with optional receipt output:

- `evidence_withdraw_study_response`
- `evidence_enforce_study_retention`
- `evidence_privacy_release`

## Compatibility

- No change to `Score`, `compose()`, `compose_styled()`, or existing composition behavior.
- No evidence-bundle schema bump is required because Series 16 introduces external governance artifacts rather than new private-bundle fields.
- New persisted contracts are registered in the schema registry.
- Public governance envelopes use strict unknown-field rejection.

## Verification gates

Before release, the series must pass:

1. `git diff --check` for every patch.
2. Tree-sitter parsing for every Rust target.
3. Duplicate enum-variant, struct-field, test-attribute, and crate-root export checks.
4. Example-import to crate-root export resolution.
5. Exact `git am --3way` application to the Series-15 baseline.
6. Exact final Git tree equality with the authored repository.
7. Byte-for-byte source archive reproduction.
8. Gzip and SHA-256 verification of every deliverable.

## Required canonical-shell verification

This environment does not contain Cargo, rustc, rustfmt, Clippy, or Nix. Before merge, run in the project development shell:

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Run the repository's normal Nix verification lane as well.

## Trust boundary

Series 16 proves deterministic relationships among versioned local artifacts. It does not prove:

- deletion of copies outside the governed bundle;
- compliance with a particular law or retention regime;
- publisher or signer authenticity without external verification;
- correctness of an external privacy mechanism;
- uniqueness of human participants;
- correctness of externally supplied logical epochs.
