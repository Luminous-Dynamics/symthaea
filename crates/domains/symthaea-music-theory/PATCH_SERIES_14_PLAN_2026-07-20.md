# Symthaea Music Theory Patch Series 14

Date: 2026-07-20

## Objective

Make authenticated listening studies safer to operate and combine by adding
one-response book assignment, study-scoped pseudonymous persistence,
small-cell-suppressed publication, strict legacy migration, and source-consistent
multi-study release portfolios.

## Patch groups

### A. Pseudonymous persistence

- Replace raw listener tokens in calibration judgments and study links with
  deterministic study-scoped SHA-256 pseudonyms.
- Keep raw tokens only inside private signed-response and verification
  boundaries.
- Prevent cohort changes within one study from creating a second persisted
  identity.
- Document that deterministic pseudonyms require high-entropy administrator
  tokens and are not anonymity.

### B. Deterministic assignment and duplicate control

- Add a versioned, self-auditing private assignment registry.
- Balance randomized books deterministically across active/completed
  enrollments.
- Permit exactly one enrollment per study-scoped pseudonym.
- Reject wrong-book, completed, revoked, duplicate, or forged enrollment state.
- Complete assignment and attach study judgments transactionally in memory.

### C. Privacy-safe public summaries

- Add release small-cell policy and exact public projections.
- Omit counts and metrics for suppressed study, renderer, and source cells.
- Bind the public projection into bundle SHA-256 and bundle audit.
- Remove exact portfolio counts and gate observations when publication
  thresholds are not met.

### D. Strict Series-13 migration

- Advance evidence bundles to v7 and integrity to v4.
- Verify the exact Series-13 v6/v3 integrity contract before migration.
- Convert legacy raw tokens in both study-link ledgers and exact attached
  judgments.
- Recompute private summaries, public summaries, judgment summaries, and bundle
  integrity.
- Refuse tampered or unknown legacy envelopes.

### E. Release-safe study portfolios

- Aggregate exact authenticated studies into a private SHA-bound portfolio.
- Produce a separate suppressed public report.
- Require all source bundles to share engine, source revision/tree, corpus,
  threshold, and evidence identities.
- Add pooled acceptance, minimum per-study size, study dominance, and
  heterogeneity gates.
- Count assessor-study units without claiming cross-study human uniqueness.

### F. Operator tooling

- Generate a private assignment registry with study books.
- Assign, audit, and revoke enrollments without persisting raw tokens.
- Support assignment-aware response attachment.
- Export public single-study reports.
- Export separate private and public multi-study portfolios.
- Remove raw listener tokens from shell arguments and verifier process listings.

### G. Adversarial regressions

- Verify duplicate attachment leaves both bundle and registry unchanged.
- Reject a response for an unassigned book without mutation.
- Verify exact legacy judgment pseudonym migration.
- Refuse portfolios that combine different source revisions.
- Verify suppressed portfolio projections omit counts and gate observations.

## Compatibility policy

- Existing non-assignment attachment APIs remain available for compatibility,
  but the assignment-aware path is the release workflow.
- Series-13 bundle v6 files remain readable only through explicit, fail-closed
  migration.
- Private study summaries remain exact; suppression applies only to explicit
  public projections.
- Portfolio evidence is revision-specific and cannot pool different engine or
  analyzer identities.
- Cross-study pseudonyms are intentionally unlinkable, so portfolio counts do
  not claim unique human participants.

## Verification requirements

Before merge in the canonical development shell:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

Recommended operator smoke tests:

```bash
cargo run --example evidence_study_books -- --help
cargo run --example evidence_study_assignment -- --help
cargo run --example evidence_listener_response_payload -- --help
cargo run --example evidence_attach_listener_response -- --help
cargo run --example evidence_study_public_report -- --help
cargo run --example evidence_study_portfolio -- --help
cargo run --example evidence_migrate_bundle -- --help
```

## Trust statement

This series makes duplicate control, source consistency, public suppression, and
provenance mechanically auditable. It does not establish participant
independence, anonymous credentials, differential privacy, signer authority,
study ethics, cultural validity, or perceptual universality.
