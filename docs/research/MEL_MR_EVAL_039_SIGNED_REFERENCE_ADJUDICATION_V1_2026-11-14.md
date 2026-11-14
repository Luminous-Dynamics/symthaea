# MEL-MR-EVAL-039 — Signed, privacy-preserving human reference adjudication

**Date:** 14 November 2026  
**Status:** First implementation tranche; exact-head qualification pending  
**Tracking:** [Issue #12098](https://github.com/Luminous-Dynamics/symthaea/issues/12098)  
**Parent:** [EVAL-037 / PR #12050](https://github.com/Luminous-Dynamics/symthaea/pull/12050)

## Why this is necessary

EVAL-037 requires reviewer-set and review-record commitments, a minimum reviewer count and an independence attestation before a reference can be marked Validated. Those fields provide integrity metadata, but do not prove that reviewers controlled different keys or that judgments were authorized for the exact task.

EVAL-039 adds a cryptographic evidence boundary around the human review process while keeping real-name mappings outside public benchmark artifacts.

## Implemented v1 contract

- A frozen task digest binds study, dataset/version, track, sample, split, source lineage, operation, instruction, target selector, EVAL-037 envelope, reference audio, renderer profile, review protocol and consent scope.
- A study trust root verifies a signed trust bundle with bounded validity, issuer keys/status and a canonical revoked-credential list.
- A trusted issuer signs a pseudonymous reviewer credential with study, role, reviewer public key, validity interval, independence-group commitment and conflict-screening commitment.
- Each reviewer signs a structured judgment over the task digest, credential digest, reviewer-key fingerprint, adequacy state, separate judgment dimensions, confidence, consent-scope digest and review timestamp.
- Panel policy requires at least two reviewers and two issuer-attested independence groups. Distinct public-key fingerprints are checked; the same public key cannot count twice.
- Credentials must match the task's study and required role; issuer/credential revocation, wrong key, stale task, wrong consent scope, future review time and malformed ratings reject.
- Adequacy must be unanimous to retain Validated. Any panel disagreement becomes Ambiguous; no majority vote silently converts disagreement into validation.
- The deterministic receipt binds task, external trust-root identity, signed trust-bundle digest, policy digest, signed review digests, unique key fingerprints, independence-group commitments and adjudication result.
- Receipt replay reruns panel evaluation against the supplied credentials, signed judgments, trust root, trust bundle and policy.
- Ten authored Rust tests cover known-answer signing/replay, disagreement, task/reference substitution, signature mutation, credential/issuer revocation, duplicate key, consent mismatch, independence threshold, trust-root tampering, bundle expiry, future timestamps, invalid ratings/policy and receipt substitution.

## Security and governance limits

A valid Ed25519 signature proves possession of the matching private key and integrity of the signed payload. It does not prove legal identity, expertise, independence, lack of bias or correctness of a musical judgment.

The independence-group and conflict-screening values are commitments signed by the study issuer. Distinct groups are only as credible as the issuer's enrollment process and study governance. The evaluation timestamp is supplied by the caller; production deployment needs a trusted time source and authenticated trust-root rotation/revocation procedures.

The consent-scope digest binds a declared scope; it does not itself prove that informed consent was obtained or that a use is legally permitted. Dataset/component rights remain governed separately by RIGHTS-001 and INTEGRATION-001.

No real participants, private keys, identities or real music assets are included. The static fixture is metadata-only and does not qualify a panel. Unit tests use deterministic synthetic keys strictly as test fixtures.

## Relationship

- EVAL-037 / #12044 / PR #12050: validity envelopes and reference adequacy.
- EVAL-021 / #9184: blinded judgments, listener/item-level calibration and unseen-renderer evaluation.
- EVAL-035 / #12003 / PR #12007: actual benchmark bytes and lineage-bound receipts.
- RIGHTS-001 / #12029: component and purpose-specific rights.
- INTEGRATION-001 / #12035: binding rights, bytes and evaluation authorization.

## Validation boundary

The production module, standalone harness mirror, Cargo dependency wiring, typed schema, static synthetic fixtures and exact-head Rust 1.96 workflow are committed in this tranche. Ten Rust test functions are authored. Formatting, compilation, tests, fixture validation and Clippy remain pending; no runtime pass is claimed until the exact-head workflow succeeds.
