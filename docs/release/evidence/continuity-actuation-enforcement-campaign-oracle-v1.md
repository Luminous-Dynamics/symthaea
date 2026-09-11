# Continuity Actuation-Enforcement Campaign Oracle V1

## Status

`STRUCTURAL_ORACLE_ONLY / NOT_QUALIFIED`

This evidence note freezes the independent standard-library Python oracle added on branch `architecture/continuity-actuation-enforcement-campaign-v1`.

It grants no execution authority, verifier authority, adapter qualification, or physical-enforcement claim.

## Purpose

The oracle independently validates the additive campaign-coherence contract that sits above `StructurallyCompleteActuationEnforcementEvidenceV1`.

The theorem is deliberately narrow:

```text
nine structurally complete records
!= one coherent qualification campaign
!= truthful evidence
!= verifier qualification
!= live resource enforcement
```

## Canonical campaign inputs

The oracle requires one exact:

- `complete_set_id`;
- `enforcement_profile_id`;
- `authentication_profile_id`;
- backend ID;
- backend implementation digest and generation;
- enforcement-boundary implementation digest;
- one-use mechanism digest;
- enforcement-profile generation;
- campaign nonce;
- qualification-harness implementation digest;
- scenario-suite manifest digest;
- environment manifest digest;
- topology/dependency manifest digest;
- hardware/firmware manifest digest;
- toolchain/container/Nix realization digest;
- campaign start/end interval;
- set of exactly nine obligation records.

The backend/boundary fields are explicit even though some are also transitively committed by V1 profile/complete-set identities. The campaign must be self-describing enough for a verifier to detect profile/backend substitution without reverse-engineering an opaque parent hash.

All digest fields are 32-byte lowercase hex and must be non-zero. Backend and enforcement-profile generations are positive `u64` values. A platform with no meaningful hardware/firmware dimension should bind an explicit non-applicability manifest digest rather than a zero/missing field.

The manifest and each evidence-record object use an exact closed field set. Unknown/shadow fields fail closed so policy or result semantics cannot exist outside the canonical preimage.

## Closed-world obligations

Exactly these nine obligations are accepted:

1. `boundary_identity`;
2. `same_boundary_checks_and_mutates`;
3. `durable_monotonic_fence`;
4. `reject_stale_generation`;
5. `reject_replay`;
6. `reject_deny_disposition`;
7. `emergency_stop_dominates`;
8. `one_use_permit_consumption`;
9. `crash_recovery_preserves_fence`.

The oracle rejects missing/duplicate/unknown obligations and duplicate record IDs.

Every evidence observation must fall inside the exact campaign interval.

## Canonicalization

Input record order is not authoritative. The oracle canonicalizes records in the fixed obligation order above.

The campaign evidence-manifest digest is SHA-256 over a domain-separated sequence of:

```text
obligation-name-length || obligation-name || record-id || observed-at-u64-le
```

for all nine obligations in canonical order.

The domain-separated campaign preimage then binds, in exact order:

```text
complete-set ID
enforcement-profile ID
authentication-profile ID
backend ID
backend implementation digest
backend generation
boundary implementation digest
one-use mechanism digest
enforcement-profile generation
campaign nonce
harness implementation digest
scenario-suite manifest digest
environment manifest digest
topology/dependency manifest digest
hardware/firmware manifest digest
toolchain realization digest
campaign start
campaign end
canonical evidence-manifest SHA-256
```

SHA-256 is used here as an independent audit/preimage digest, not as a replacement for any future Rust campaign-ID algorithm. A future Rust implementation should reproduce the same semantic preimage and may domain-separate/hash it under the repository's selected identity algorithm.

## Frozen parity fixture

The exact positive fixture is checked in at:

`tests/fixtures/continuity/actuation_enforcement_campaign_v1.json`

The exact expected structural output is:

`tests/fixtures/continuity/actuation_enforcement_campaign_v1.expected.json`

For the deterministic fixture, the oracle produces:

- evidence-manifest SHA-256: `f0c0690664c45f4ef0f875c88e6ba5a8150bd5649f5f7cf9e8b7906fbdff0c3a`;
- canonical-preimage SHA-256: `b1c898d59fe1218e2cb9ccd1eb18cd15163e9c5e855990c10307cbae29a3ddeb`.

The evidence-manifest hash remains unchanged from the earlier fixture because the exact same nine obligation record IDs/timestamps are used; the campaign preimage changes because backend/profile lineage is now bound explicitly.

## Executed self-tests

The oracle was syntax-checked and its in-memory self-test executed before commit.

The self-test establishes:

- reversing input record order leaves the canonical preimage unchanged;
- an observation before the campaign start fails closed;
- duplicate obligations fail closed;
- a zero environment digest fails closed;
- duplicate record IDs fail closed;
- unknown top-level fields fail closed;
- unknown per-record fields fail closed;
- toolchain-realization drift changes the canonical preimage;
- backend-generation drift changes the canonical preimage;
- enforcement-profile substitution changes the canonical preimage.

## Required future Rust parity

The additive Rust campaign wrapper should be accepted only after an independent parity test demonstrates:

1. the exact same closed-world obligation ordering;
2. the exact same campaign interval semantics;
3. exact campaign/environment/toolchain binding;
4. exact enforcement/authentication/backend/boundary/one-use lineage binding;
5. no caller-controlled omission of a required campaign identity;
6. no uncommitted shadow fields or extension semantics in V1;
7. canonical evidence manifest over the exact nine V1 record IDs;
8. exact parity against the checked-in positive fixture;
9. mixed-campaign or environment/backend/profile-drift cases fail closed;
10. the wrapper remains descriptive/non-authoritative until verifier-owned admission in #1550.

## Non-claims

This oracle does not establish that any evidence record is true. It does not authenticate a campaign, establish current verifier adoption, prove the backend actually rejects stale fences, or grant mutation authority.

The physical theorem remains downstream in #1528.
