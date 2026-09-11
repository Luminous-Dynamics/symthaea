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
- `campaign_nonce`;
- qualification-harness implementation digest;
- scenario-suite manifest digest;
- environment manifest digest;
- topology/dependency manifest digest;
- hardware/firmware manifest digest;
- toolchain/container/Nix realization digest;
- campaign start/end interval;
- set of exactly nine obligation records.

All digest fields are 32-byte lowercase hex and must be non-zero.

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

The campaign preimage is domain-separated and contains the eight fixed campaign digests, start/end timestamps, and the canonical evidence-manifest SHA-256.

SHA-256 is used here as an independent audit/preimage digest, not as a replacement for any future Rust campaign-ID algorithm. A future Rust implementation should reproduce the same semantic preimage and may domain-separate/hash it under the repository's selected identity algorithm.

## Executed self-test fixture

The oracle was syntax-checked and its in-memory self-test executed before commit.

For the deterministic fixture with digest bytes `01..08`, record-ID bytes `20..28`, campaign interval `[1000, 2000]`, and observations `1100..1108`, it produced:

- evidence-manifest SHA-256: `f0c0690664c45f4ef0f875c88e6ba5a8150bd5649f5f7cf9e8b7906fbdff0c3a`;
- canonical-preimage SHA-256: `6781f81213b8eb04a022a1d11001e78ffa11cd2d76edc331da5e7717f15cd78f`.

The self-test additionally established:

- reversing input record order leaves the canonical preimage unchanged;
- an observation before the campaign start fails closed;
- duplicate obligations fail closed;
- a zero environment digest fails closed;
- duplicate record IDs fail closed;
- toolchain-realization drift changes the canonical preimage.

## Required future Rust parity

The additive Rust campaign wrapper should be accepted only after an independent parity test demonstrates:

1. the exact same closed-world obligation ordering;
2. the exact same campaign interval semantics;
3. exact campaign/environment/toolchain binding;
4. no caller-controlled omission of one of the eight campaign digests;
5. canonical evidence manifest over the exact nine V1 record IDs;
6. mixed-campaign or environment-drift cases fail closed;
7. the wrapper remains descriptive/non-authoritative until verifier-owned admission in #1550.

## Non-claims

This oracle does not establish that any evidence record is true. It does not authenticate a campaign, establish current verifier adoption, prove the backend actually rejects stale fences, or grant mutation authority.

The physical theorem remains downstream in #1528.
