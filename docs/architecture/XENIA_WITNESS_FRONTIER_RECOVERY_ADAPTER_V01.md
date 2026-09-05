# Xenia Witness Frontier Recovery Adapter V0.1

## Status

Authored draft. No compiler, test, Clippy, or qualification claim exists until the exact-head workflow executes and the checked-in `Cargo.lock` is byte-fresh.

## Purpose

This adapter is the intentionally small bridge between two already-separated trust domains:

1. `symthaea-xenia-authority` independently authenticates a Xenia durable witness anchor plus fresh challenge-bound currentness evidence and returns `VerifiedXeniaWitnessFrontierV1`.
2. `symthaea-qualification-witness-frontier` (#452) performs transport-neutral ancestry/rollback classification over `VerifiedExternalWitnessFrontierV1`.

Neither crate should absorb the other's semantics. This bridge therefore depends on both and translates only an already-verified Xenia object into the generic recovery representation.

## Invariant

```text
VerifiedXeniaWitnessFrontierV1
        │
        ├─ source id / epoch / sequence
        ├─ witness id
        ├─ witness high watermark
        ├─ reservation head
        ├─ frontier-statement digest
        └─ freshness-evidence digest
        │
        ▼
internal ExternalWitnessFrontierClaimV1
        │
        ▼
#452 structural frontier validation
        +
exact equality to the original verified Xenia projection
        │
        ▼
VerifiedExternalWitnessFrontierV1
```

The public adapter accepts no raw Xenia anchor, observation, challenge, time interval, external claim, signature, source sequence, or witness frontier fields.

## Why the generic verifier still runs

`VerifiedXeniaWitnessFrontierV1` already establishes Xenia authenticity/currentness. The adapter nevertheless routes its projection through `verify_external_witness_frontier_v1` so #452 independently checks its own closed-world structural contract, including the witness frontier-statement digest.

The adapter's private `ExactVerifiedXeniaSource` then requires every generic claim field to equal the already-verified Xenia snapshot. This is defense in depth against projection/refactor mistakes; it is not a second cryptographic verifier.

## Retaining source-specific evidence

The returned `XeniaExternalWitnessFrontierV1` owns both opaque proofs:

- `VerifiedXeniaWitnessFrontierV1`;
- `VerifiedExternalWitnessFrontierV1`.

#452 only needs the generic proof, but the retained Xenia proof preserves source-specific forensic commitments such as:

- Xenia anchor fingerprint;
- Xenia anchor operation id;
- signed observation timestamp.

Translation therefore does not erase useful source evidence.

## Explicit non-claims

This crate does **not**:

- verify Ed25519 signatures;
- generate or consume the Xenia currentness challenge;
- verify trusted time;
- establish Xenia source freshness;
- read or write witness SQLite state;
- append or reconcile an external anchor;
- classify ancestry itself;
- create a publication permit;
- create, restore, amplify, reserve, or spend execution authority.

It can only translate chronology evidence that the Xenia verifier has already accepted.

## Intended composition

```text
Xenia durable anchor
        +
fresh challenge-bound Xenia observation
        +
subject/policy-bound trusted time
        ↓
VerifiedXeniaWitnessFrontierV1
        ↓
this adapter
        ↓
VerifiedExternalWitnessFrontierV1
        ↓
#452 ancestry-aware classification
        ↓
#456 guarded publication / re-anchor decision
```

## Failure direction

If the Xenia proof cannot be represented as a structurally valid #452 claim, adaptation fails. There is no fallback to a caller-built claim and no "best effort" field dropping.

If a field is substituted between projection and generic verification, the private exact-source verifier rejects it.

Failure therefore loses availability rather than weakening chronology evidence.

## Cargo.lock discipline

The adapter introduces no registry dependency family. It creates one new source-less workspace package node with direct dependencies on:

- `symthaea-authority`;
- `symthaea-xenia-authority`;
- `symthaea-qualification-witness-frontier`;
- `thiserror`.

Its parent #467 also adds the already-workspace-pinned `getrandom 0.2` direct edge to `symthaea-xenia-authority` without hand-editing `Cargo.lock`.

The dedicated workflow allows Cargo to produce a diagnostic candidate only when the candidate contains exactly those reviewed changes. It still fails the final qualification gate until the checked-in lock is byte-identical to Cargo's output.

## Next boundary

After this bridge and its parents compile/qualify, the next useful composition is not another verifier. It is the concrete guarded recovery flow:

```text
XeniaExternalWitnessFrontierV1::external()
        +
#456 guarded audited local history
        ↓
#452 recovery relation
        ↓
PublishAllowed / AnchorRequired / Contained
```

That composition must continue to treat chronology evidence as distinct from execution authority.
