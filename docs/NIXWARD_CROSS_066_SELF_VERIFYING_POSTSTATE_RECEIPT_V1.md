# Nixward CROSS-066 — Self-Verifying Post-State Receipt V1

## Purpose

CROSS-066 closes a false-green surface in durable post-state receipts.

Before this tranche, a serialized receipt could retain a state digest and a `Proven` claim without retaining enough underlying semantic state to independently recompute the digest or re-evaluate the effect predicate.

A digest prevents accidental or malicious mutation **after** the digest is established. It does not, by itself, prove that the original state represented by that digest was truthful.

CROSS-066 therefore makes the durable receipt self-verifying at the semantic layer.

## Persisted state

The receipt now retains:

- exact systemd Unit object path;
- LoadState;
- ActiveState;
- SubState;
- UnitFileState;
- Service.Result;
- post InvocationID;
- systemd manager unique owner.

The same semantic fields are present in the observer-produced post-state observation.

The canonical semantic state digest is recomputed from these fields using the existing domain-separated hash. Stability samples must carry exactly the recomputed digest.

## Receipt verification

`NixPostStateReceiptV1::validate_shape()` now independently recomputes:

1. the effect digest from the receipt's bound effect fields;
2. the semantic state digest from persisted post-state fields;
3. the effect-specific postcondition from persisted Job and state fields.

The stored `postcondition` must equal the independently recomputed result.

For `Proven`, the receipt must therefore demonstrate from its own persisted data:

- the expected native systemd Job type;
- the exact target unit;
- the exact numeric Job ID/path correlation;
- terminal Job result `done`;
- the operation-specific ActiveState or UnitFileState predicate;
- restart invocation change where required;
- a valid stability sequence whose final sample matches the recomputed semantic state and receipt identity.

## Why the raw stability digest is no longer sufficient

An attacker who could fabricate:

```
state_digest = H("active")
sequence_digest = H(sequence containing that digest)
claim = Proven
```

could previously satisfy the purely structural sequence checks without proving that the persisted receipt represented an actually observed Active state.

CROSS-066 defeats that construction because the verifier reconstructs the state digest from the durable state fields and requires the sequence's state digest to equal that value.

The same principle applies to postcondition semantics: the receipt cannot declare `Satisfied` while persisting an incompatible ActiveState, UnitFileState, JobType, JobResult, unit, or invocation relation.

## Observer boundary

The raw state and receipt remain serializable records. Trust still comes from their provenance chain:

```
read-only observer
    -> observer-sealed post-state observation
    -> observer-sealed stability sequence
    -> durable self-verifying receipt
```

The receipt verifier does not treat an opaque observer identity string as proof.

## State vocabulary

The observer keeps strict closed parsing for LoadState, ActiveState, and UnitFileState.

`Service.Result` remains an open string rather than a closed enum. The observer requires it to be non-empty and preserves it exactly so future systemd result vocabulary is recordable without silently treating unknown values as known success.

## Claim ceiling

Self-verification of a receipt is not external attestation.

It does not prove:

- cryptographic authenticity of the D-Bus peer;
- atomic CAS semantics against concurrent actors;
- absence of an unobservable transient outside the persisted properties;
- equality of file contents when only FragmentPath/DropInPaths identities match.

It does make the durable receipt's declared state and postcondition independently derivable from its persisted fields rather than trusting a precomputed digest or claim label.

## Qualification

The branch is not PASS because it is mergeable or because tests are queued.

Exact-head GitHub Actions must pass the checked-in Nixward qualification path before this tranche is called qualified.
