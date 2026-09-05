# Xenia Guarded Witness Frontier V0.1

## Status

Authored draft. No Rustfmt, compile, test, Clippy, lock-freshness, or qualification claim exists until the exact-head workflow executes successfully and the checked-in `Cargo.lock` is byte-identical to Cargo's candidate.

## Purpose

This tranche composes three already-separated evidence boundaries without adding a new authority mechanism:

1. `symthaea-xenia-authority` authenticates/freshens Xenia witness chronology and returns `VerifiedXeniaWitnessFrontierV1`.
2. `symthaea-xenia-witness-frontier-adapter` converts only that opaque proof into #452's transport-neutral `VerifiedExternalWitnessFrontierV1` while retaining the original Xenia evidence.
3. `symthaea-qualification-witness-frontier-sqlite` (#456) audits local witness history after acquiring a SQLite `BEGIN IMMEDIATE` writer barrier and performs #452 ancestry classification while the barrier remains live.

The missing composition was provenance retention across the final guarded decision.

## Boundary

```text
Xenia durable anchor
        +
fresh affine challenge response
        +
subject/policy-bound VerifiedAuthorityTime
        ↓
VerifiedXeniaWitnessFrontierV1
        ↓
#469 XeniaExternalWitnessFrontierV1
        ├── source-specific Xenia proof
        └── generic verified external frontier
                         │
local #449 SQLite       │
        ↓               │
BEGIN IMMEDIATE         │
        ↓               │
full local audit        │
        ↓               │
#452 ancestry classification
        ↓
GuardedXeniaWitnessFrontierDecisionV1
        ├── guarded local decision
        └── exact Xenia evidence that justified it
```

The composition does not accept raw Xenia fields, a caller-built external claim, a copied publication disposition, or an unguarded local snapshot.

## Provenance-retaining permits

The crate exposes Xenia-specific wrappers rather than handing callers a detached generic permit:

```text
GuardedXeniaPublicationPermitV1
    = live #456 publication permit
      + exact Xenia evidence

GuardedXeniaAnchorPermitV1
    = live #456 re-anchor permit
      + exact Xenia evidence
```

The underlying #456 permit remains a private field of the wrapper. A later Xenia-aware publisher/anchor adapter can therefore require the Xenia-specific permit type rather than accepting a copied boolean or generic permit whose source evidence has been discarded.

The wrapper exposes only the read-only facts needed for later composition:

- witness identity;
- guarded local frontier;
- recovery relation for the re-anchor case;
- exact retained `XeniaExternalWitnessFrontierV1`.

## Lifetime invariant

`GuardedXeniaWitnessFrontierDecisionV1` borrows both:

- the live SQLite publication guard;
- the exact Xenia evidence object.

It therefore cannot outlive either trust input.

Its publication/re-anchor permits borrow the guarded decision, so they cannot outlive the writer barrier or the retained external chronology.

This intentionally makes the trust lifetime visible to Rust's type system.

## Closed-world outcomes

Exact current equality:

```text
local audited frontier == verified current Xenia frontier
        ↓
AnchoredCurrent
        ↓
PublishAllowed
        ↓
GuardedXeniaPublicationPermitV1 only
```

Local state ahead on the same proven reservation chain:

```text
Xenia frontier is an exact historical prefix
        +
local audited frontier is newer
        ↓
LocalAheadVerifiedDescendant
        ↓
AnchorRequired
        ↓
GuardedXeniaAnchorPermitV1 only
```

Rollback/fork/missing-prefix states remain contained by #452/#456 and produce neither publication nor re-anchor permission here.

## End-to-end tests authored

The integration tests do not use a test-only constructor for Xenia evidence.

They:

1. create a real #449 SQLite witness sequence store;
2. reserve witness chronology;
3. construct and Ed25519-sign the exact Xenia V1 durable anchor;
4. generate a real affine Xenia currentness challenge through OS entropy;
5. sign the fresh Xenia observation over that exact challenge;
6. obtain multi-authority `VerifiedAuthorityTime` bound to the exact anchor/challenge/freshness policy;
7. consume the pending Xenia challenge to obtain `VerifiedXeniaWitnessFrontierV1`;
8. adapt through #469;
9. acquire the #456 SQLite writer barrier;
10. classify while the barrier is live.

Two cases are frozen:

- exact current Xenia/local equality yields only a provenance-retaining publication permit;
- Xenia sequence 1 as a proven prefix of local sequence 2 yields only a provenance-retaining re-anchor permit.

The second case additionally verifies that the re-anchor permit carries the newer local frontier while retaining the older exact Xenia evidence that established why re-anchoring is required.

## Cargo.lock diagnostic model

The checked-in lock currently predates several recent source-less Agency workspace packages, including the qualification-witness and Xenia authority layers.

The dedicated workflow therefore distinguishes diagnostic compilation from release qualification.

Before compilation, Cargo may add committed source-less workspace/path package nodes. The workflow rejects:

- any package removal;
- any new registry/Git package;
- any mutation of an unrelated existing package;
- an unexpected `symthaea-xenia-authority` dependency surface;
- a `getrandom` dependency outside the reviewed 0.2 line;
- an unexpected #469 adapter dependency surface;
- an unexpected guarded-composition dependency surface.

This lets Rustfmt/tests/Clippy run against Cargo's exact candidate even when inherited local package nodes are missing from the checked-in lock.

The final gate still requires:

```text
checked-in Cargo.lock == Cargo-generated candidate
```

So a compiler-clean stale-lock run is diagnostic evidence only, never qualification.

## Deliberate non-claims

This crate does **not**:

- verify Xenia signatures itself;
- create/freshen the Xenia challenge itself;
- verify trusted time itself;
- implement the Xenia anchor store;
- reconcile an `OutcomeUnknown` external write;
- create or mutate #449 witness sequence state;
- release the SQLite writer barrier early;
- publish an attestation;
- create consent, capability, budget, reservation, retry permission, or execution authority.

It only preserves the causal relationship between already-verified external chronology and an already-audited guarded local recovery decision.

## Next boundary

After this stack receives real compiler/lock evidence, the next production-facing component should accept **only** `GuardedXeniaPublicationPermitV1` for release publication, and **only** `GuardedXeniaAnchorPermitV1` for the Xenia re-anchor path.

The re-anchor path must continue to use #457's `Applied / ProvenNotDispatched / OutcomeUnknown` semantics and must never hold the SQLite writer barrier indefinitely while an external result is ambiguous.

The publication path should consume a short-lived Xenia-specific publication permit and emit privacy-minimized evidence tying the publication to:

- the exact local witness frontier;
- the exact Xenia anchor/currentness evidence;
- the exact published artifact commitment.

Evidence chronology remains distinct from execution authority throughout.
