# Agency witness frontier recovery V0.1

## Purpose

This profile defines the recovery semantics between the local durable witness-sequence history from #449 and a separately authenticated/current-enough external frontier anchor.

It deliberately does **not** implement Xenia, TPM NV, SCITT, or any other anchor transport. The goal is to freeze the state machine before coupling it to one external system.

The central rule is:

> A larger local counter is not proof that local state descends from the externally trusted history.

For local-ahead recovery, the externally trusted reservation head must match the audited local chain at the external anchor's exact sequence.

## Inputs

### Local history

`LocalWitnessFrontierHistory` is a trusted adapter contract with three responsibilities:

1. audit the complete local witness reservation chain and current frontier;
2. return the current witness frontier point;
3. return the exact historical reservation head at an explicitly requested sequence.

The classifier never infers a historical prefix from the current counter.

### External anchor

`ExternalWitnessFrontierClaimV1` contains:

- anchor schema version;
- external source ID;
- source epoch;
- source-local monotonic sequence;
- witness ID;
- witness high watermark;
- witness reservation head;
- exact witness frontier-statement digest;
- freshness-evidence digest.

Before the claim becomes opaque `VerifiedExternalWitnessFrontierV1`, two checks occur:

1. this crate validates the internal frontier commitment;
2. an `ExternalWitnessFrontierVerifier` validates the external source.

The external verifier contract is deliberately strong: it must establish authentication/integrity **and currentness/freshness under that source's own policy**.

A cryptographically valid old signed anchor is not sufficient if a newer source frontier may be suppressed.

The `freshness_evidence_digest` is retained as evidence binding; this generic crate does not interpret the underlying challenge, ledger checkpoint, quorum proof, trusted time, or transparency proof.

## Recovery relation

The classifier emits one of the following states.

### `EmptyUnanchored`

No local reservations and no external anchor exist.

Disposition: `AnchorRequired` before publishing an anchored-profile witness result.

### `InitialAnchorRequired`

Local audited history exists, but there is no external anchor.

Disposition: preserve local state and establish an external anchor.

### `AnchoredCurrent`

Local high watermark, reservation head, and frontier-statement commitment exactly equal the verified external anchor.

Disposition: `PublishAllowed`.

This is the only V0.1 state that directly permits publication under the anchored profile.

### `LocalAheadVerifiedDescendant`

The local high watermark is greater than the external high watermark **and** the local audited reservation head at the external sequence exactly equals the trusted external reservation head.

Disposition: preserve local state, advance the external anchor, then publish.

Do not roll local state backward to make it equal the older external anchor.

### `RollbackOrMissingLocal`

The external anchor proves a higher sequence than local state, or external state exists while the local witness history is absent.

Disposition: `Contained`.

### `DivergentAtSameHeight`

Local and external states claim the same sequence but disagree on reservation head/frontier commitment.

Disposition: `Contained`.

### `DivergentTrustedPrefix`

Local sequence is larger than external, but the local reservation head at the externally trusted sequence is not the trusted head.

Disposition: `Contained`.

This is the subtle case that simple counter comparison misses.

## Recovery table

```text
local absent, external absent
    -> EmptyUnanchored
    -> anchor required

local present, external absent
    -> InitialAnchorRequired
    -> anchor local current frontier

local absent, external present
    -> RollbackOrMissingLocal
    -> contain

local height < external height
    -> RollbackOrMissingLocal
    -> contain

local height == external height
    + same head/digest
    -> AnchoredCurrent
    -> publication allowed

local height == external height
    + different head/digest
    -> DivergentAtSameHeight
    -> contain

local height > external height
    + local historical head at external height == external head
    -> LocalAheadVerifiedDescendant
    -> preserve local, re-anchor, then publish

local height > external height
    + historical head missing or different
    -> divergence / unavailable-prefix failure
    -> contain
```

## Why ancestry is required

Consider two histories:

```text
trusted: A -> B
local:   A -> B -> C
```

The local state is safely ahead.

But this is different:

```text
trusted: A -> B
local:   A -> X -> C
```

Both can report local counter `3` and trusted counter `2`. Counter comparison alone would call both cases "local ahead".

The V0.1 classifier explicitly asks the local-history adapter for the head at sequence 2. Only the first history is a verified descendant.

## Publication boundary

The classifier intentionally separates local durability from publication readiness.

```text
AnchoredCurrent
    -> PublishAllowed

EmptyUnanchored
InitialAnchorRequired
LocalAheadVerifiedDescendant
    -> AnchorRequired

RollbackOrMissingLocal
DivergentAtSameHeight
DivergentTrustedPrefix
    -> Contained
```

This prevents the external anti-rollback layer from being merely advisory.

A later composed witness service should persist/sign locally first, obtain or advance the external anchor, reclassify to `AnchoredCurrent`, and only then release the attestation outside the anchoring boundary.

## External write ambiguity

This PR defines read/recovery semantics only.

A later anchor writer must not blindly retry an externally ambiguous append. It needs an idempotency key or source-specific reconciliation primitive so:

```text
anchor write outcome unknown
    !=
permission to append again
```

This is the same conservative outcome-unknown rule used elsewhere in the Agency action runtime.

## Freshness suppression non-claim

A signed external frontier proves only what that source signed.

If an attacker can suppress a newer source checkpoint and present an older still-valid one, this generic classifier cannot detect that by itself. The concrete external verifier must establish currentness, for example through an online challenge, fresh Xenia ledger checkpoint, witness quorum, TPM monotonic state, or a transparency mechanism with an appropriate freshness policy.

Therefore the correct composition is:

```text
external source evidence
    ↓
authentication + freshness verification
    ↓
VerifiedExternalWitnessFrontierV1
    ↓
ancestry-aware local recovery classifier
```

not:

```text
old valid signature
    ↓
TrustedExternalFrontier
```

## Tests authored

The V0.1 source suite covers:

- exact anchored equality allows publication;
- local-ahead state with matching trusted ancestor requires re-anchoring;
- a larger local counter with a wrong trusted prefix is contained;
- external-ahead state is classified as rollback;
- same-height different head is contained;
- externally anchored state with missing local history is contained;
- local state without an external anchor requires anchoring;
- internally inconsistent external frontier claims are rejected before source verification.

## Deliberate integration boundary

V0.1 does not yet provide a production `LocalWitnessFrontierHistory` adapter for #449's SQLite store because historical-prefix lookup is itself a security-sensitive storage API. That adapter should be added explicitly and tested against the exact DB reservation chain, rather than reaching around the sequence store with an ad-hoc SQL query.

Likewise, no Xenia/TPM/SCITT verifier is selected in this PR.

This keeps three responsibilities separately reviewable:

```text
#449 local durable history
        ↓
V0.1 ancestry/recovery semantics
        ↓
future concrete external anchor adapter
```

## Non-authority statement

External anchoring strengthens evidence chronology and rollback detection. It never creates, expands, restores, or substitutes for execution authority.
