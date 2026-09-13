# Replicator Safety Kernel — External Monotonic Anchor v0.1

**Status:** Reference anti-rollback contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## Purpose

A locally stored integrity hash does not prove that the local state is the newest state. A restored disk, process snapshot, or stale durable image can roll back both data and its ordinary integrity metadata together.

The governing distinction is:

```text
integrity-protected local state
!=
rollback-resistant monotonic state
```

This contract defines the external observation RSK requires before local anti-rollback state can support new positive authority.

## External observation

Production code conceptually consumes an opaque, externally authenticated observation binding:

```text
VerifiedMonotonicAnchorObservation {
    namespace,
    epoch,
    counter,
    anchored_state_digest,
    provider_profile,
    provider_identity,
    trust_snapshot_digest,
    freshness_evidence,
}
```

The reference implementation uses `AuthenticatedMonotonicAnchorEvidence` only as a stand-in for the output of a separate provider/trust verifier. It does not authenticate the provider itself.

## Exact local state binding

For the schema-registry tracker, the anchored digest commits the complete canonical local anti-rollback state:

- registry identity;
- highest accepted sequence;
- accepted snapshot digest;
- latest accepted issuance time;
- registry policy digest;
- forked/non-operational state;
- all retained schema identities and lifecycle states.

The state digest is domain-separated and canonicalized. A matching sequence number alone is insufficient.

## Namespace and epoch

Namespace and epoch are exact scope boundaries.

- namespace mismatch -> freeze;
- epoch mismatch -> freeze;
- epoch changes only through governed fresh-epoch recovery;
- ordinary operation cannot reset the counter by changing namespace/epoch.

## Equality rule

For the current schema-registry profile:

```text
anchor.counter == local.highest_sequence
anchor.anchored_state_digest == Digest(local.complete_tracker_state)
```

Both equalities are required.

If the local side is ahead, freeze. If the anchor is ahead, freeze. If the counter matches but the digest differs, freeze.

The verifier must not choose the numerically newer or more permissive side automatically.

## Freshness

The full trusted evaluation interval must be contained in the observation validity interval.

Unavailable, stale, expired, or ambiguous anchor evidence removes positive-authority eligibility. It never extends old state validity.

## Crash consistency

Updating local durable state and the external monotonic observation can fail between writes.

Two possible partial states are therefore expected:

```text
local N+1, anchor N
local N,   anchor N+1
```

Both are fail-closed states. Restart freezes new positive authority until governed reconciliation/recovery supplies evidence sufficient to establish one valid history.

This v0.1 contract intentionally prefers fail-closed behavior over automatic repair.

## No automatic repair

On mismatch, RSK must not automatically:

- overwrite local state from the anchor;
- overwrite the anchor from local state;
- choose the larger counter;
- choose the newer timestamp;
- reset/decrement the anchor;
- create a fresh epoch silently.

State replacement is a recovery action, not an ordinary verification result.

## Forked local state

A matching external observation cannot repair a local tracker that is already forked/non-operational.

```text
local.forked == true -> freeze
```

The anchor proves continuity of a state, not constitutional validity of that state.

## Reference profile

The current reference digest is:

```text
Domain = "symthaea.rsk.monotonic-anchor-state.v1\0"
Digest = SHA256(Domain || canonical_json_complete_tracker_state)
```

The committed abstract golden vector binds tracker digest:

```text
798b6606fac22ff5fd09b5baf3255bed88448ffbcd725959dbaa4618b9285c00
```

## Verification versus mutation

The comparison verifier does not expose an API to advance or reset the external anchor.

Production deployments should keep observation verification, anchor mutation, ordinary operation, and recovery as distinct roles where practical.

## Composition

The intended chain is:

```text
verified registry semantics
    -> local AntiRollbackState
    + fresh external monotonic observation
    -> exact counter/state-digest agreement
    -> anchored anti-rollback predicate
```

The monotonic observation cannot mint grants, clear negative state, widen budgets, extend expiry, or translate authority across schemas.

## Required tests

The reference test profile covers:

- exact golden acceptance;
- unavailable observation;
- local-ahead mismatch;
- anchor-ahead mismatch;
- same-counter/different-digest mismatch;
- namespace/epoch/provider/trust substitution;
- stale or invalid trusted interval;
- locally forked state;
- authority-relevant tracker mutations changing the digest;
- crash ambiguity in either write order;
- software fixture remaining non-authoritative.

## Production promotion gates

Production still requires independently verified provider evidence, trusted-time integration, durable recovery semantics, exact runtime/build identity, hostile rollback testing, and Class A executed evidence.

This contract does not establish production admission.
