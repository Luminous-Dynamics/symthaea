# Replicator Safety Kernel — Xenia State-Witness Adapter v0.1

**Status:** Cross-system composition contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## Purpose

RSK's monotonic-anchor contract requires an externally authenticated observation
that agrees exactly with the complete local schema-registry anti-rollback state.
Xenia draft PR `Luminous-Dynamics/xenia-peer#335` introduces a generic witnessed
state commitment that can provide one evidence carrier for that observation.

This document freezes the adapter semantics between the two systems. It does not
make Xenia a replication authority and does not make a witnessed commitment
sufficient for production admission by itself.

## Responsibility split

Xenia may prove:

- exact commitment bytes were signed by a configured trusted-key quorum;
- the commitment is for the caller's expected namespace/target/epoch/trust context;
- exact replay, adjacent predecessor continuity, rollback, and same-counter fork facts.

RSK remains solely responsible for:

- defining the complete registry tracker state;
- computing the RSK tracker-state digest;
- trusted-time/freshness requirements;
- signer identity, lifecycle, revocation, and true failure-domain policy;
- exact local-state versus external-witness equality;
- fork/quarantine/revocation dominance;
- governed epoch recovery;
- every replication-authority decision.

A Xenia witness result never mints, widens, restores, or extends replication
authority.

## Fixed namespace

The v0.1 adapter namespace is:

```text
symthaea.rsk.schema-registry.v1
```

The relying-system admission path must treat this as an exact canonical machine
identifier. A namespace mismatch denies the external continuity predicate.

## Target identity

A witness chain must be bound to one exact deployed registry target, not merely a
human-readable registry name.

The adapter defines:

```text
TargetId = SHA256(
    "symthaea.rsk.xenia-target.v1\0"
    || canonical_bytes({
         deployment_identity,
         registry_id
       })
)
```

`deployment_identity` must come from the admitted deployment/runtime identity
boundary tracked by RSK build/runtime admission. `registry_id` is the exact
schema-registry tracker identity.

If an admitted deployment identity is unavailable or ambiguous, the adapter is
not eligible for production use.

## Epoch identity

RSK's monotonic-anchor policy uses a positive integer recovery epoch. Xenia's
generic commitment uses an opaque 32-byte epoch ID.

The adapter defines:

```text
EpochId = SHA256(
    "symthaea.rsk.xenia-epoch.v1\0"
    || TargetId
    || u64_be(rsk_recovery_epoch)
)
```

Changing the RSK recovery epoch therefore changes the Xenia epoch identity.
Ordinary operation cannot reset the Xenia counter by choosing a new epoch.
A new epoch is admitted only through RSK's governed fresh-epoch recovery path.

## Counter mapping

The Xenia commitment counter is exactly:

```text
xenia.counter == local_registry_tracker.highest_sequence
```

No offset is permitted.

This creates an important bootstrap requirement because the first accepted
registry snapshot has sequence 1 while Xenia requires counter-zero genesis to
have an all-zero predecessor.

### Epoch bootstrap

Before registry sequence 1 can be accepted under a newly admitted epoch, the
system must first establish a witnessed Xenia commitment for the canonical empty
RSK registry tracker at:

```text
counter = 0
previous_commitment = 0x00..00
state_digest = RSK_Digest(empty_epoch_tracker_state)
```

That counter-zero commitment is the epoch's external continuity genesis.

The first accepted registry state at sequence 1 then uses:

```text
counter = 1
previous_commitment = Fingerprint(counter_0_witnessed_commitment)
```

There is no special sequence-1 exception.

## State digest mapping

Xenia's opaque `state_digest` is exactly the 32-byte value represented by RSK's
existing lowercase-hex tracker digest:

```text
RSK_Digest(state)
  = SHA256(
      "symthaea.rsk.monotonic-anchor-state.v1\0"
      || canonical_json_complete_tracker_state
    )
```

The adapter decodes the 64-character lowercase SHA-256 hex representation to 32
raw bytes without rehashing it.

Therefore:

```text
xenia.state_digest == bytes(RSK_Digest(local_complete_tracker_state))
```

The complete tracker state remains the object already defined by
`RSK_MONOTONIC_ANCHOR_V0_1.md`.

## Previous-commitment mapping

For counter `N > 0`:

```text
xenia.previous_commitment
  == XeniaFingerprint(exact accepted commitment at counter N-1)
```

This predecessor is **not** derived from the RSK tracker-state digest alone.
The deployment must retain or independently recover the exact previously
accepted Xenia commitment/witness evidence.

If the previous commitment is unavailable or ambiguous, forward continuity is
not proven and new positive RSK authority freezes.

A larger counter cannot skip missing witness history under the adjacent v0.1
profile.

## Trust-context mapping

Xenia's `trust_context_digest` is not supplied by the witness bundle as a
self-asserted trust claim. RSK derives its expected value from the exact verified
external-witness policy for the current RSK epoch.

The canonical production target must bind at least:

- RSK anchor-policy identity;
- admitted Xenia state-witness profile/version;
- required witness quorum;
- verified witness signer identities or key bindings;
- verified signer lifecycle/revocation policy;
- verified failure-domain policy;
- trust-snapshot identity used to authenticate those facts;
- accepted signature-suite policy.

The exact canonical encoding remains a separate implementation artifact. Until
it is frozen and independently verified, this adapter remains non-production.

Changing the trust-context digest inside one Xenia epoch is not ordinary
continuity. It requires a governed trust/epoch transition rather than silent
reinterpretation.

## Timestamp mapping

Xenia's commitment timestamp is evidence metadata and a monotonic continuity
field. It is **not** RSK trusted time.

The adapter may bind a verified observation/issuance timestamp, but RSK must
continue to evaluate its separate trusted-time interval and freshness rules.

A valid Xenia witness timestamp cannot:

- make stale RSK evidence fresh;
- extend an authorization expiry;
- resolve trusted-time ambiguity;
- replace the trusted-time verifier.

## Provider/profile mapping

A verified Xenia state witness is one input to RSK's provider-verification
boundary, not a complete `VerifiedMonotonicAnchorObservation` by itself.

Before adapting it to RSK, a higher-level verifier must additionally establish:

- admitted Xenia provider/profile identity;
- exact relying context;
- verified trust snapshot;
- signer identity/lifecycle/revocation facts;
- required independent failure domains;
- required freshness/trusted-time facts.

Only after those checks may the RSK adapter construct the opaque provider
observation consumed by the exact-agreement verifier from #2239.

## Exact agreement remains mandatory

Even after Xenia witness verification succeeds, RSK must recompute its own local
tracker digest and require:

```text
xenia.counter == local.highest_sequence
xenia.state_digest == bytes(RSK_Digest(local.complete_tracker_state))
```

A valid signature over a different state does not authorize replacement of the
local state.

Local-ahead, witness-ahead, and same-counter/different-state cases all freeze new
positive authority.

## Crash ambiguity

The cross-system update spans at least two durability domains: local RSK durable
state and external witness/retention state.

Both partial states are expected failure modes:

```text
local N+1, witnessed N
local N,   witnessed N+1
```

Neither side wins automatically. Restart freezes new positive authority until a
governed reconciliation/recovery process proves one valid history.

Xenia witness continuity is evidence for that recovery process; it is not an
automatic repair instruction.

## Fresh-epoch recovery

A governed fresh epoch:

1. establishes the new RSK recovery epoch;
2. derives a new `EpochId`;
3. constructs the canonical fresh-epoch local tracker state;
4. witnesses a new counter-zero commitment with all-zero predecessor;
5. admits no prior grant, authorization token, runtime witness, or old Xenia
   predecessor implicitly.

Old witnessed history remains audit evidence but is not active continuity in the
new epoch.

## Negative-state dominance

A matching Xenia commitment cannot clear or override:

- RSK quarantine;
- revocation;
- local fork/non-operational state;
- expired evidence;
- failed containment/runtime-assurance predicates.

External continuity is a necessary predicate where configured, never a positive
authority source by itself.

## Required composition tests

A future executable adapter must cover at least:

1. exact counter-zero bootstrap;
2. sequence-1 binding to the witnessed bootstrap fingerprint;
3. exact target derivation sensitivity to deployment and registry identity;
4. exact epoch derivation sensitivity to recovery epoch;
5. exact raw-byte tracker digest mapping;
6. wrong Xenia namespace/target/epoch/trust-context denial;
7. local-ahead and witness-ahead denial;
8. same-counter/different-state denial;
9. missing predecessor witness denial;
10. skipped witness-history denial;
11. stale Xenia evidence cannot refresh RSK time;
12. matching Xenia witness cannot clear local negative state;
13. fresh epoch begins with a new zero-counter genesis and no authority carryover.

## Non-claims

This contract does not establish:

- that Xenia PR #335 is merged or qualified;
- durable external retention/CAS service availability;
- trusted witness failure-domain independence;
- trusted-time correctness;
- automatic crash reconciliation;
- production RSK admission.

It only freezes the cross-system semantics so later implementation cannot silently
change the meaning of either protocol.
