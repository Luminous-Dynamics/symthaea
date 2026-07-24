# Fabrication Containment and Requalification

Series 12 adds the negative and recovery authorities that remain necessary after a release has already passed promotion, rollout, resilience, and rollback ceremonies.

## Why this layer exists

A valid signature, a prior promotion, or a previously safe machine is not permanently trustworthy. Keys can be copied, witnesses can equivocate, gateways can be decommissioned, and hardware evidence can invalidate an active rollout. Emergency response must not erase historical evidence, but it must be able to remove authority immediately and durably.

The containment layer therefore treats these actions as explicit capabilities:

- quarantine a compromised signer;
- prove witness equivocation;
- permanently tombstone a decommissioned gateway;
- revoke rollout authority globally, by phase, or by machine;
- earn a new bounded machine capability after rollback;
- seal every transition into one hash-linked state and replay contract.

## Signer compromise

`SignerCompromiseNotice` binds one algorithm/key identity, its affected `KeyUsage` domains, the source trust snapshot, evidence digest, discovery time, effective time, and reason. A `signer-compromise-containment` threshold ceremony authorizes it.

`SignerCompromiseTracker` is append-only. Later records may broaden the affected usage set, but may not narrow it or move the effective time backward. `verify_threshold_ceremony_with_containment` applies this state after normal cryptographic and lifecycle verification, so a mathematically valid signature can no longer count once emergency containment is effective.

Containment is not a replacement for trust-snapshot rotation. It is the immediate bridge to a later revocation-bearing snapshot.

## Witness gossip and equivocation

`SignedWitnessGossip` lets a witness independently sign the checkpoint root and log size it observed. Two verified statements by the same witness identity and organization for the same log size but different roots produce a portable `WitnessEquivocationProof`.

`WitnessGossipTracker` preserves both underlying observations before it accepts the proof. This prevents an isolated proof object from entering durable state without the signed evidence that supports it.

## Gateway tombstones

The mutable decommission tracker ends at `Decommissioned`. `GatewayTombstone` then seals:

- the authorized decommission plan and ceremony;
- final retirement record;
- last gateway state;
- credential revocation;
- planned and independently verified erase evidence;
- successor membership;
- terminal timestamps.

`GatewayTombstoneRegistry` allows exactly one immutable tombstone per gateway and requires every successor state to retain it byte-for-byte.

## Hardware rollout revocation

`RolloutRevocationEvidence` can stop:

- an entire promotion;
- one rollout phase and every phase above it;
- a bounded set of machine identities.

The evidence binds the original rollout plan, incidents, optional signer-compromise state, effective time, and reason. A `hardware-rollout-revocation` threshold ceremony grants authority. `RolloutRevocationTracker` is append-only and is checked before machine authority is considered usable.

Revocation does not delete rollout observations or promotion history. It adds a new, stronger negative fact.

## Post-rollback requalification

Rollback restores an older release, not the historical world in which that release was originally safe. `PostRollbackRequalificationEvidence` requires:

- the rollback target to match a fresh `AssuredReleasePromotion`;
- release lineage to show that target as active;
- an intact incident ledger;
- every incident that triggered rollback to be resolved;
- a bounded, clean hardware observation set;
- minimum observation duration and successful-job count;
- an explicit machine allowlist;
- a short authorization lifetime.

Failures, uncertain outcomes, or emergency stops can be configured to block requalification completely. The resulting capability is machine-scoped and time-bounded.

## Durable state and replay

`FabricationContainmentState` hash-links generations and embeds:

- signer-compromise state;
- witness gossip and equivocation evidence;
- gateway tombstones;
- rollout revocations;
- post-rollback requalification capabilities;
- the exact prior release-resilience generation and digest.

Successors must preserve all historical prefixes and cannot substitute another resilience state at the same generation.

`ContainmentReplayContract` binds the state to the source tree, active promotion, trust snapshot, resilience state, and each embedded tracker digest. Replay succeeds only when all supplied evidence reconstructs the same contract.

## Authority rule

A production gateway should grant hardware authority only when all applicable positive capabilities are valid **and** no negative containment fact applies. In particular:

1. verify trust and signatures;
2. apply signer-compromise containment;
3. verify promotion, assurance, and machine/session authority;
4. deny tombstoned gateways;
5. deny revoked rollout scope;
6. after rollback, require an unexpired machine-specific requalification capability;
7. persist the resulting containment state before dispatch.

No Series 12 API claims that software evidence alone validates physical safety. Supervised machine testing and independent operational procedures remain required.
