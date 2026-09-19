# Authority State Lineage V3

## Purpose

Verified authority-state v0.2 proves fresh threshold agreement on an exact source-frontier sequence and digest, but its sequence is not explicitly scoped to a causal source lineage/incarnation.

V3 adds that missing identity without changing the meaning of historical V2 signatures.

## Core theorem

```text
sequence number
    != global chronology

exact AuthorityFrontierLineageV1
+ exact frontier sequence
+ exact frontier digest
+ fresh threshold witness agreement
    -> lineage-scoped current frontier evidence
```

The V3 statement and snapshot commitments use new domains and include:

- lineage namespace;
- non-zero incarnation commitment;
- source-frontier sequence;
- source-frontier digest;
- state sequence;
- authority epoch;
- current authority-context state;
- canonical relevant negative-authority facts.

`VerifiedAuthorityStateV3` remains non-Clone, non-Serde, and is not execution authority.

## Ordering

Within the same exact lineage:

```text
current sequence < reference sequence
    -> Rollback

current sequence == reference sequence
and current digest != reference digest
    -> Contradiction

same sequence + same digest
    -> Same

higher sequence
    -> Newer
```

Across different lineages:

```text
DifferentLineage
```

No numeric comparison is permitted. A sequence of 10,000 in a new lineage does not outrank sequence 50 in an old lineage merely because the integer is larger.

## Migration and recovery

This tranche deliberately does not mint a lineage-transition receipt.

A future migration/recovery theorem should bind something like:

```text
OldLineageHead
+ exact NewLineageGenesis
+ migration/recovery authorization
+ anti-rollback evidence
    -> LineageTransitionReceipt
```

Absent that proof, old live execution authority must not silently survive source replacement, disaster recovery, split-brain repair, or sequence reset.

## Protocol separation

V3 has fresh challenge, statement, and snapshot domains. V2 messages cannot be replayed as V3 evidence merely by adding local metadata.

The V2 witness policy and negative-authority fact commitments are reused because their semantics are unchanged; the signed V3 snapshot is what binds those facts into the new lineage-aware protocol.

## Qualification

The focused exact-head lane is draft-safe and uses a PR-stable concurrency group. It ratchets:

- explicit lineage identity;
- V2/V3 domain separation;
- witness agreement on exact lineage;
- rollback detection;
- equal-sequence contradiction detection;
- different-lineage incomparability;
- verifier-owned/non-Serde verified state.

Cargo.lock reconciliation is intentionally a hard gate. The lockfile is not hand-edited in this source tranche.

## Nonclaims

V3 does not provide consensus, choose a storage technology, authenticate a migration between lineages, mint a capability, reserve use budget, authorize an effect, prove executor identity, or provide continuous actuation authority.

It supplies the causal identity/ordering fact that those later layers can consume.
