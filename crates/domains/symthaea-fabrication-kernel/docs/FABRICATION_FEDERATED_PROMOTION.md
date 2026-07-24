# Fabrication Federated Promotion

Series 10 adds the authority boundary between a certified release candidate and
permission to promote or roll it out.

## Authority chain

```text
certified candidate
  + canonical artifact inventory
  + exact gateway consensus
  + active gateway membership epoch
  + majority-weight fenced partition lease
  + transparency-log inclusion proof
  + fresh signed transparency checkpoint
  + threshold promotion ceremony
        |
        v
  AuthorizedReleasePromotion
        |
        +-- canary observation
        +-- threshold rollout advance
        +-- limited observation
        +-- threshold rollout advance
        v
  general release
```

A release signature is not sufficient by itself. Promotion requires all of the
following evidence to agree on the same source tree, candidate, gateway state,
consensus result, membership epoch, lease, transparency root, and artifact set.

## Threshold ceremonies

`threshold.rs` deliberately implements an explicit multi-signature quorum, not
an aggregated threshold-signature scheme. Every detached signature remains
visible. Policy can require:

- a minimum number of distinct signers;
- algorithm diversity;
- particular algorithms;
- a key allowlist;
- one lifecycle-governed `KeyUsage`.

Purpose and payload substitution are rejected. Approval windows and trust
snapshot freshness are evaluated before authority is granted.

## Gateway membership

Membership is canonical and epoch-numbered. Each member has voting weight and a
failure domain. A rotation must:

- advance by exactly one epoch;
- preserve minimum membership and voting weight;
- preserve failure-domain diversity;
- retain sufficient voting weight from the previous roster;
- stay below the configured maximum removed-weight fraction;
- carry a matching threshold ceremony over the exact transition digest.

## Partition-safe leases

A partition lease binds one holder to one membership epoch, consensus digest,
gateway state, generation, nonce, sequence, and monotonically increasing
fencing token. The endorsing consensus must represent strictly more than half
of membership voting weight and the configured number of failure domains.

`LeaseAuthorityTracker` rejects:

- membership rollback;
- lease-sequence rollback;
- fencing-token reuse or rollback;
- same-sequence substitution;
- overlapping conflicting leases;
- expired leases.

## Transparency

The transparency log is an append-only Merkle tree. It supports inclusion proofs
for each release or authority entry and prefix verification between snapshots.
Signed checkpoints bind the exact tree size and root to a lifecycle-governed
`TransparencyLog` key. Checkpoint tracking prevents truncation, same-size root
substitution, and unlinked growth.

## Reproducible artifacts

`ReleaseArtifactSet` records each release path, media type, byte length, and
SHA-256 digest in canonical path order. Unsafe paths, duplicate entries,
unlisted outputs, missing files, length drift, and digest drift are rejected.

## Promotion and rollout

`AuthorizedReleasePromotion` binds the exact candidate, artifact set, gateway
replay contract, membership, partition lease, transparency proof, and
checkpoint. It requires a separate threshold ceremony with purpose
`release-promotion`.

Rollout is staged through `Canary`, `Limited`, and `General`. An advance requires
minimum observation time and attempt count, a bounded failure rate, zero
emergency stops, and another threshold ceremony. The tracker rejects phase
skips, promotion substitution, sequence rollback, and same-sequence drift.

## Replay

`PromotionReplayContract` binds the promotion and rollout evidence so a later
reconstruction detects drift in any of the following:

- source tree or artifact inventory;
- release candidate;
- gateway replay evidence;
- membership epoch;
- partition lease and fencing token;
- transparency checkpoint and root;
- threshold ceremony;
- rollout authorization.

## Non-claims

Series 10 does not claim:

- implementation of a cryptographic threshold-signature primitive;
- Byzantine consensus transport or leader election;
- safe physical rollout without supervised integration testing;
- independence from the external signature providers;
- protection against a majority of authorized members colluding.

Those remain system-level responsibilities outside this crate.
