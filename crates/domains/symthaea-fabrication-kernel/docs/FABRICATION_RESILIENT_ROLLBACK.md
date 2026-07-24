# Fabrication Resilient Rollback

Version 0.16.0 adds an explicit resilience layer above federated release
promotion. The layer is deliberately separate from the physical print authority
pipeline: it governs which gateway software and policy bundle may operate the
pipeline after regional failure, transparency equivocation, compromised
infrastructure, or a defective software promotion.

## Authority graph

```text
verified gateway consensus
        + active gateway membership
        -> regional quorum evidence

verified transparency checkpoint
        + independent witness signatures
        -> witnessed checkpoint quorum

release artifact set
        + independent builder statements
        -> verified reproducible provenance

threshold-authorized promotion
        + regional quorum
        + witnessed checkpoint
        + artifact provenance
        -> assured release promotion

current assured promotion
        + prior assured promotion
        + triggering incidents
        + compatibility evidence
        + threshold rollback ceremony
        -> authorized release rollback
```

Rollback never rewrites or deletes the promotion that failed. It creates a new,
short-lived authority object that points from one exact promotion digest to one
previously authorized target promotion digest. The append-only release lineage
records both the original promotion and the rollback.

## Cross-region quorum

`RegionalQuorumEvidence` maps only the gateways already accepted by
`VerifiedGatewayConsensus` into the active `GatewayMembership`. Policy can
require:

- a minimum number of represented failure domains;
- named required regions;
- a minimum fraction of total membership voting weight; and
- a maximum fraction of represented weight controlled by one region.

The evidence is canonical and rejects duplicate gateways, non-canonical region
or gateway order, inconsistent weight totals, unknown gateways, inactive
memberships, and single-region dominance.

`RegionalQuorumTracker` persists the latest accepted membership epoch and
gateway generation. It rejects epoch rollback, gateway-generation rollback, and
same-generation substitution.

## Transparency witnesses

A transparency checkpoint is signed by the log operator. A
`SignedTransparencyWitness` is a separate observation by an independent key,
organization, and region. The witness signs the exact checkpoint digest, tree
size, Merkle root, and observation time.

`TransparencyWitnessPolicy` can require independent signer, organization,
region, and algorithm counts. Witness verification is lifecycle-aware and uses
the `TransparencyWitness` key usage.

`TransparencyWitnessTracker` accepts only witnesses included in an already
verified witness quorum. It prevents a witness from:

- moving backward to a smaller log;
- changing the root at the same tree size;
- substituting a checkpoint digest at the same tree size; or
- moving its observation clock backward.

## Artifact provenance

`ArtifactProvenanceStatement` binds one exact `ReleaseArtifactSet` to:

- source-tree digest;
- builder identity and region;
- build-environment digest;
- dependency-lock digest;
- canonical named input digests;
- reproducible-match count; and
- build time.

The verified capability can require multiple distinct builders, multiple
regions, algorithm diversity, recent statements, and a minimum number of
matching reproductions. This does not assert that the build environment itself
is trustworthy; it makes the claimed environment and inputs explicit and
signature-bound for external verification.

## Gateway decommission

Gateway removal is split into two related authorities:

1. `AuthorizedGatewayMembership` proves that a successor membership removes the
   gateway while retaining the configured quorum properties.
2. `AuthorizedGatewayDecommission` binds that transition to the gateway's final
   state, credential-revocation evidence, secure-erase evidence, quarantine
   interval, decommission time, and reason.

The durable tracker advances only through:

```text
Quarantined -> EraseVerified -> Decommissioned
```

A gateway is excluded from authority immediately after quarantine. Existing
records cannot disappear, move backward, change plans, or lose retained erase
evidence in a successor state.

## Rollback authority

`ReleaseRollbackEvidence` requires:

- a current promotion and a strictly older target promotion;
- exact target artifact provenance;
- current regional quorum bound to the active promotion's gateway state;
- witnesses for the current promotion's transparency checkpoint;
- one or more canonical triggering incident digests;
- nonzero compatibility evidence when policy requires it;
- a bounded authorization window; and
- a threshold ceremony whose purpose is `release-rollback`.

Rollback is not an automatic restart or downgrade. External deployment systems
must still execute the authorized transition, preserve operational audit
records, and verify application-level data and protocol compatibility.

## Lineage, replay, and durable state

`ReleaseLineage` is an append-only hash chain over promotion and rollback
events. It preserves the prior active promotion, resulting active promotion,
authority digest, timestamp, and predecessor event digest.

`RollbackReplayContract` binds:

- current and target release assurance;
- rollback authority;
- release lineage and chain head;
- gateway decommission tracker; and
- transparency witness tracker.

`ReleaseResilienceState` hash-chains generations of all retained resilience
evidence. Successor validation rejects:

- generation rollback or skipped generations;
- predecessor-digest mismatch;
- release-lineage truncation or alteration;
- regional quorum rollback;
- witness evidence rollback;
- gateway retirement rollback; and
- removal of a retained rollback replay digest.

## Transparency publication

`publish_release_rollback` appends the rollback digest as a
`release-rollback` transparency entry. Verification requires the exact Merkle
inclusion proof, a matching verified checkpoint, and a witness quorum for that
checkpoint.

## Non-claims

Version 0.16.0 does **not** claim:

- Byzantine consensus across live networks;
- an aggregated threshold-signature primitive;
- globally independent witness organizations;
- reproducible builds without external build execution;
- cryptographic secure erase merely because an erase digest exists;
- safe database or protocol downgrade for every application; or
- automatic physical-machine rollback while jobs are executing.

Those properties require deployment-specific cryptographic providers,
independent infrastructure, reproducible builders, storage and hardware
attestation, compatibility testing, and supervised operational drills.
