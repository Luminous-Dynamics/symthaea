# Connectome correspondence evidence policy V0.1

This policy is a synthetic-fixture and verifier contract, not a biological validation standard.

## Relation-specific evidence

- `ARTIFACT_IDENTITY`: exact source artifact manifest bytes and a source-authoritative release identifier; matching names or object labels alone do not suffice.
- `SEGMENT_LINEAGE`: verified edit transaction or lineage receipt tying source and target segments. Geometry similarity may support review but cannot replace edit ancestry when the claim is lineage.
- `NEURON_CORRESPONDENCE`: independently adjudicated stable anchor / cell identity, or a separately validated multi-anchor identity policy. Segment lineage or abstract topology alone is insufficient.
- `SYNAPSE_CORRESPONDENCE`: synapse-local evidence and independently validated pre-/post-partner mapping. Presence alone is insufficient; partner and polarity must be validated separately.

## Authority is a claim until evidence is authenticated

The `authority` enum is descriptive input. A record stating `INDEPENDENTLY_VERIFIED` or `EXPERT_ADJUDICATED` is not trusted unless the referenced bytes, verifier, signer identity, adjudication protocol and provenance are independently checked. V0.1 does not yet implement that trust adapter.

## Alternative sets

Multiple mappings from the same source object with the same non-null `alternative_set_id` represent mutually exclusive candidate correspondences. They must not be interpreted as simultaneous one-to-many lineage. Every alternative must be retained; a preferred candidate cannot be chosen without a separately declared and validated resolution policy.

## Cardinality

Cardinality is a deterministic function of the source and target ID list lengths:
- 1:1 = ONE_TO_ONE;
- 1:N where N >= 2 = ONE_TO_MANY;
- N:1 where N >= 2 = MANY_TO_ONE;
- N:M where N >= 2 and M >= 2 = MANY_TO_MANY;
- N:0 where N >= 1 = UNMAPPED.

The verifier rejects a label inconsistent with those lengths. A relation may not use empty source lists.

## Promotion constraints

No mapping is trusted as `CLAIMED_VERIFIED` by this policy alone. The value is only an input claim until evidence bytes and authority have been authenticated. The V0.1 semantic verifier reports COUNTERS_RECOMPUTED and trusted_promotion=false because it does not authenticate evidence artifacts, lineage transactions, expert adjudication, registration anchors, signer identities or revocation state.

No biological identity certificate may be issued from V0.1 synthetic fixtures.
