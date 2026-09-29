# Scientific lineage fixture v1

This fixture defines the first cross-domain lineage contract for the shared
scientific substrate. It is deliberately synthetic: it demonstrates typed
identity and provenance boundaries, not physical performance.

## Canonical lineage

```
theorem --constrains--> physical_model
simulation --instantiates--> physical_model
simulation --produces--> prediction --supports--> scientific_claim
process --produces--> material --has_property--> predicted_property
experiment --observes--> observation --quantifies--> uncertainty
```

## Evidence boundaries

- A theorem constrains a model; it does not establish that the physical model matches reality.
- A simulation produces a prediction; a prediction is not a measurement.
- A process produces a material candidate; the candidate is not automatically qualified.
- An experiment produces an observation; uncertainty remains an explicit object.
- `supports` is epistemic and must never become a deterministic qualification invalidation edge.
- Qualification is a separate projection over explicitly admitted evidence.

## Required invariants

1. Every object has a stable `EngineeringObjectId`.
2. Every relation has a stable relation kind and endpoint semantics.
3. Prediction and observation remain distinct object kinds.
4. Predicted and measured properties remain distinct.
5. Uncertainty remains explicit rather than being folded into a scalar score.
6. Epistemic relations remain outside deterministic engineering closure.
7. Removing a model, parameter set, process route, toolchain, or calibration input invalidates only projections that actually depend on it.
8. Historical lineage remains addressable after currentness changes.

## Next implementation step

The executable fixture now lives in `symthaea-engineering::scientific_lineage`.
It validates the relation endpoint oracle and provides a bounded qualification
projection whose invalidation closure excludes epistemic relations. The graph
also deliberately permits epistemic cycles, demonstrating that recursive DKG
knowledge and deterministic qualification topology are separate concerns.


## Replay artifact contract

`QualificationProjection` is also the replay artifact for the bounded
qualification plane. Its serialized representation records:

- `schema`: `symthaea.qualification-projection.v1`
- `source_graph_digest`: SHA-256 identity of the complete scientific graph snapshot,
  including isolated nodes and relations
- `qualification_policy`: `symthaea.qualification-policy.v1`
- ordered, unique typed relations admitted to qualification
- `authority_ceiling`: `synthetic-qualification`

The projection digest is computed only from the projection schema, policy,
authority ceiling, and ordered relation identities. The source graph digest is
recorded separately. Therefore an epistemic-only DKG mutation can change the
knowledge snapshot without changing the deterministic qualification projection,
while replay can still require the exact source snapshot.

Deserialization is not trusted blindly: the projection uses a validated wire
representation and rejects unknown schema/policy, malformed identities,
non-qualification relations, duplicate or unordered relations, and cyclic
projection topology.

This establishes the replay boundary:

```text
scientific DKG snapshot
        │
        ├── source_graph_digest
        ▼
validated QualificationProjection
        │
        ├── projection_digest
        ├── qualification_policy
        └── authority_ceiling
        ▼
CP-04 qualification / replay
```

A projection digest is not physical-performance evidence and never grants
operational authority.

## CP-04 adapter boundary

The `Cp04QualificationArtifact` is the explicit contract-neutral adapter from
a validated `QualificationProjection` into the CP-04 typed evidence vocabulary.

The adapter is intentionally conservative:

- it validates the complete source projection before conversion;
- it accepts only CP-04 node kinds and edge types with the CP-04 endpoint ontology;
- it carries each original `EngineeringObjectId` and relation digest forward;
- it preserves the source graph digest, projection digest, qualification policy,
  and `synthetic-qualification` authority ceiling;
- it rejects unsupported scientific relations rather than reinterpreting them
  as compute-evidence dependencies;
- it cannot introduce a node or edge that was absent from the source projection.

This makes the adapter a semantic firewall: the scientific DKG may contain richer
mathematical, physical, materials, or epistemic structure, while CP-04 receives
only the explicitly admitted typed dependency projection.

A deterministic CP-04 hand-off fixture is checked in at `docs/engineering/data/cp-04-scientific-lineage-adapter-v1.json`. Its source, projection, and artifact digests are fixed so the future CP-04 qualifier integration can consume a concrete cross-contract replay vector rather than reconstructing semantics independently.
