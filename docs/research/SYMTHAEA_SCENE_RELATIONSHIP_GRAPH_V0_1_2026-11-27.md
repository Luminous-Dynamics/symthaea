# Symthaea Scene Relationship Graph v0.1

**Date:** 2026-11-27  
**Status:** contract proposal; no schema v2 or endpoint renderer implementation claimed  
**Parent:** polyembodiment scene compiler and visual-language qualification

## 1. Problem statement

The current `SceneElement` contract carries `role`, `glyph`, optional `weight`, and `motion`. It does not carry a node identity, an edge, or source/target endpoints.

The new deterministic visual-layout module is therefore deliberately role-aware but does not draw purported semantic connections between objects. Inferring edges from array order, proximity, or renderer assumptions would manufacture relationships that the canonical cue never supplied.

The next architectural increment is a versioned scene relationship graph. It must distinguish a visual layout attachment from a semantic relationship.

## 2. Two different kinds of relationship

### 2.1 Layout attachment

A layout attachment is a renderer instruction that groups or positions a visual marker relative to a stable anchor. It does not assert that two underlying concepts are semantically related.

Examples:
- place a turn-state marker near the embodiment core;
- position an attention indicator in a form-specific focus region;
- keep a control affordance in a stable, discoverable location.

Layout attachments may be generated deterministically from an approved form grammar.

### 2.2 Semantic relationship

A semantic relationship asserts something about the event represented by the scene. It must be derived from an explicit typed source field and remain bound to the canonical cue digest.

Examples may include:
- an explicit attention target;
- an explicitly supplied alternative set;
- an explicitly recorded evidence-to-claim relation;
- an explicit conversation handoff target.

The renderer may change the visual syntax of a relation, but may not invent, remove, reverse, or strengthen its meaning.

## 3. Proposed v2 data model

The proposal adds frame-local node identities and typed relations without changing the canonical cue itself.

Conceptual structures:

```rust
struct SceneNodeId(u16);

struct SceneNode {
    id: SceneNodeId,
    role: SceneRole,
    glyph: SceneGlyph,
    weight: Option<f32>,
    motion: SceneMotion,
    layout_anchor: LayoutAnchor,
}

enum LayoutAnchor {
    Core,
    Periphery { slot: u16 },
    AttachedTo { node: SceneNodeId, slot: u16 },
}

struct SceneRelation {
    source: SceneNodeId,
    target: SceneNodeId,
    kind: SceneRelationKind,
    origin: RelationOrigin,
}

enum RelationOrigin {
    ExplicitSourceField { field: &'static str },
    RendererLayoutOnly,
}
```

The schema should use stable strings and explicit enums rather than serializing Rust implementation details. The sketch is illustrative, not a committed schema.

The `RendererLayoutOnly` origin is not a semantic edge and must not be presented to a user as evidence of a causal or factual relationship. Prefer keeping layout attachments and semantic relations in separate arrays so they cannot be confused.

## 4. Contract invariants

- Every node ID is unique within one frame.
- Every relation endpoint refers to an existing node in the same frame.
- No relation is created from array order or proximity.
- Relation kind is an explicit allowlisted enum.
- Renderer layout may transform coordinates and visual style, but not the relation kind or direction.
- Node/edge ordering is deterministic for byte-identical scene inputs.
- All coordinates and geometry parameters remain finite and bounded.
- No raw evidence-reference ID or restricted semantic target is copied into a renderer-facing graph.
- The canonical cue digest remains unchanged across forms; a separate scene-graph digest binds renderer-facing layout data.
- A privacy downgrade removes affected nodes and relations before rendering.
- Effective permit expiry and live policy rechecks continue to govern the whole frame.

## 5. Cross-form mapping

Each form renders the same node/edge graph through a different grammar:

| Form | Node grammar | Relation grammar |
|---|---|---|
| Living Constellation | anchors and stars | filaments and orbit arcs |
| Puppet Theatre | joints and pose anchors | threads and articulated segments |
| Mycelial Being | junctions and hypha nodes | branching paths |
| Swarm | particles and clusters | directed trails |
| Living Instrument | resonators and markers | strings and harmonic links |
| Hearth Companion | ember anchors | hearth arcs |
| Cognitive Mandala | central and radial nodes | spokes and concentric links |

A relation is preserved as the same typed relation even when its visual syntax differs.

## 6. Benchmark and tests

Structural gates:
- 100% endpoint-reference validity;
- zero inferred semantic edges;
- zero relation direction reversals;
- 100% stable node IDs and scene-graph digest on replay;
- 100% typed relation preservation across seven forms;
- zero privacy-class upgrades;
- zero frame retention after effective permit expiry.

Human evaluation:
- >=95% correct interpretation of relation direction and type on held-out scenes;
- >=95% same-event relation interpretation agreement across forms;
- report per-relation and per-form results, not only an average.

Negative controls:
- reorder the node array without changing IDs and verify relation meaning remains unchanged;
- mutate a relation kind while keeping the cue digest and verify rejection;
- remove a target node and verify all dangling edges are rejected;
- change privacy classification and verify restricted relations are removed;
- present a scene with no source relation and verify no semantic edge appears.

## 7. Migration

- Keep scene schema v1 immutable and supported for replay.
- Introduce a separately versioned scene schema v2 with explicit nodes and relationships.
- Do not reinterpret v1 element ordering as v2 edge semantics.
- Add positive and negative v2 fixtures before renderer migration.
- Migrate the Bevy adapter only after the v2 core compiler and verifier pass.
- Continue to compute the canonical-cue digest independently of renderer geometry.

## 8. Immediate implementation sequence

1. Finalize the v2 schema and enum vocabulary.
2. Add node/edge compiler output to Soma with deterministic fixtures.
3. Add a structural/causal graph verifier and negative controls.
4. Extend the interaction trace with the scene-graph digest.
5. Add Bevy endpoint-connected geometry for each form.
6. Validate graph meaning across forms with held-out human tasks.

## 9. Qualification status

This document specifies the next contract increment. The current renderer uses deterministic, role-aware placement but still has no explicit endpoint-connected semantic graph. No v2 schema, graph compiler, edge renderer, or human study has been implemented or qualified.
