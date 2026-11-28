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
- The graph envelope carries the exact canonical semantic key and visual channel; a runtime verifier enforces that `attention_to` is derived only from an `attention.focus` cue and `handoff_to` only from a `conversation.turn.yield` cue.
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

## 9. Machine-readable v2 contract precursor (2026-11-27)

Two machine-readable artifacts have now been added:

- `SYMTHAEA_SCENE_GRAPH_SCHEMA_V2_2026-11-27.json`: Draft 2020-12 structural schema for frame identity, canonical cue digest, scene-graph digest, seven forms, privacy/destination, epistemic status, nodes, typed semantic relations, layout-only attachments, and effective expiry.
- `SYMTHAEA_SCENE_GRAPH_FIXTURES_V2_2026-11-27.json`: two positive fixtures and four negative graph cases (duplicate node IDs, dangling relation endpoint, layout-only relation forged as semantic, and local-sensitive remote destination).

The schema deliberately keeps semantic relations separate from renderer layout attachments. It carries the exact semantic key and declares the visual channel, and currently permits only two source-field-bound semantic relation kinds: `attention_to` and `handoff_to`. The runtime verifier must additionally enforce that `attention_to` corresponds to `attention.focus` and `handoff_to` corresponds to `conversation.turn.yield`; structural JSON Schema validation cannot perform the required source-cue/digest comparison. Other relation kinds require explicit canonical event fields and a reviewed schema extension.

Structural schema validation and causal endpoint verification are separate. The duplicate-ID and dangling-endpoint cases are expected to pass structural JSON shape validation but must be rejected by the runtime graph verifier. The schema and fixtures have been authored but have not been run through an independent Draft 2020-12 validator; no Rust graph compiler/verifier or Bevy edge renderer exists yet.

## 10. Rust compiler and verifier implementation (2026-11-28)

The dependency-light Soma module `crates/domains/symthaea-soma/src/embodiment_scene_graph.rs` now implements the scene graph v2 contract:

- `compile_scene_graph_v2` delegates cue/permit scope checks to the existing scene compiler, emits one stable core and at most one cue-specific node, creates a semantic edge only for an explicit attention target or explicit yield target, and emits a separate layout attachment.
- `verify_scene_graph_v2` checks the canonical cue digest, permit-derived metadata, scene graph digest, stable node set, glyph/form mapping, relation source field and direction, unique IDs, endpoint validity, layout attachment completeness, privacy-aware description, and bounded geometry weights.
- Raw target strings are not copied into nodes, semantic relations, or renderer-facing descriptions for local-sensitive cues. In fact, raw targets are not copied for public cues either; the canonical cue digest binds the source cue.
- Child motion is derived from the explicit interaction cue and reduced to static under static permission.

Eleven Rust unit tests cover all seven forms, explicit attention and handoff relations, no-invented-edge listening scenes, relation tampering, duplicate IDs, dangling endpoints, orphan nodes, missing layout attachments, target non-disclosure, and graph digest tampering.

The schema fixtures remain structural JSON Schema fixtures. Their digest strings are shape placeholders; runtime acceptance requires the real canonical cue and permit plus digest recomputation in the Rust verifier. This distinction is intentional and must remain visible until fixture generation is integrated with the compiler.

The tests are authored but have not been compiled or executed. No Bevy endpoint-connected graph rendering is claimed; the adapter still renders the earlier role-aware layout until the typed graph adapter is implemented.


## 9. Qualification status

This document specifies the next contract increment. The current renderer uses deterministic, role-aware placement but still has no explicit endpoint-connected semantic graph. No v2 schema, graph compiler, edge renderer, or human study has been implemented or qualified.
