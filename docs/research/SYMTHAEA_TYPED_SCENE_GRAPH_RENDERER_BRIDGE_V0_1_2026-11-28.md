# Symthaea typed scene-graph renderer bridge v0.1 — 2026-11-28

## Problem

SceneGraphV2 now distinguishes semantic relationships from layout-only attachments. The first Bevy adapter still renders SceneFrame v1 flat elements, so it cannot yet draw a relationship as an explicit, source-provenanced connector.

The `renderers/symthaea-soma-bevy/src/scene_graph_layout.rs` converts SceneGraphV2 into deterministic Bevy-native node placements and quadratic semantic paths. The new `scene_graph_adapter.rs` now provides an authorized queue and ECS submission path.

## Implemented

- Form-specific position grammars for all seven embodiments.
- Stable node-ID ordering, parent resolution and fail-closed missing-parent checks.
- Only explicit `SceneRelationV2` records produce semantic connector paths.
- `SceneLayoutAttachmentV2` / `LayoutAnchor` are used for placement only and do not become visible relationship edges.
- Returned geometry preserves node ID, role, glyph, motion, weight and relation ID/kind.
- `AuthorizedSceneGraphSubmission::new` verifies the graph against its original canonical cue, opaque permit and Authorized receipt before queueing.
- The graph ECS path rechecks live policy and source clock before entity creation and every update; revoked, expired or stale output is despawned.
- Graph node/edge entities carry graph-digest and cue-digest provenance metadata; semantic edges include directional arrowheads.
- Graph output is intensity-capped, and the graph and legacy flat-scene queues hard-cut one another to avoid overlapping representations.
- Adapter acknowledgements describe ECS acceptance only, not compositor presentation or user observation.

The module plans geometry; it is not yet wired into the existing AuthorizedSceneSubmission/ECS spawn lifecycle. It does not bypass the existing permit, clock, revocation, expiry or output-receipt checks.

## Next

1. Add tests for the authorized graph constructor's receipt/permit/clock mismatch cases.
2. Record Submitted only after the adapter acknowledgement; require the independent compositor observer for Observed/Stopped.
3. Add screenshot-based geometric and human-legibility tests for each of the seven forms, plus explicit edge-direction and node-identity tests.
4. Exercise policy revocation, source expiry, clock staleness and queue supersession in an actual Bevy app.

## Qualification

Source-level geometry mapping and an authorized ECS submission path are implemented, with six authored layout/adapter tests. They have not been compiled, rendered, screenshot-tested or visually assessed. The existing v1 flat-scene path remains available; the new graph path is wired in the plugin but has no runtime qualification yet.
