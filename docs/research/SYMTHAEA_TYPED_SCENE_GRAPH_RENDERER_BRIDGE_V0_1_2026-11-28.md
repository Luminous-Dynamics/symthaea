# Symthaea typed scene-graph renderer bridge v0.1 — 2026-11-28

## Problem

SceneGraphV2 now distinguishes semantic relationships from layout-only attachments. The first Bevy adapter still renders SceneFrame v1 flat elements, so it cannot yet draw a relationship as an explicit, source-provenanced connector.

The new `renderers/symthaea-soma-bevy/src/scene_graph_layout.rs` converts SceneGraphV2 into deterministic Bevy-native node placements and quadratic semantic paths.

## Implemented

- Form-specific position grammars for all seven embodiments.
- Stable node-ID ordering, parent resolution and fail-closed missing-parent checks.
- Only explicit `SceneRelationV2` records produce semantic connector paths.
- `SceneLayoutAttachmentV2` / `LayoutAnchor` are used for placement only and do not become visible relationship edges.
- Returned geometry preserves node ID, role, glyph, motion, weight and relation ID/kind.

The module plans geometry; it is not yet wired into the existing AuthorizedSceneSubmission/ECS spawn lifecycle. It does not bypass the existing permit, clock, revocation, expiry or output-receipt checks.

## Next

1. Add an authorized graph submission type that verifies graph digest, canonical cue, permit and Authorized receipt in one constructor.
2. Spawn graph nodes and semantic edge meshes under one scene group.
3. Apply source/target permit checks on every update and clear the whole graph on target denial.
4. Record Submitted only after ECS acceptance; require the independent compositor observer for Observed/Stopped.
5. Add screenshot-based geometric and human-legibility tests for each of the seven forms, plus explicit edge-direction and node-identity tests.

## Qualification

Source-level geometry mapping is implemented and authored tests are included. It has not been compiled, rendered, screenshot-tested or visually assessed. The existing Bevy adapter path is unchanged and remains the only fully wired source path.
