# VART-REALITY-001 — Nested World Provenance Separation

Status: preregistered design; execution not authorized by this document.

## Question

Can Symthaea enter nested digital/counterfactual/dream/replay worlds, store and
recall events from them, and later preserve the correct world lineage without
silently upgrading hypothetical or replayed events into parent-world history?

## Primary integrity outcomes

The prospective study should create a small known world tree containing:

- one `DigitalCommitted` parent studio;
- two sibling `Counterfactual` branches;
- one nested counterfactual inside one branch;
- one `Dream` child;
- one `Replay` child.

Each world receives disjoint sentinel events with content digests that are easy
to distinguish but whose semantic surface can intentionally overlap.

Primary pass conditions must be frozen before execution and include:

1. every recalled sentinel preserves exact `world_id` and `lineage_id`;
2. every recalled sentinel preserves its original `RealityLayer`;
3. no counterfactual/dream sentinel is admitted with
   `may_claim_happened_in_current_world = true`;
4. no replay sentinel is admitted as a new present observation;
5. no derived physical record is admitted as direct sensor observation;
6. all ledger chains verify after serialization/reload;
7. removing, reordering or rewriting a record causes verification failure;
8. a selected counterfactual may be materialized only through an external
   authority-bound `CounterfactualCommitReceipt` with exact state-hash equality;
9. after materialization, the original hypothetical record remains attached to
   its counterfactual lineage.

## Controls

Include at minimum:

- flat committed-world control with no nested worlds;
- deliberately mislabeled source/layer records that must fail closed;
- wrong-parent nested world entry;
- wrong target in counterfactual commit receipt;
- state-hash mismatch during commit;
- dream-to-commit attempt;
- unknown provenance recall.

## Non-claims

A PASS would establish provenance separation in the tested software path. It
would not establish consciousness, phenomenology, metaphysical reality,
physical sensor correctness, memory accuracy beyond provenance, or general AI
safety.

## Future host extension

After the host-neutral study passes, repeat it with a real Symtropy/Bevy host:
real GPU parent observation, real rendered four-ghost counterfactuals, selected
materialization, and post-session recall. The Bevy study should retain exact
engine/GPU/environment fingerprints and world/state hashes.
