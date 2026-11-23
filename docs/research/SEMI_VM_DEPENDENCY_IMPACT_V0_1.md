# SEMI-VM-IMPACT-001 — transitive VM snapshot dependency impact closure v0.1

**Issue:** [#13463](https://github.com/Luminous-Dynamics/symthaea/issues/13463)  
**Implementation:** `symthaea-semiconductor::benchmark::vm_snapshot_dependency_impact`  
**Status:** Rust graph/closure primitive with authored tests; exact-head validation pending.

## Research thesis

A physical label can be consumed by one training snapshot, then influence a family of models through fine-tuning, warm-starting, ensembles, distillation or calibration heads. Requalifying only the direct training snapshot is insufficient when the programme cannot identify the model descendants that inherited the affected state.

The unit of requalification becomes an explicit model-derivation edge, not a guessed relationship inferred from names, files or commit chronology.

## Typed dependency model

Nodes bind a snapshot ID, artifact digest, consumed-label manifest digest, training/evaluation purpose and derivation kind.

Edges bind:
- parent and child snapshot IDs;
- exact parent artifact digest;
- explicit inherited dependency dimensions;
- a derivation-evidence digest;
- a deterministic edge digest.

Dependency dimensions include model weights, feature schema, calibration state and evaluation target.

Derivation semantics:
- fine-tune, warm-start, ensemble and distillation edges must declare model-weight inheritance;
- calibration-head edges must declare feature-schema inheritance; they may additionally inherit model weights or calibration state;
- evaluation-only edges must declare the evaluated-model dependency and cannot declare model-state inheritance;
- independent-retrain edges cannot declare model-weight inheritance and may retain feature-schema references without inheriting the parent's trained weights.

A model-state impact (blocked or unknown K1 evidence) propagates only over declared model-weight or calibration-state dependencies. Feature-schema-only edges do not automatically inherit a parent label-revocation risk. Evaluation-target edges turn a blocked/historical parent into a historical-only evaluation result, not a model rebuild command.

## Validation and closure

The implementation validates:
- graph and edge schemas, digests and unique identities;
- endpoint existence and parent artifact digest match;
- explicit dependency semantics for each derivation kind;
- duplicate parent-child edges and duplicate dependency dimensions;
- cycle-free graph topology;
- one K1 requalification receipt per snapshot;
- each receipt's digest, historical manifest reference and read-only authority flags.

It uses deterministic Kahn topological sorting with snapshot-ID tie-breaks. Rebuild candidates are emitted in parent-before-child order.

If the owner graph inventory is incomplete, dependency ordering is incomplete or a snapshot requalification report is missing, the output is UNKNOWN and lists unresolved snapshots. It never reports an empty impact set as proof that no dependency exists.

## Authored tests

Fourteen unit tests cover:
1. clean root snapshot;
2. blocked base propagating through a fine-tune/warm-start chain;
3. independent retrain with feature-schema-only reference remaining unimpacted by parent label failure;
4. evaluation-only child becoming historical-only rather than a rebuild candidate;
5. unknown parent propagating indeterminate state;
6. incomplete graph inventory returning UNKNOWN;
7. non-total ordering returning UNKNOWN;
8. cycle rejection;
9. parent-artifact digest mismatch rejection;
10. independent-retrain model-weight inheritance rejection;
11. node/edge input-order-invariant graph digest;
12. required requalification report missing;
13. tampered requalification receipt rejection;
14. dependency-dimension-order-invariant edge and graph digest.

Tests are authored; exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference workflow are required before claiming a pass.

## Claim ceiling

This is a synthetic dependency-impact index. It does not prove that an actual model build consumed the declared parent, authenticate build-system receipts, retrain or roll back models, establish production model health, or authorize wafer disposition or process control. A dependency edge is only as trustworthy as the independent build provenance that records it.
