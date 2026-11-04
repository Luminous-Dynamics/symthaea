# PITB DAPS graph import adapter

A standalone Rust adapter that imports DAPS Mathlib graph exports into PITB's typed evidence graph **without promoting DAPS edges to formal Lean dependencies**.

## Why candidate-only?

The pinned DAPS extractor walks constants used by each Lean declaration, including constants in both the declaration type and value. The graph builder then labels edges from the target node's subcategory (for example, `USES_CLASS`, `USES_DEFINITION`, `USES_INSTANCE` and `DEPENDS_ON`). These export labels do not distinguish type references from proof-term references, transparent body references, instance-synthesis events or other semantic roles. The sample graph is useful retrieval evidence, but it is not a typed elaborator receipt for each edge.

Therefore every imported DAPS edge has:
- `status = candidate`;
- `evidence_class = daps_graph_export`;
- `confidence = null` (DAPS does not export calibrated edge confidence);
- derived role `candidate_retrieval_hint`;
- no scheduler admission.

The adapter never maps `USES_CLASS` to `type_reference`, `USES_DEFINITION` to `definition_body_reference`, or `USES_INSTANCE` to `instance_synthesis`.

Three candidate relations preserve endpoint roles:
- `daps_dependency_candidate`: declaration to declaration;
- `daps_import_candidate`: module to module;
- `daps_membership_candidate`: module/file membership to declaration (`CONTAINS`).

Unknown nodes, unsupported edge kinds, duplicate declarations, unsupported endpoint combinations and malformed provenance fail closed. Original upstream node fields and edge JSON are retained in metadata/content-addressed evidence files.

## Pinned source fixture

The checked-in sample is DAPS's public 30-node/165-edge graph demo, not the full Mathlib graph.

| Source | Pinned identity |
|---|---|
| DAPS repository commit | `978ca6bebb4c56e03795b4f5e34fcbdc6cb2d153` |
| DAPS graph demo Git blob | `ff3e07e7a4867c969f4ed739497f78b6a77209f` |
| DAPS Lean extractor Git blob | `66eb2045724da71e32b765223c6c4fdcc903f466` |
| DAPS Lake manifest Git blob | `778cba54c3e6117ad357e192d421d196b7d1eb6c` |
| Lean toolchain | `leanprover/lean4:v4.22.0` |
| Mathlib Git commit (from pinned Lake manifest) | `79e94a093aff4a60fb1b1f92d9681e407124c2ca` |

The exact DAPS Apache 2.0 license text is retained under `third_party/`. The importer records raw input SHA-256 alongside Git object identifiers; those are different digest domains and are not conflated.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-daps-graph-import/Cargo.toml
cargo test --locked --manifest-path tools/pitb-daps-graph-import/Cargo.toml
cargo run --locked --manifest-path tools/pitb-daps-graph-import/Cargo.toml -- --self-test

cargo run --locked --manifest-path tools/pitb-daps-graph-import/Cargo.toml -- \
  --convert \
  tools/pitb-daps-graph-import/fixtures/daps-example-graph.json \
  /tmp/pitb-daps-import \
  978ca6bebb4c56e03795b4f5e34fcbdc6cb2d153 \
  ff3e07e7a4867c969f4ed739497f78b6a77209f \
  66eb2045724da71e32b765223c6c4fdcc903f466 \
  778cba54c3e6117ad357e192d421d196b7d1eb6c \
  79e94a093aff4a60fb1b1f92d9681e407124c2ca \
  leanprover/lean4:v4.22.0

cargo run --locked --manifest-path tools/pitb-graph-evidence/Cargo.toml -- \
  --verify /tmp/pitb-daps-import/graph.json /tmp/pitb-daps-import/evidence
```

The final verifier command checks the generated manifest and all statement/edge evidence bytes under the output root. It does not turn candidate edges into scheduler inputs.

## Output and trust boundary

The output directory contains:
- `graph.json`: PITB typed graph manifest with pinned snapshot identity and upstream provenance;
- `summary.json`: counts and an explicit `scheduler_admission: false`;
- `evidence/statements/`: exact statement bytes used for node digests;
- `evidence/edges/`: each original DAPS edge and its array index, preserved as a content-addressed evidence file.

The import preserves upstream source path, module, line/column, category/subcategory, signature, docstring, proof text, blacklist marker and Lean snippet as upstream metadata. These are not verified source files or elaborator receipts. The snapshot is externally sourced candidate evidence, not a measured PITB Lean extraction.

## Next work

1. Add a thin DAPS selector-output importer that retains checkpoint, query formalization, split, candidate pool and top-K provenance.
2. Compare DAPS and LeanGraphSearch candidates under identical target/environment and downstream prover budgets.
3. Independently rerun a pinned Lean extractor to reconstruct typed signature, proof, definition, field, inheritance and instance edges.
4. Join an external edge to a formal PITB edge only after exact node identity and independently reproduced semantic role; otherwise keep it candidate-only.
5. Measure useful-novel-premise rate and verified exact-target completion, not only Recall@K.
