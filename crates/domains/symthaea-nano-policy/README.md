# Symthaea Nano Policy (research-only prototype)

A small Rust CLI that selects a subset of a supplied candidate pool using either:
- maximin: greedy farthest-point selection over min-max-normalized numeric descriptors;
- random: a seeded, deterministic random-control policy.

It is intentionally **not** an atomistic model, optical solver, candidate generator, toxicity model, or materials predictor. It has no target-label, energy, spectrum, physics-engine, network, file-system, or model-checkpoint interface. Strict Serde structs reject unknown fields, so an attempted extra target/label field in a candidate record fails closed at JSON parsing.

## Status and interface

- Interface: one JSON request from stdin; one JSON response to stdout; errors to stderr and exit code 2.
- Request schema version: nano-selection-input-v1.
- Machine-readable input contract: schema/input.schema.json (JSON Schema Draft 2020-12).
- Candidate input fields: id, features only.
- Input budgets/candidate/feature cells are bounded.
- Feature normalization uses min/max across the permitted unlabeled candidate pool; this is a declared transductive descriptor operation, not a use of target labels.
- Ties are deterministic by lexicographically sorted candidate ID.
- Random control uses a pinned local SplitMix64 implementation, with no OS entropy.
- Maximin scores are squared Euclidean distances in normalized feature space. The first selection, if there is no prior selection, uses distance to the candidate-pool centroid; later selections use minimum distance to any selected candidate.
- already_selected IDs must be members of the input pool and are excluded from new selections.

## Example

~~~json
{
  "schema_version": "nano-selection-input-v1",
  "policy": "maximin",
  "budget": 2,
  "seed": 7,
  "already_selected": [],
  "candidates": [
    {"id": "geometry-a", "features": [0.0, 0.2]},
    {"id": "geometry-b", "features": [0.1, 0.25]},
    {"id": "geometry-c", "features": [0.95, 0.8]},
    {"id": "geometry-d", "features": [1.0, 1.0]}
  ]
}
~~~

Output contains the policy/version, pool dimensions, requested budget, prior selections and ordered selected candidate IDs with policy-specific scores. Scores are selection diagnostics—not probabilities, uncertainty, physical properties, novelty, stability, safety or optical performance.

## Build and test

From the Symthaea workspace root:

~~~bash
cargo test -p symthaea-nano-policy --locked
cargo run -p symthaea-nano-policy --release --locked < request.json
~~~

This crate is intentionally small and uses the workspace's pinned serde and serde_json dependencies. A dedicated targeted workflow checks format, tests, and Clippy. If the Rust toolchain is unavailable locally, don't claim local compilation; rely on exact-head workflow evidence only when its job actually completes successfully.

## Required next steps before scientific use

1. Obtain descriptor rows from a rights-cleared and pinned nanophotonic dataset, and establish exact identity and group/split lineage.
2. Use only predeclared geometry-derived descriptors or allowed frozen embeddings; no target spectra or outcomes may appear in the candidate input.
3. Add independent baselines and an output receipt digest/canonicalization contract.
4. Compare random, maximin, task-native conventional search (e.g. MOCLIP/adjoint where appropriate), and one Symthaea policy under equal candidate/solver budgets.
5. Use a held-out target set, group-aware uncertainty, negative controls, and independent spectral/forward-solver evaluation.
6. Demonstrate useful value before adding this to any production interface.

No scientific benchmark has been run by the existence of this prototype.
