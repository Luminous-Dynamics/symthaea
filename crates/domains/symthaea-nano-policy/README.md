# Symthaea Nano Policy (research-only prototype)

A small Rust CLI that selects a subset of a supplied candidate pool using either:
- maximin: greedy farthest-point selection over min-max-normalized numeric descriptors, maintaining nearest-selected distances incrementally;
- random: a seeded, deterministic random-control policy.

It is intentionally **not** an atomistic model, optical solver, candidate generator, toxicity model, or materials predictor. It has no target-label, energy, spectrum, physics-engine, network, file-system, or model-checkpoint interface. Strict Serde structs reject unknown fields, so an attempted extra target/label field in a candidate record fails closed at JSON parsing.

## Status and interface

- Interface: one JSON request from stdin; one JSON response to stdout; errors to stderr and exit code 2.
- Request schema version: nano-selection-input-v1.
- Machine-readable input/output contracts: schema/input.schema.json and schema/output.schema.json (JSON Schema Draft 2020-12).
- A deterministic plumbing fixture is provided in fixtures/smoke-request.json with expected selected IDs in fixtures/smoke-expected.json.
- Candidate input fields: id, features only.
- Input budgets/candidate/feature cells are bounded.
- Maximin normalizes features with min/max across the permitted unlabeled candidate pool; this is a declared transductive descriptor operation, not a use of target labels. The random-control arm deliberately skips normalization to avoid irrelevant overhead.
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

Output contains the policy/version, pool dimensions, requested budget, prior selections and ordered selected candidate IDs with policy-specific scores. The CLI also emits input_blake3, the BLAKE3 digest of the exact raw request bytes (including whitespace); orchestration should hash exact response bytes separately for a complete run receipt. Scores are selection diagnostics—not probabilities, uncertainty, physical properties, novelty, stability, safety or optical performance.

## Build and test

From the Symthaea workspace root:

~~~bash
cargo test -p symthaea-nano-policy --locked
cargo run -p symthaea-nano-policy --release --locked < request.json
~~~

This crate is intentionally small and uses the workspace's pinned serde and serde_json dependencies. A dedicated targeted workflow checks format, tests, and Clippy. If the Rust toolchain is unavailable locally, don't claim local compilation; rely on exact-head workflow evidence only when its job actually completes successfully.

## Prototype acceptance and performance profile

Software acceptance before scientific use:
- all 12 unit tests pass on the exact commit;
- formatting and Clippy pass with warnings denied;
- same canonical request produces the same ordered IDs for both policies, independent of input row order;
- unknown target/outcome fields fail parsing;
- all candidate IDs and budget slots are accounted for, and no candidate is selected twice;
- the locked dependency graph is unchanged except for this package.

The maximin loop has target complexity O(N · D · (P + K)) where N is pool size, D is descriptor count, P is the number of already-selected IDs used for initialization, and K is the new selection budget; this is a complexity analysis, not a measured runtime claim.

Profile in release mode using synthetic descriptor pools first (synthetic data is a software test, not a scientific result): pool sizes 1,000 / 10,000 / 100,000; dimensions 8 / 64 / 256; budgets 1 / 10 / 100, then higher only within memory/time limits. Record p50/p95 wall time, peak resident memory, input size, selected count and deterministic output digest on a named CPU. Benchmark random and maximin separately. Do not set an absolute runtime pass target until a reference machine and baseline are measured; once measured, preregister the selector-overhead budget relative to the actual forward oracle.

## Required next steps before scientific use

1. Obtain descriptor rows from a rights-cleared and pinned nanophotonic dataset, and establish exact identity and group/split lineage.
2. Use only predeclared geometry-derived descriptors or allowed frozen embeddings; no target spectra or outcomes may appear in the candidate input.
3. Add independent baselines and an output receipt digest/canonicalization contract.
4. Compare random, maximin, task-native conventional search (e.g. MOCLIP/adjoint where appropriate), and one Symthaea policy under equal candidate/solver budgets.
5. Use a held-out target set, group-aware uncertainty, negative controls, and independent spectral/forward-solver evaluation.
6. Demonstrate useful value before adding this to any production interface.

No scientific benchmark has been run by the existence of this prototype.
