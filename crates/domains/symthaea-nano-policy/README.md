# Symthaea Nano Policy (research-only prototype)

A small Rust CLI that selects a subset of a supplied candidate pool using three policies:
- maximin: greedy farthest-point selection over min-max-normalized numeric descriptors, maintaining nearest-selected distances incrementally;
- hdc_maximin: encode each normalized descriptor row as a Symthaea BinaryHV using dimension-role binding and a correlated 256-level scalar codebook, bundle it, then select by normalized Hamming distance;
- random: a seeded, deterministic random-control policy.

It is intentionally **not** an atomistic model, optical solver, candidate generator, toxicity model, or materials predictor. It has no target-label, energy, spectrum, physics-engine, network, file-system, or model-checkpoint interface. Strict Serde structs reject unknown fields, so an attempted extra target/label field in a candidate record fails closed at JSON parsing.

## Status and interface

- Interface: one JSON request from stdin; one JSON response to stdout; errors to stderr and exit code 2.
- Request schema version: nano-selection-input-v1.
- Machine-readable input/output contracts: schema/input.schema.json and schema/output.schema.json (JSON Schema Draft 2020-12).
- Deterministic plumbing fixtures are provided for maximin (fixtures/smoke-request.json and fixtures/smoke-expected.json) and HDC (fixtures/hdc-smoke-request.json and fixtures/hdc-smoke-expected.json).
- Candidate input fields: id, features only.
- Input budgets/candidate/feature cells are bounded.
- Maximin and hdc_maximin normalize features with min/max across the permitted unlabeled candidate pool; this is a declared transductive descriptor operation, not a use of target labels. The random-control arm deliberately skips normalization to avoid irrelevant overhead.
- HDC uses Symthaea core's 16,384-bit BinaryHV. Its 256 scalar levels are constructed by progressively flipping a deterministic permutation of half the bits, so level-code Hamming distance from level zero increases monotonically with normalized descriptor value. Descriptor dimension is positional: callers must freeze feature order, meaning, and units. This is a descriptor-encoding hypothesis, not a physical prior.
- Ties are deterministic by lexicographically sorted candidate ID.
- Random control uses a pinned local SplitMix64 implementation, with no OS entropy.
- Maximin scores are squared Euclidean distances in normalized feature space. HDC scores are normalized Hamming distances in encoded hypervector space. The first selection, when no prior selection exists, uses distance to the pool centroid/bundle; later selections use minimum distance to any selected candidate. Random outputs no score.
- HDC is capped at 10,000 candidates and 256 descriptor features per candidate. Stored candidate vectors use 2 KiB each; the maximum pool therefore needs about 20 MiB for encoded vectors, excluding feature arrays and transient bindings. These are defensive implementation ceilings, not measured performance limits.
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

Output contains the echoed input_schema_version, policy/version, pool dimensions, requested budget, prior selections and ordered selected candidate IDs with policy-specific scores. The CLI also emits input_blake3, the BLAKE3 digest of the exact raw request bytes (including whitespace); orchestration should hash exact response bytes separately for a complete run receipt. Scores are selection diagnostics—not probabilities, uncertainty, physical properties, novelty, stability, safety or optical performance.

## Build and test

From the Symthaea workspace root:

~~~bash
cargo test -p symthaea-nano-policy --locked
cargo run -p symthaea-nano-policy --release --locked < request.json
~~~

This crate uses workspace-pinned serde, serde_json, and blake3 plus Symthaea core's BinaryHV API. That path dependency pulls in Symthaea core's scientific/utility dependency graph. A dedicated targeted workflow checks format, tests, and Clippy. If the Rust toolchain is unavailable locally, don't claim local compilation; rely on exact-head workflow evidence only when its job actually completes successfully.

## Prototype acceptance and performance profile

Software acceptance before scientific use:
- all 21 active unit tests pass on the exact commit;
- formatting and Clippy pass with warnings denied;
- same canonical request produces the same ordered IDs for all three policies, independent of input row order;
- the ignored manual profile completes five repeats per case with a stable selected-ID digest;
- unknown target/outcome fields fail parsing;
- all candidate IDs and budget slots are accounted for, and no candidate is selected twice;
- the locked dependency graph is unchanged except for this package.

The maximin loop has target complexity O(N · D · (P + K)) where N is pool size, D is descriptor count, P is the number of already-selected IDs used for initialization, and K is the new selection budget; this is a complexity analysis, not a measured runtime claim.

The ignored Rust profile harness runs five repeats for these synthetic cases: random (10k candidates, 64 features, budget 100); Euclidean maximin (1k × 8 features, budget 100; 10k × 64 features, budget 10); HDC maximin (100 × 8, 1k × 8, 1k × 64, 10k × 8 features; budget 10). It reports p50/p95 wall time and a stable selected-ID digest for each case. Run with `cargo test -p symthaea-nano-policy --locked manual_profile_selector_scaling -- --ignored --nocapture`; on Linux, wrap it with `/usr/bin/time -v` to record process peak RSS. The manual CI dispatch can run this profile and reports memory via `/usr/bin/time -v`. These are source-defined profile cases, not measured runtime results or pass/fail targets. HDC has a lower hard candidate cap than descriptor-only arms. Do not set an absolute runtime pass target until a reference machine and baseline are measured; once measured, preregister the selector-overhead budget relative to the actual forward oracle.

## Required next steps before scientific use

1. Obtain descriptor rows from a rights-cleared and pinned nanophotonic dataset, and establish exact identity and group/split lineage.
2. Use only predeclared geometry-derived descriptors or allowed frozen embeddings; no target spectra or outcomes may appear in the candidate input.
3. Add independent baselines and an output receipt digest/canonicalization contract.
4. Compare random, maximin, task-native conventional search (e.g. MOCLIP/adjoint where appropriate), and one Symthaea policy under equal candidate/solver budgets.
5. Use a held-out target set, group-aware uncertainty, negative controls, and independent spectral/forward-solver evaluation.
6. Demonstrate useful value before adding this to any production interface.

No scientific benchmark has been run by the existence of this prototype.
