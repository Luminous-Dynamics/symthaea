# Connectome universe adequacy verifier

Rust-only semantic auditor for the query-scoped universe-adequacy report in:

- `research/connectomics/CONNECTOME_UNIVERSE_ADEQUACY_BENCHMARK_V0.1.md`
- `research/connectomics/connectome-universe-adequacy-report-v0.1.schema.json`
- `research/connectomics/connectome-universe-structure-artifacts-v0.1.schema.json`

## Invocation

From this crate directory, on a Rust-equipped host:

```sh
cargo test --locked
cargo run --locked -- report.json candidate-structures.json reference-structures.json
```

The three JSON inputs are separate files. Candidate-generation code must never receive the evaluator-only reference structure manifest.

Exit status:
- `0`: counters and report consistency independently recomputed;
- `2`: report, sidecar, digest, or semantic inconsistency rejected;
- `64`: usage error;
- `66`: input file could not be read.

The output is deterministic JSON with `status=COUNTERS_RECOMPUTED`, `source_report_sha256`, and `trusted_promotion=false`. It reports the input `claimed_enumeration_closure` separately from `worldset_closure_validation=NOT_PERFORMED_BY_THIS_TOOL`; it never echoes an upstream `CLOSED_VERIFIED` claim as a result established by this executable. The report's `evaluator_protocol_sha256`, `rules_sha256`, `canonicalizer_sha256`, `report_schema_sha256` and `structure_schema_sha256` must match the exact SHA-256 values of the versioned protocol, benchmark-rules and schema files in this repository. These are document/schema identity hashes, not a source-binary/build attestation.

## What is independently recomputed

- SHA-256 over exact candidate/reference sidecar bytes against the report's manifest digests;
- exact evaluator-protocol, adequacy-rules, canonicalizer, report-schema and structure-schema file digests;
- SHA-256 of the exact input report bytes in the verifier output (the report does not contain a self-referential report digest).
- Structural graph SHA-256 from canonical directed nodes/edges, with node and edge array order normalized.
- Sidecar/report world-ID set equality and uniqueness.
- Candidate/reference structural inclusion using recomputed graph hashes, not world-ID string equality.
- Query-state consistency for structurally equivalent worlds under the same frozen query.
- Candidate and reference duplicate-structure group counts.
- Evidence compatibility (compatible / incompatible / indeterminate) separately from query estimability within compatible worlds;
- Structural inclusion across all evidence-compatible references, and query-state coverage / decision flips only across query-estimable compatible references;
- Omitted query-flipping worlds, false invariance, false ambiguity, candidate abstention and per-family denominator reconciliation.
- Per-error-family counters, required-family coverage and aggregate denominator reconciliation.
- The reported adequacy disposition against independently recomputed counters.

## Deliberate limitations

This first semantic verifier does **not**:
- authenticate the #10204 pre-outcome freeze signature, trust root or append-only event log;
- independently replay the #10137 Cartesian-world closure receipt;
- prove that the declared or reference worlds are biologically exhaustive;
- canonicalize source EM meshes, voxel masks, neurite geometry or coordinate-frame transformations;
- estimate population prevalence or assign probabilities to challenge worlds;
- prove anatomical correctness, synaptic truth or causal connectivity;
- authorize annotation writes or scientific certification.

Consequently, this version cannot promote `ADEQUACY_SUPPORTED_WITHIN_TESTED_CHALLENGES`. A report without an observed counterexample remains `ADEQUACY_NOT_ESTIMABLE` until authenticated closure and freeze receipts are integrated. A computed `ADEQUACY_FAILED` is scoped only to the supplied, byte-pinned reference challenge set.

The graph canonicalizer V1 is a synthetic directed-graph identity representation. It preserves node/edge IDs and canonical JSON attributes; it is not a biological geometry canonicalizer. Do not use its hashes to assert equivalence of real neuronal meshes or voxel segmentations.

## Validation status

Fifteen Rust unit tests are authored. They cover:
- omitted query-flipping worlds and false ambiguity;
- compatibility-versus-query-estimability separation;
- sidecar-byte mutation, graph-digest mutation and schema/protocol digest mismatch;
- node/edge array-order invariance, edge-direction / attribute sensitivity, duplicate node IDs and dangling endpoints;
- inconsistent query states for one canonical structure;
- candidate abstention and false-invariance suppression;
- duplicate-structure group counting;
- unknown report fields, malformed receipt digests and refusal to promote without external trust receipts.

They have **not** been compiled or executed in this authoring environment because `cargo`, `rustc` and `rustfmt` are unavailable. No Rust test, formatting, Clippy or CI pass is claimed.
