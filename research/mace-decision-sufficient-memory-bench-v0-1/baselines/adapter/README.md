# MACE temporal BM25 candidate-fixture adapter

This isolated Rust package binds the strict candidate-visible JSON fixture to the
standalone BM25/RRF reference and resolves every returned BM25 event ID back to
its typed source authority and lineage metadata.

The parser uses Serde with `deny_unknown_fields`. It accepts only
`mace_raw_trajectory_candidate_v0_1`; evaluator labels, hidden truth or
unrecognized candidate fields are not accepted. The adapter runs the frozen
synthetic JSON fixture through:

1. strict JSON decoding;
2. source-event metadata validation and bitemporal scope/time index construction;
3. BM25 retrieval at the declared scope and as-of time;
4. exact event-ID resolution to source ID, authority, uncertainty, parent links,
   source family/root and valid-time interval;
5. weighted RRF over supplied candidate-visible ranked channels.

The returned authority fields are evidence inputs, not an authorization result.
The MACE qualifier remains the authority for admissibility and decision
sufficiency. A BM25 or RRF score never overrides revocation, scope or trust gates.

Run using the nested package manifest:

`cargo test --locked --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/baselines/adapter/Cargo.toml`

This is a deterministic synthetic fixture adapter, not a production Add/Search
service, a full JSON Schema engine, an AML submission client, or a benchmark
result.
