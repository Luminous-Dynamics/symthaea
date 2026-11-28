# topo-adapt-benchmark

A small deterministic evaluator for the synthetic fixtures owned by
[TOPO-ADAPT-001 #14123](https://github.com/Luminous-Dynamics/symthaea/issues/14123).

## Run

From the repository root:

```sh
cargo test -p topo-adapt-benchmark
cargo run -p topo-adapt-benchmark
```

The crate uses only the workspace's existing `serde` and `serde_json`
dependencies. The fixture is embedded at build time from
`fixtures/topo_adapt_fixture_v0_1.json`.

## Contract

- Directed integer capacities.
- Exactly one source per episode.
- Demand schedules are forward-time and sink-specific.
- Candidate edges are inactive until selected.
- Graph-edit budget is enforced before evaluation.
- Node/edge references and interventions are validated.
- Maximum flow is computed with an augmenting-path residual network.
- For these small fixtures, minimum cut is independently obtained by
  exhaustive partition enumeration; a flow/cut mismatch is a hard error.
- Graphs above 20 nodes are rejected by this v0.1 exhaustive-cut oracle.

This is a benchmark evaluator, not a learned controller. It does not model
biological transport, bioelectricity, physical repair, or industrial actuation.
