# topo-adapt-benchmark

A deterministic evaluator for the synthetic fixtures owned by TOPO-ADAPT-001
(issue #14123).

## Run

From the repository root:

    cargo test -p topo-adapt-benchmark
    cargo run -p topo-adapt-benchmark

The crate uses the workspace's existing serde, serde_json and blake3 dependencies.
The JSON fixture is embedded at build time from the fixtures directory.

The CLI prints the fixture BLAKE3 digest and an optional GITHUB_SHA. When the
source SHA is unavailable it emits null; such a run is not a source-bound
qualification receipt.

## Contract

- Directed integer capacities.
- Exactly one source per episode.
- Forward-time, sink-specific demand schedules.
- Candidate edges remain inactive until selected.
- Graph-edit budget enforced before evaluation.
- Node, edge and intervention references validated.
- Maximum flow calculated with an augmenting-path residual network.
- Minimum cut independently obtained by exhaustive partition enumeration.
- Flow/cut disagreement is a hard error.
- The exhaustive cut oracle is capped at 20 graph nodes in v0.1.

The evaluator reports flow, min-cut, accepted edit count, declared edit cost and
one-step edge-maintenance cost separately. Message bytes and controller updates
are declared fixture budgets but are not charged to a policy until a controller
actually emits message/update traces.

This is a benchmark evaluator, not a learned controller. It does not model
biological transport, bioelectricity, physical repair or industrial actuation.
