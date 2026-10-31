# Connectome perturbation synthetic oracle

This standalone, dependency-free Rust crate provides a deterministic known-answer harness for the first perturbation-dynamics engineering milestone.

It is deliberately **not** a biological simulator, a neural activity predictor, or a causal discovery model. The discrete-time linear system exists to validate response propagation and scoring semantics before any biological dataset adapter is introduced.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-perturbation-oracle/Cargo.toml
cargo run --locked --manifest-path tools/connectome-perturbation-oracle/Cargo.toml -- --json
cargo run --locked --manifest-path tools/connectome-perturbation-oracle/Cargo.toml -- --text
```

The crate is a standalone workspace under tools/ and has no third-party dependencies. It must not be added to the production workspace dependency graph by accident.

## Current known-answer cases

1. Direct one-step transmission with an exact response amplitude.
2. Signed two-hop propagation with a known path-delay and weight product.
3. Explicit edge delay, distinguished from the number of graph hops.
4. Recurrent feedback with a finite, known impulse response over the declared horizon.
5. Shared external input that creates passive co-variation but no interventional effect at the untreated recipient.
6. A rewired-topology negative control that fails to reproduce the reference response.

The scorer compares complete finite-horizon trajectories, not only peak response or correlation. The library also exposes directed reachability and an exact pairwise edge-deletion interaction oracle. Its fixtures distinguish redundant parallel routes (two individually dispensable edges become jointly critical) from serial bottlenecks (each edge is individually critical). This graph-query interaction is separate from physiological causal effect. Invalid node/edge indices, duplicate removals, zero delays, non-finite parameters, invalid horizons, unreachable baselines for interaction contrasts, and shape mismatches fail closed.

## Scientific claim ceiling

The report always labels its evidence class `synthetic_known_answer` and claim ceiling `SYNTHETIC_ONLY`. Passing tests establishes only that this small mathematical reference implementation behaves as specified. It does not establish biological validity, model quality, causal identification in an animal, or state-of-the-art performance.

## Edge-criticality oracle

`directed_path_exists` evaluates a frozen reachability query after removing declared edge indices. `pairwise_edge_reachability_interaction` computes the exact binary-loss interaction for two edge removals:

`I(a,b) = L(G \\ {a,b}) - L(G \\ {a}) - L(G \\ {b}) + L(G)`

A positive interaction in the parallel-route fixture means neither deletion alone breaks reachability, but their joint deletion does. A negative interaction in the serial-chain fixture means either deletion alone already breaks the query. These are graph-query semantics, not biological importance scores; biological criticality remains task-, state- and model-dependent.

## Planned extensions

Only after the current reference suite is independently reviewed:
- intervention-dose and duration curves;
- multiple perturbation targets and factorial interaction fixtures;
- nonlinear saturation and state-dependent dynamics;
- measurement noise and calcium observation kernels;
- partial observation, missing cells and registration uncertainty;
- animal-level population variability;
- matched-capacity activity-only, graph-constrained and graph-shuffled model adapters;
- evaluator/runner separation for measured biological outcomes.
