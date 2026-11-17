# Active symmetry experiment reference v0.1

First executable finite reference for SYMM-ACTIVE-EXPERIMENT-001 #12663.

## Core capability

Given a finite hypothesis family and a frozen deterministic probe/outcome table, this crate computes an exact minimum worst-case cost adaptive decision tree. It uses dynamic programming over candidate-subset masks and considers only probes explicitly marked admissible.

This differs from ranking probes by one-step information gain: the optimal first probe accounts for the entire future adaptive policy and its worst-case accumulated cost.

## Included baselines

- Exact minimax decision-tree search (correctness target).
- Greedy minimum-largest-outcome-partition probe.
- Maximum entropy probe under an explicit uniform-hypothesis prior.

Greedy and entropy recommendations are non-authoritative comparison arms. The uniform prior is an assumption; it is not a posterior.

## Certificate and replay

The certificate carries canonical bytes for the complete hypothesis/probe/cost/admissibility/outcome contract, the full adaptive tree, worst-case cost, explored subset count and work units. The independent verifier recomputes the exact plan from the original problem and compares the complete tree. Observation replay narrows hypotheses by following the matching branch.

Canonical bytes are deterministic but are not a cryptographic digest or signature. The exact deterministic observation table is assumed to be the declared finite model.

## Frozen fixture

The example includes four competing hypotheses and three probes. Exact minimax chooses the balanced orientation probe first and identifies any hypothesis in worst-case cost 2. It is cross-checked against a separate exhaustive recursive oracle.

## Reproduction

cargo fmt --manifest-path research/symmetry-active-experiment-001/Cargo.toml --all -- --check

cargo test --locked --manifest-path research/symmetry-active-experiment-001/Cargo.toml

cargo clippy --locked --manifest-path research/symmetry-active-experiment-001/Cargo.toml --all-targets -- -D warnings

cargo run --release --locked --manifest-path research/symmetry-active-experiment-001/Cargo.toml --example decision_tree_probe

## Claim boundary

Finite deterministic experiments only. It does not model noisy measurements, causal interventions, continuous group posteriors, physical experiment safety, or proof that a surviving candidate is a physical symmetry. No probe may be physically executed without a separate safety/authorization gate.
