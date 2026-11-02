# CivBench Social Learning Core

A dependency-free, deterministic Rust reference model for testing whether richer cognitive/social-learning mechanisms add value beyond simple diffusion rules.

This is a benchmark baseline, not a human-behaviour simulator and not a theory of culture. It contains two distinct task slices: a binary-proposition diffusion model, and a Reward Network III task environment with explicit reward edges and local-observation policies.

## Implemented reference rules

- **NoLearning:** agents retain their current state.
- **MajorityImitation:** adopt the strict majority belief among informed neighbours; ties preserve the current state.
- **ThresholdMajority:** require a declared minimum number of informed neighbours before strict-majority adoption.
- **PrestigeImitation:** copy the belief of the highest-prestige informed neighbour; equal prestige ties resolve to the lowest node ID.
- **EvidenceWeighted:** combine neighbours' declared reliability weights and an optional self-prior weight; adopt only when the weighted margin reaches a declared threshold.

Declared source reliability is an experimental input. It is not evidence that the source is truly reliable, and it must not be confused with the receiving agent's confidence or trustworthiness.

## Evaluation

The evaluator knows the synthetic ground truth, but the learning policy does not receive it. Metrics report coverage, accuracy among informed agents, consensus, confidence, and high-confidence error fraction separately.

Updates are synchronous: all agents read the previous state and write the next state. This prevents node iteration order from changing the result.

## Reward Network III protocol

See [REWARD_NETWORK_III_PROTOCOL_V0_1.md](REWARD_NETWORK_III_PROTOCOL_V0_1.md) for the empirical task definition, study design, published result references, baseline ladder, transfer/correction experiments, and Zenodo ingestion gates. The Rust module `src/reward_network.rs` implements Random, ImmediateGreedy, LossSeeking and LossThenGreedy policies plus a full-information finite-horizon oracle. The oracle is an upper bound, not a participant policy.

The current 12-node/30-edge test graph is synthetic. It checks the published task shape and reproduces the 2,000-versus-2,650 reward contrast under a deliberately constructed three-loss route. It is not an original Reward Network III network. The Zenodo v2.0 archive has not been downloaded or hashed in this branch; no paper replication is claimed.


## Generational cultural-transmission model

The separate `src/cultural_evolution.rs` module is a parameterized reference ABM with:
- separate individual-learning budgets for the seed and later generations;
- distinct human/machine search budgets and optimal-exploration biases;
- noisy demonstration performance;
- uniform or performance-softmax teacher selection from sampled candidates;
- distinct demonstration observation, strategy acquisition, transmission corruption and active integration probabilities;
- strategy downgrading under corrupted demonstrations;
- generation-by-generation prevalence, acquisition, integration and teacher-selection metrics.

It follows the broad mechanism sequence described in the 2026 Reward Network III paper but is **not** a reproduction of the authors' ABM. The published supplementary parameters and archived implementation must be ingested before claiming replication. Tests cover fixed-seed reproducibility, perfect transmission, empty-teacher and no-transmission controls, observation without acquisition, acquisition without integration, corruption, and the distinction between individual discovery and social acquisition.

## Run

```sh
cargo test --locked --manifest-path research/civbench-social-learning-core/Cargo.toml
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml
```

The example binary prints a small chain-network adoption trace.

## Limitations

- Binary beliefs are a tractable reference task, not a model of language, skill acquisition, norms, institutions, or plural interpretation.
- Topologies and source metadata are supplied by the experiment.
- There is no learning from raw artefacts or semantic content in V0.
- No cryptographic peer identity or transport protocol is implemented.
- No claim of psychological validity, cultural universality, human collective intelligence, or consciousness is made.

Use this crate to establish simple baseline behaviour and regression fixtures before integrating Symthaea's HDC/cognitive components.
