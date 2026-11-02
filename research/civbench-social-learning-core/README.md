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

The current 12-node/30-edge graph is a synthetic unit-test fixture only; it does not claim to match the published task shape. It checks the abstract three-loss/2,000-versus-2,650 incentive contrast. The paper specifies ten moves, but the exact final v2.0 network topology must be taken from the Zenodo archive. The older public GitHub repository snapshot is separately audited in [LEGACY_GITHUB_SNAPSHOT_AUDIT.md](LEGACY_GITHUB_SNAPSHOT_AUDIT.md): its 841 networks are 10-node/20-edge, levels 0–4, with 8-transition solutions. Do not substitute that historical profile for Zenodo v2.0. No final v2.0 archive digest or paper replication is claimed yet.


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
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin cultural-transmission-sweep > cultural-transmission-sweep.csv
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin cultural-frontier-sweep > cultural-frontier-sweep.csv
```

The example binary prints a small chain-network adoption trace.

## Multi-seed cultural-transmission sweep

The `cultural-transmission-sweep` binary runs 256 fixed seeds per condition and writes per-generation means and 95% normal-approximation confidence half-widths for independent optimal discovery, integrated optimal prevalence, demonstration observation, strategy acquisition, optimal-strategy acquisition, integration, optimal integration and top-teacher selection. Conditions compare uniform vs performance-softmax teachers, 0/5/20% transmission corruption and a no-social-transmission control. The two optimal-strategy metrics must remain separate: one measures individual discovery; the other measures the final integrated repertoire. The sweep is a sensitivity experiment over the simplified reference ABM, not published Reward Network III participant data or an exact replication of the authors' ABM. Configuration values and seeds must be frozen in any reported run receipt.

The `cultural-frontier-sweep` binary runs 256 common seeds across a 3×3×3 grid:
- optimal-strategy per-trial discovery probability: 0.0001 / 0.001 / 0.01;
- social acquisition probability (learnability): 0.25 / 0.60 / 0.95;
- demonstrable selective advantage: 0.00 / 0.25 / 0.50 above the fixed myopic reward signal of 0.5.

It reports independent discovery, integrated prevalence and simulation-replicate confidence half-widths, social-only optimal acquisition/integration, top-teacher selection and the probability of optimal strategy appearing in every post-seed generation. This operationalizes the paper's discovery–learnability–advantage frontier in the simplified model; it does not estimate human parameters or reproduce the authors' ABM.


## Limitations

- Binary beliefs are a tractable reference task, not a model of language, skill acquisition, norms, institutions, or plural interpretation.
- Topologies and source metadata are supplied by the experiment.
- There is no learning from raw artefacts or semantic content in V0.
- No cryptographic peer identity or transport protocol is implemented.
- No claim of psychological validity, cultural universality, human collective intelligence, or consciousness is made.

Use this crate to establish simple baseline behaviour and regression fixtures before integrating Symthaea's HDC/cognitive components.
