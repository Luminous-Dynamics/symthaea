# CivBench Social Learning Core

See [CIVBENCH_RESEARCH_ROADMAP_V0_1.md](CIVBENCH_RESEARCH_ROADMAP_V0_1.md) for the benchmark portfolio, industry comparison, proposed targets, reproduction gates, and the staged path for making defensible superiority claims.

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

## Networked cultural learning and algorithmic mediation

The separate `src/networked_mediation.rs` module models a connected ring social graph with configurable long-range shortcuts and a global top-k recommender. It exposes peer-only, recommender-only and hybrid access through a mediation fraction. It tracks demonstration observation, successful acquisition, actual adoption, individual innovation, original-seed lineage, quality coverage, unique lineages, lineage HHI and effective diversity separately.

The `networked-mediation-sweep` binary evaluates 4 shortcut probabilities × 5 mediation fractions × 2 recommender objectives (40 conditions), each with 256 common seeds, 32 agents and 8 generations. It compares performance-only ranking with performance plus a bounded lineage-rarity bonus. It reports mean/maximum quality, knowledge coverage, seed-lineage persistence, distinct lineage count, effective diversity and HHI concentration, innovation, observation, acquisition, peer adoption and mediated adoption. The key question is whether lineage-aware ranking improves diversity without sacrificing too much capability quality.

This is a simplified mechanism experiment, not a reproduction of Czaplicka, Baumann & Rahwan (2025). Their model has branch/complexity-level cultural traits, finite effort budgets, innovation/social/recommender costs, and an Erdős–Rényi social network. The reference port is separately tracked; do not use this sweep as a reproduction claim.

Run:
```sh
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin networked-mediation-sweep > networked-mediation-sweep.csv
```

## Czaplicka et al. 2025 source-faithful cultural mediation port

The separate `src/czaplicka_2025.rs` module is a Rust semantic port of the pinned author C implementation `aga-cz/Mutual-benefits-of-SL-and-RS/modelQinn.c` (blob `8555ec4ad8bd7eec0605bcda28565fbfe2645957`). It preserves the source model's Erdős–Rényi graph, level-indexed global best-payoff archive, finite effort budget, innovation/social/recommendation costs, sequential within-step updates, death-birth reset, neighbour-payoff selection and best-trait recommendation. It adds explicit trait lineage/adoption telemetry and preserves the source's odd prefix-max metrics under legacy names alongside true maxima.

This is a deterministic semantic port, not a bit-for-bit reproduction: the C source seeds platform `rand()` from wall-clock time; Rust uses fixed SplitMix64 seeds. The source parameter X is allocated but unused in the C update rules. The reference path preserves source semantics where practical and explicitly documents edge-case and RNG differences.

The `czaplicka-2025-sweep` binary has three modes:
- default quick smoke sweep;
- `--paper-grid`: N=100, T=20N, Q=200, 20 ER densities × 11 recommendation shares, reference global-best policy only;
- `--extension-grid`: smaller declared sensitivity grid comparing the reference with an exposure/adoption-count-aware UCB-style recommender.

The UCB-style recommender is a new CivBench intervention, not part of the authors' model. It uses a bonus for promising but less-adopted traits and must be evaluated on a Pareto frontier of payoff, complexity, unique traits and lineage concentration. It is not assumed to improve culture just because it increases diversity.

See [CZAPLICKA_2025_REFERENCE_PORT_PROTOCOL_V0_1.md](CZAPLICKA_2025_REFERENCE_PORT_PROTOCOL_V0_1.md) for exact source identifiers, update semantics, discrepancies and reproduction gates. The separate `src/networked_mediation.rs` ring-plus-shortcut model remains a different synthetic mechanism experiment; do not pool its results with this reference model.

## Iterated content drift and cultural attractors

The separate `src/iterated_transmission.rs` module models repeated transmission of four normalized content attributes: toxicity, positivity, difficulty and length. A channel step applies a declared attribute-specific bias toward a configurable attractor plus noise, scaled by an open-endedness parameter. The evaluator reports signed/absolute drift by attribute, final distance to each attractor, initial/final pairwise distance and a convergence index.

The `telephone-attractor-sweep` binary tests 4 chain lengths × 4 open-endedness values (16 conditions), 128 replicate seed sets per condition, and four distinct initial signals. Attractor targets, bias strengths and noise are synthetic controls—not measured parameters of a particular LLM.

This is a metric/experiment-design fixture, not an implementation of text generation or a reproduction of Perez et al. (ICLR 2025). Real-model comparisons require pinned provider/model/prompt identities, preserved text chains, calibrated attribute scorers and independent human review.

Protocol: [ITERATED_TRANSMISSION_ATTRACTOR_PROTOCOL_V0_1.md](ITERATED_TRANSMISSION_ATTRACTOR_PROTOCOL_V0_1.md).

Run:
```sh
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin telephone-attractor-sweep > telephone-attractor-sweep.csv
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin branching-culture-sweep > branching-culture-sweep.csv
```

## Branching cultural populations

The iterated-transmission module also exposes `simulate_branching_population`: a fixed-size population where each descendant selects one parent from the previous generation, inherits that root lineage and transforms the content through the same attribute-specific channel. It records parent indices, root identity, lineage extinction, HHI/effective diversity, pairwise content distance and cumulative distance from the root artifact.

The `branching-culture-sweep` binary evaluates 4 open-endedness values × 4 generation depths (16 conditions), with 128 replicate populations per condition and four distinct initial lineages. This complements the linear telephone chain: linear chains measure cumulative drift along one path; branching populations measure which cultural lineages survive, dominate or disappear. The parent rule is uniform random, so results are a neutral-drift mechanism fixture—not an empirical model of human or LLM parent selection.

Run:
```sh
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin branching-culture-sweep > branching-culture-sweep.csv
```

## Zero-shot rule transfer

The Reward Network engine exposes `evaluate_zero_shot_transfer`, which applies one frozen policy to multiple named graph profiles and reports policy reward, full-information oracle reward, regret, oracle fraction and invalid moves. Tests compare the same loss-seeking and immediate-greedy rules across two distinct synthetic profiles: the abstract 10-move incentive fixture and a historical-shape 8-transition fixture. The result is a *rule-transfer sanity check*, not evidence that Symthaea's learned representations or human cultural skills transfer across real graph families. Final v2.0 transfer tests must use held-out Zenodo networks grouped by topology/reward-family, not the synthetic fixtures.

## Reward Network III protocol

See [REWARD_NETWORK_III_PROTOCOL_V0_1.md](REWARD_NETWORK_III_PROTOCOL_V0_1.md) for the empirical task definition, study design, published result references, baseline ladder, transfer/correction experiments, and Zenodo ingestion gates. The Rust module `src/reward_network.rs` implements Random, ImmediateGreedy, LossSeeking and LossThenGreedy policies plus a full-information finite-horizon oracle. The oracle is an upper bound, not a participant policy.

The paper explicitly specifies 12 nodes, 30 edges, hidden levels 0–3, +400 rewards from level 3, and at least three negative edges on every level-0-to-level-3 path. The Rust engine's `validate_published_profile()` checks those structural invariants. The included 12-node/30-edge graph is a synthetic unit-test fixture that matches the published task shape and demonstrates the abstract three-loss/2,000-versus-2,650 incentive contrast; it is not one of the paper's generated networks. The older public GitHub repository snapshot is separately audited in [LEGACY_GITHUB_SNAPSHOT_AUDIT.md](LEGACY_GITHUB_SNAPSHOT_AUDIT.md): its 841 networks are 10-node/20-edge, levels 0–4, with 8-transition solutions. Do not substitute that historical profile for the paper's ten-move published task or Zenodo v2.0. No final v2.0 archive digest or paper replication is claimed yet.


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

## Paper-shaped exploratory ABM

The separate `src/paper_abm.rs` models the paper's **exploratory agent-based simulation**, not the human experiment and not the task-level DQN. Its reference population has 20 generations and 20 agents per generation; heterogeneous conditions seed five machine agents in generation zero. Human settings use 10 exploration trials and optimality bias 0.01; myopic discovery rate is 0.5. It also represents ten demonstrations, optimal/myopic demonstration rewards of 1/0.5, and reward noise with standard deviation `epsilon * sqrt(K_demo)`.

Discovery is sampled from the exact terminal-outcome distribution implied by independent per-trial choices: each trial attempts optimal with probability `q` (otherwise myopic) and succeeds at its strategy's per-trial discovery rate; when both strategies are discovered, optimal takes priority. This closed form avoids simulating up to 100,000 RNG steps per machine agent while preserving the terminal distribution under that model. The exact authors' random draw ordering and any supplementary implementation details still require Zenodo v2.0 review.

Not every ABM parameter is specified in the article's main text. In particular, V0's five-teacher candidate count, default `d_opt=0.001`, default machine capacity, `tau`, and the two placement treatments are sensitivity-design choices, not claimed author defaults. The paper-shaped sweep varies these explicitly where appropriate.

The `paper-abm-sweep` binary runs 50 fixed-seed replications per condition across 167 source-informed CivBench conditions in four families: discovery capacity/bias; machine availability crossed with social transmission, selection temperature and reward noise; robustness to machine count (1–5), population size (20–100) and declared placement schemes; and teacher candidate-set size (1/5/10/20). It reports total, human-only and machine-only optimal prevalence, machine-originated prevalence among humans, myopic/null prevalence, reward and transmission rates, plus 95% simulation-replicate confidence half-widths. The supplementary specifies the capacities, biases, tau/epsilon values and availability/prevalence treatments; the numeric teacher sample size remains unreported in the extracted description and is therefore varied as a CivBench sensitivity. The sweep is source-informed, not an exact reproduction of the author's RNG stream or every source condition.

Run:
```sh
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin paper-abm-sweep > paper-abm-sweep.csv
```

## Cross-cultural situation and verifier evaluation

The separate `src/cultural_situation.rs` module provides an adapter-neutral evaluation layer inspired by LiveCultureBench (Pham et al., ACL 2026). It accepts per-episode goal completion, norm violation, profile faithfulness, contextual awareness and coherence scores; aggregates them overall and by explicitly declared context ID; and computes per-context verifier Brier score, log loss, fixed-bin ECE, precision/recall/F1, human-audit coverage and abstention rate.

The module keeps norm violation separate (lower is better) and exposes Pareto dominance rather than a single blended score. Missing human labels remain unaudited, not negative; a missing verifier prediction is recorded as abstention. Context IDs are declared task metadata and must not be inferred from sensitive person attributes.

This is **not** the official LiveCultureBench runner or a reproduction. Its official repository currently contains only a README saying its data download link is still to be updated. The official environment, prompt/code/data schema, verifier definitions, calibration split and human annotations must be pinned before an adapter or comparative result is claimed. See [LIVECULTUREBENCH_ADAPTER_PROTOCOL_V0_1.md](LIVECULTUREBENCH_ADAPTER_PROTOCOL_V0_1.md).

## LiveCultureBench adapter layer (ACL 2026)

Pham et al.'s LiveCultureBench evaluates agents in a simulated town on goal completion, norm violation, profile faithfulness, contextual awareness, coherence and verifier reliability. The official upstream repository currently has only a placeholder README indicating that the data download link is still to be added. CivBench therefore does not claim to run or reproduce LiveCultureBench yet.

The separate `src/cultural_situation.rs` module provides adapter-neutral, explicit-context aggregation; per-context verifier Brier score, log loss, ECE, confusion metrics, human-audit coverage and abstention; and Pareto comparison that minimizes norm violations while maximizing the other four dimensions. It does not infer cultural identity from sensitive traits and does not collapse task success and norm safety into one score.

Protocol: [LIVECULTUREBENCH_ADAPTER_PROTOCOL_V0_1.md](LIVECULTUREBENCH_ADAPTER_PROTOCOL_V0_1.md). Upstream ingestion, exact reproduction and human-audited verifier validation are tracked in [issue #10038](https://github.com/Luminous-Dynamics/symthaea/issues/10038).

## Run

```sh
cargo test --locked --manifest-path research/civbench-social-learning-core/Cargo.toml
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin cultural-transmission-sweep > cultural-transmission-sweep.csv
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin cultural-frontier-sweep > cultural-frontier-sweep.csv
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin paper-abm-sweep > paper-abm-sweep.csv
```

The example binary prints a small chain-network adoption trace.

## Paper ABM sensitivity sweep

The `paper-abm-sweep` binary targets the article's separate 20-generation/20-agent exploratory ABM and runs 50 replications per condition. It reports 167 source-informed conditions in four families: (A) discovery capacity/bias and representative log-spaced optimal discoverability; (B) no/seed-only/all-generation machine availability crossed with social transmission, selection temperature and demonstration noise; (C) machine count 1–5, population size 20–100 and clustered/uniform-random placement; and (D) teacher candidate count 1/5/10/20 crossed with published temperature values. It reports overall, human-only and machine-only optimal prevalence separately; human machine-origin lineage; myopic/null prevalence; reward and social transmission; and 95% replicate confidence half-widths. The supplementary specifies machine capacities 10/1,000/100,000, biases 0.01/0.5, tau and epsilon values 0.1/0.2/0.4, and availability/prevalence conditions. It does not report the numeric candidate-set size in the extracted description. This suite is a source-informed sensitivity design, not an exact reproduction of the authors' full parameter grid or random stream.

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
