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

The `networked-mediation-sweep` binary evaluates 4 shortcut probabilities × 5 mediation fractions × 2 recommender objectives (40 conditions), each with 256 common seeds, 32 agents and 8 generations. The adaptive-transmission simulator also uses identity-keyed discovery draws (agent/recipe) and route-slot acquisition draws (sender/receiver/route slot, excluding payload), so same-seed content-shuffle comparisons preserve the acquisition chance for the same scheduled route. See `ADAPTIVE_TRANSMISSION_PROGRAMS_PROTOCOL_V0_1.md` for remaining limits from policy-dependent candidate and route sets. It compares performance-only ranking with performance plus a bounded lineage-rarity bonus. It reports mean/maximum quality, knowledge coverage, seed-lineage persistence, distinct lineage count, effective diversity and HHI concentration, innovation, observation, acquisition, peer adoption and mediated adoption. The key question is whether lineage-aware ranking improves diversity without sacrificing too much capability quality.

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

## Held-out few-shot cultural transfer

The `src/few_shot_transfer.rs` adapter implements the cultural-transmission score used in the MEDAL-ADR paper. For held-out task expert score (E), learner score (A_{full}) with the expert present throughout, learner score (A_{half}) with the expert present for the first half and then absent, and solo learner score (A_{solo}):

`CT = 0.5 * ((A_full - A_solo)/E + (A_half - A_solo)/E)`

The adapter reports full-expert following gain, half-exposure gain, the paper's CT score, the half-exposure score normalized by expert score, and optional action fidelity. It rejects non-finite/non-positive expert scores, duplicate task IDs, training/evaluation task overlap, and (when required) evaluation tasks outside predeclared held-out task families. It macro-averages one vote per task and supplies a task-level 95% normal-approximation interval. This is a metric/split adapter, not a MEDAL-ADR implementation or GoalCycle3D environment.

The score is distinct from Reward Network III generational persistence. MEDAL-ADR evaluates agent-to-agent imitation, recall and transfer after expert dropout; Reward Network III evaluates discovery, teacher selection, transmission and retention across a human population. Keep their results on separate tracks.

## Original Reward Network III network assets

The optional `.github/workflows/reward-network-iii-task-assets.yml` workflow and `scripts/hydrate_reward_network_task_assets.py` hydrate only the generated train/test/experiment network sets and myopic/take-loss/random solution files from the archive's Git LFS server. It verifies each object OID and declared size, audits task-ID split disjointness, and checks graph/solution structure. Raw objects stay in temporary runner storage; the uploaded artifact contains aggregate counts and integrity receipts only. It explicitly excludes participant tables, raw sessions, move histories and written strategies. A successful LFS download is required before claiming that original network assets are available in the benchmark.

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

## Simplified strategy-adoption ablation model

The separate `src/paper_abm.rs` module is a compact, controllable strategy-adoption model. It is useful for testing causal mechanisms such as acquisition, integration, corruption and ancestry, but it is **not** a reproduction or semantic port of the Reward Network III archived exploratory ABM. Its sweep is now named `simplified-strategy-abm-sweep` to prevent this distinction being lost.

Run:
```sh
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin simplified-strategy-abm-sweep > simplified-strategy-abm-sweep.csv
```

## Archived Reward Network III exploratory ABM

The separate `src/reward_network_abm_reference.rs` module is the source-informed semantic port of the archived Zenodo v2.0 notebook `abm/abm.ipynb` (not the older `abm/model.py` helper). The notebook source snapshot SHA-256 is `49040b229968a5bbfa736d23b2d6fb687fadc9ae404a2225211fba636cba6999`; the enclosing archive SHA-256 is `a479efeb0024ce92be9952f9791b5c764fff82fbaa93e154e2cc3c31a66b7b36`.

The port preserves the notebook's actual transition structure:
- Strategy states: random (-1), myopic (0), optimal (1).
- Individual exploration: `k_opt ~ Binomial(K_ind, q_opt)`, with the remainder allocated to myopic attempts; per-trial discoverability is converted to cumulative discovery probability, with optimal discovery taking priority if both are found.
- Demonstration rewards: optimal=1, myopic=0.5, random=0; reward noise has standard deviation `epsilon * sqrt(K_demo)`.
- Teacher choice: sample `n_teacher` distinct candidates from the previous generation, then select stochastically using a softmax over their prior demonstration rewards with temperature `tau`.
- Social learning: transmit the selected teacher's previous strategy with probability `lambda`; the archived notebook maps failure to strategy 0 (myopic).
- Integration: `max(individual_strategy, social_strategy)`.
- Machine placement: none, first-generation seed only, or the first machine slots in every generation.

The port uses a deterministic Rust RNG, so it preserves the transition distributions and semantics but is **not NumPy bit-stream compatible**. The failed-social-learning fallback is retained exactly as the archive's reference treatment; a separate preserve-individual fallback is exposed as a counterfactual ablation. We also track individual discovery, social acquisition, integration and machine ancestry separately; these extra attribution fields do not change the state transition.

The archived notebook's main parameter families are represented by the sweep profiles:
- `human_only_discovery`: 50 replications; 20 agents × 20 generations; (K_h) log-spaced 10–10,000, (q_h=2^{-8}ldots2^{-1}), (d_{opt}=10^0ldots10^{-6}), (d_{myo}=0.5), (lambda=0.9), (K_{demo}=10).
- `machine_seed_discoverability`: (K_m in {10,1000,100000}), (q_m in {0.01,0.5}), ten (d_{opt}) values and ten (lambda) values.
- `teacher_temperature`: human / seed-machine / all-generation-machine conditions, (	au in {0.1,0.2,0.4}).
- `demonstration_noise`: (epsilon in {0.1,0.2,0.4}).
- `population_robustness`: population sizes 20/100 and machine counts 1/5.

A default smoke profile runs four small conditions with ten replications. Use `--profile all` for the full archived parameter families and `--replications 50` to match the paper's replication count (the full grid is computationally large). Example:
```sh
cargo run --release --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin reward-network-abm-reference-sweep -- --profile quick
cargo run --release --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin reward-network-abm-reference-sweep -- --profile all --replications 50 > reward-network-abm-reference.csv
```

**Reproduction boundary:** this is a deterministic semantic port of the archived notebook's model equations and experimental parameter families, not a bit-for-bit replay of NumPy RNG streams or proof that all author notebook outputs have been reproduced. The old `abm/model.py` helper in the same archive is materially different and has an apparent call-signature mismatch; do not substitute it for the self-contained notebook code. Original LFS network/model payloads and paper analysis still require hydration, verification and independent reruns.

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
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin simplified-strategy-abm-sweep > simplified-strategy-abm-sweep.csv
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin reward-network-abm-reference-sweep -- --profile quick
```

The example binary prints a small chain-network adoption trace.

## Simplified strategy-adoption sensitivity sweep

The `paper-abm-sweep` binary targets the article's separate 20-generation/20-agent exploratory ABM and runs 50 replications per condition. It reports 167 source-informed conditions in four families: (A) discovery capacity/bias and representative log-spaced optimal discoverability; (B) no/seed-only/all-generation machine availability crossed with social transmission, selection temperature and demonstration noise; (C) machine count 1–5, population size 20–100 and clustered/uniform-random placement; and (D) teacher candidate count 1/5/10/20 crossed with published temperature values. It reports overall, human-only and machine-only optimal prevalence separately; human machine-origin lineage; myopic/null prevalence; reward and social transmission; and 95% replicate confidence half-widths. The supplementary specifies machine capacities 10/1,000/100,000, biases 0.01/0.5, tau and epsilon values 0.1/0.2/0.4, and availability/prevalence conditions. It does not report the numeric candidate-set size in the extracted description. This suite is a source-informed sensitivity design, not an exact reproduction of the authors' full parameter grid or random stream.

## Multi-seed cultural-transmission sweep

The `cultural-transmission-sweep` binary runs 256 fixed seeds per condition and writes per-generation means and 95% normal-approximation confidence half-widths for independent optimal discovery, integrated optimal prevalence, demonstration observation, strategy acquisition, optimal-strategy acquisition, integration, optimal integration and top-teacher selection. Conditions compare uniform vs performance-softmax teachers, 0/5/20% transmission corruption and a no-social-transmission control. The two optimal-strategy metrics must remain separate: one measures individual discovery; the other measures the final integrated repertoire. The sweep is a sensitivity experiment over the simplified reference ABM, not published Reward Network III participant data or an exact replication of the authors' ABM. Configuration values and seeds must be frozen in any reported run receipt.

The `cultural-frontier-sweep` binary runs 256 common seeds across a 3×3×3 grid:
- optimal-strategy per-trial discovery probability: 0.0001 / 0.001 / 0.01;
- social acquisition probability (learnability): 0.25 / 0.60 / 0.95;
- demonstrable selective advantage: 0.00 / 0.25 / 0.50 above the fixed myopic reward signal of 0.5.

It reports independent discovery, integrated prevalence and simulation-replicate confidence half-widths, social-only optimal acquisition/integration, top-teacher selection and the probability of optimal strategy appearing in every post-seed generation. This operationalizes the paper's discovery–learnability–advantage frontier in the simplified model; it does not estimate human parameters or reproduce the authors' ABM.



## Intentional teaching under bandwidth constraints

The `src/intentional_teaching.rs` module separates **what should be taught** from **who should be trusted as a teacher**. It implements five packet-selection policies:
- random;
- declared-benefit only;
- evidence-calibrated (claimed benefit × reliability × (1 − uncertainty));
- learner-context aware;
- diversity-aware (evidence/context score plus a bounded lineage-rarity bonus).

The teacher operates under an explicit bandwidth budget. Each knowledge item has separate fields for claimed benefit, evaluator-only true benefit, reliability, uncertainty, intrinsic difficulty, learning-resource cost, validity and held-out transferability. Selection policies cannot access the true-benefit, validity or transferability fields.

The model records teacher offer, observation, acquisition, integration, correct execution, retention and held-out transfer as distinct stages. Individual-only attempts are reported as attempts rather than teacher offers, and run under a matched resource budget. The new `intentional-teaching-sweep` binary compares six conditions over 256 fixed seeds each and reports correct-execution/benefit means with simulation-replicate confidence intervals. It is a synthetic mechanism benchmark, not a reanalysis of the cited human experiments.

Protocol and literature boundary: [INTENTIONAL_TEACHING_PROTOCOL_V0_1.md](INTENTIONAL_TEACHING_PROTOCOL_V0_1.md).


## Intentional teaching and selective knowledge transmission

The `src/intentional_teaching.rs` module answers a different question from demonstrator selection: **which knowledge items should be taught under a limited bandwidth budget?** It separates resource cost from intrinsic learning difficulty and keeps evaluator-only true benefit, validity and held-out transferability out of the teacher-selection policy.

The module compares random, declared-benefit, evidence-calibrated, learner-context-aware and diversity-aware packet selection. It records teacher offers, observations, acquisitions, integrations, correct executions, false adoptions, retention and held-out transfer separately. The individual-only baseline receives the same resource budget and is explicitly not counted as a teacher offer.

The `intentional-teaching-sweep` binary runs 256 deterministic seeds per condition and reports mean correct-execution rate and realized benefit with 95% normal-approximation intervals over simulation replicates. These are synthetic mechanism checks, not effect estimates for the cited studies.

Protocol: [INTENTIONAL_TEACHING_PROTOCOL_V0_1.md](INTENTIONAL_TEACHING_PROTOCOL_V0_1.md).

## Limitations

- Binary beliefs are a tractable reference task, not a model of language, skill acquisition, norms, institutions, or plural interpretation.
- Topologies and source metadata are supplied by the experiment.
- There is no learning from raw artefacts or semantic content in V0.
- No cryptographic peer identity or transport protocol is implemented.
- No claim of psychological validity, cultural universality, human collective intelligence, or consciousness is made.

Use this crate to establish simple baseline behaviour and regression fixtures before integrating Symthaea's HDC/cognitive components.


## State-aware adaptive transmission over recipe DAGs

The new `src/adaptive_transmission.rs` module implements a small deterministic recipe-DAG world. It separates an agent's possession of a rule from possession of its prerequisites and from successful execution of that rule. This supports delayed complementarity: an agent can learn a useful recipe now and execute it only after receiving the missing ingredients later.

Six policies are compared: no social transmission, fixed-ring forwarding, fixed-ring with shuffled content, random network/content routing, state-aware frontier routing, and a state-aware content-shuffled ablation that preserves the state-aware sender/receiver schedule but changes which rule is sent. The state-aware heuristic prioritizes immediate executability, prerequisite match, collective novelty and downstream unlock potential.

The `adaptive-transmission-sweep` binary runs 324 conditions (3 synthetic DAG families × 3 population sizes × 2 shortcut densities × 3 recipient-attention capacities × 6 policies), with 128 seeds per condition. It reports final result coverage, maximum discovered depth, mean results/recipes held per agent, individual discoveries, social acquisitions, new collective results, total/redundant transmissions and recipient-attention usage.

This is a synthetic hand-coded reference baseline, not the 720-element Collective LittleAlchemy environment and not an LLM-evolved program. The 2026 paper's reported improvements (up to 37%) are not a score target or a result of this crate; tasks and score definitions differ. The central ablation to preserve is content shuffling under the same sender/receiver schedule.

Protocol: [ADAPTIVE_TRANSMISSION_PROGRAMS_PROTOCOL_V0_1.md](ADAPTIVE_TRANSMISSION_PROGRAMS_PROTOCOL_V0_1.md).

Run:
```sh
cargo test --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --all-targets
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin adaptive-transmission-sweep > adaptive-transmission-sweep.csv
```



## Privacy-preserving recipient-state disclosure experiment

The adaptive-transmission engine now distinguishes:
- `CentralFullInventory`: explicit research upper bound with full recipient state available to the scheduler;
- `CoarseRecipeReadiness`: recipient-side need signal containing only an unheld recipe ID and prerequisite-readiness scalar; the scheduler does not receive the recipient's full element/recipe inventory.

The coarse mode rejects collective-novelty filters/weights because those require a global inventory signal. Every run records its disclosure mode. The new `privacy-disclosure-sweep` binary compares centralized, coarse-readiness and no-social conditions over 3 task families × 256 fixed seeds, with paired coverage deltas, confidence intervals, messages, attention, acquisitions, redundancy and independent discovery metrics. This is a minimization experiment, **not** a cryptographic privacy or differential-privacy guarantee; repeated need signals can still reveal recipient state.

Protocol: [PRIVACY_PRESERVING_TRANSMISSION_PROTOCOL_V0_1.md](PRIVACY_PRESERVING_TRANSMISSION_PROTOCOL_V0_1.md).

Run:
```sh
cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin privacy-disclosure-sweep > privacy-disclosure-sweep.csv
```

## Privacy-preserving recipient-state disclosure

The adaptive-transmission engine now supports two explicit routing information boundaries:
- `CentralFullInventory`: centralized full recipient-state visibility as a research upper bound;
- `CoarseRecipeReadiness`: a recipient-side need signal with an unheld recipe ID and a prerequisite-readiness scalar, without giving the scheduler the recipient's full element/recipe inventory.

The coarse mode rejects collective-novelty program features that require a global inventory. Every `Run` records its disclosure mode. The `privacy-disclosure-sweep` binary compares central, coarse and no-social conditions over three synthetic task families and 256 fixed seeds per family, with paired coverage differences and message/attention/acquisition metrics. CI now executes this sweep after the Rust test gate and uploads the CSV artifact.

This is **data minimization, not a formal privacy guarantee**. The request/readiness signal can still reveal which recipe is needed and how close the recipient is to executing it. It does not implement differential privacy, secure aggregation, cryptographic confidentiality, consent enforcement or rights authorization.

Protocol: [PRIVACY_PRESERVING_TRANSMISSION_PROTOCOL_V0_1.md](PRIVACY_PRESERVING_TRANSMISSION_PROTOCOL_V0_1.md). Leakage-attack and authorization tests are tracked in [issue #10413](https://github.com/Luminous-Dynamics/symthaea/issues/10413).

## Cross-project adaptive-transmission event contract

The proposed event envelope and JSON Schema are:
- [ADAPTIVE_TRANSMISSION_INTERCHANGE_V0_1.md](ADAPTIVE_TRANSMISSION_INTERCHANGE_V0_1.md)
- [ADAPTIVE_TRANSMISSION_EVENT_V0_1.schema.json](ADAPTIVE_TRANSMISSION_EVENT_V0_1.schema.json)

The event references existing skill capsules and claim/run receipts. It records sender/recipient references, exact payload digest and scope, policy identity, consent mode, minimized recipient-need signal reference, route/budget and separate lifecycle outcomes (delivery, observation, acquisition, integration, execution, retention, correction and transfer). It does not require a central service to receive full private agent inventories and is not a second general-purpose evidence ledger. This is a proposed contract; runtime authorization and rights enforcement remain separate implementation work.
