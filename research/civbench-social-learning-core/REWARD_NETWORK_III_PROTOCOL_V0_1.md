# Reward Network III: CivBench protocol v0.1

**Status:** literature-grounded protocol; raw archive not yet ingested or hashed. No reproduction claim.

## Source

Brinkmann, L., Eisenmann, T.F., Nussberger, A.-M., Derex, M., Bonati, S., Chirkov, V. & Rahwan, I. (2026). *Propagation and preservation of AI-discovered problem-solving strategies in human culture*. Nature Communications 17, 8255. 18 August 2026. DOI: https://doi.org/10.1038/s41467-026-76113-2

- Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13487184/
- Code/data: https://zenodo.org/records/18547122
- Experiment UI: https://center-for-humans-and-machines.github.io/reward-network-iii
- Code repository: https://github.com/center-for-humans-and-machines/reward-network-iii/tree/v2.0

## Task and empirical design

The paper reports 1,155 included participants, in 30 populations of five generations and eight positions each. Fifteen populations were Human-Only and fifteen Human-Machine; three Generation-0 positions in each Human-Machine population were machines. 148 of 1,303 completers were excluded under preregistered time/missing-move criteria and replaced.

The paper specifies a 12-node/30-edge task network, hidden node levels 0–3, rewards in `{-50, 0, 100, 200, 400}`, +400 edges sourced only at level 3, and at least three negative edges on every path from level 0 to level 3. Participants make ten moves. The maximum theoretical trajectory takes three −50 moves followed by seven +400 moves (2,650 points), while ten +200 moves yield the myopic ceiling (2,000 points). Participants begin at a predetermined node and initially see outgoing edges from that node; further outgoing edges are revealed as they visit nodes. The paper-level shape is therefore explicit. The exact generated network instances, edge orientation/revisit handling, tie-breaking, score implementation and instance IDs still must be imported from Zenodo v2.0 before claiming exact task reproduction.

The reported best score is 2,650 (three -50 moves followed by seven +400 moves). The myopic ceiling is 2,000 (ten +200 moves). The paper generated separate 1,000-network training, validation and experiment sets; the latter was held out. About 9% of generated networks were excluded because the myopic rule outperformed the loss-seeking rule.

Human participants in Generation 0 completed six individual-learning networks; later generations did two individual-learning networks plus four social-learning networks, including demonstrator selection from five candidates, observation/replay, replication, own attempt and feedback. Demonstration trials supplied content for later generations.

The published DQN used a GRU with two 15-unit ReLU linear layers, 20,000 training episodes, and a 1,000-network training set. Reproduce the paper's actual DQN only from its pinned code, data, weights and configuration; the transparent heuristics in this crate are not the DQN.

Reported outcomes: machine-discovered strategies were preserved across every generation in 9/15 Human-Machine populations; six additional groups had temporary adoption. Fourteen of fifteen Human-Only populations never discovered the strategy. 86.7% selected the highest-scoring demonstrator; 90% of Generation-1 learners in Human-Machine groups selected a machine. These are empirical reference results, not hard-coded expected outputs.

## Source-version reconciliation

A separate source audit is recorded in [LEGACY_GITHUB_SNAPSHOT_AUDIT.md](LEGACY_GITHUB_SNAPSHOT_AUDIT.md). The **2026 paper explicitly specifies** a 12-node/30-edge task, hidden node levels 0–3, +400 reward edges sourced at level 3, and at least three negative-reward edges on every level-0-to-level-3 path. The generic Rust engine exposes these as `PUBLISHED_NODE_COUNT`, `PUBLISHED_EDGE_COUNT`, `PUBLISHED_MAX_NODE_LEVEL`, `PUBLISHED_TERMINAL_LEVEL`, `PUBLISHED_REQUIRED_NEGATIVE_EDGES`, and `validate_published_profile()`.

The older public GitHub repository snapshot inspected separately contains 841 networks of 10 nodes/20 stored edges, levels 0–4 (the old model validator admits up to 5), and 8-transition heuristic solutions. Its frontend follows directed outgoing edges and caps the task at 8 transitions. It is **not** the source of truth for the paper's final ten-move task and is not asserted to be the final Zenodo v2.0 archive. The generic Rust engine retains explicit `LEGACY_GITHUB_*` constants and generic shape/level/terminal-contract validation so these profiles cannot be silently conflated. The exact v2.0 network files, traversal/revisit semantics, tie-breaking and source split still require archive ingestion.


## Zero-shot transfer and held-out generalization

The crate now includes `evaluate_zero_shot_transfer`, which applies one frozen, uninterpreted policy to multiple named network profiles and records policy reward, full-information oracle reward, regret, oracle fraction and invalid moves. Unit tests compare loss-seeking and immediate-greedy rules across the abstract ten-move synthetic incentive fixture and a separate synthetic compatibility fixture matching the older GitHub snapshot's eight-transition shape.

This is intentionally labelled **zero-shot rule transfer**, not learned-agent generalization. It establishes only that the reference rule behaves consistently across two constructed graph profiles. The definitive transfer benchmark must partition the Zenodo v2.0 task pool by network identity and generator/topology family, train/select policies on training/validation only, then evaluate on held-out network families with no shared network IDs or near-duplicate graph seeds. Report raw reward, regret to a clearly distinguished full-information oracle, reward fraction, invalid action rate, and performance by family. Also compare the frozen strategy description and teach-back artifact against simple heuristics; no policy should access hidden node levels or future edges.


## Separate exploratory agent-based model (ABM)

The paper's exploratory ABM is a different experimental layer from the human population study above. The Methods and Supplementary Note 5 specify `M=50` replications, `G=20` generations and `N_gen=20` agents per generation. The heterogeneous seed condition uses `N_m=5` machines. Humans have `k_h=10` individual exploration trials and `q_h=0.01) bias toward optimal exploration; the per-trial rates are `d_myo=0.5` and manipulated `d_opt`. Machine exploration capacity is `K_m ∈ {10, 1,000, 100,000}` and machine optimality bias `q_m ∈ {0.01, 0.5}`. The supplement varies social transmission `lambda` through learnability `1-lambda`, teacher-selection temperature `tau ∈ {0.1, 0.2, 0.4}`, demonstration-reward noise `epsilon ∈ {0.1, 0.2, 0.4}`, machine availability (none, seed generation only, every generation), seed machine count `N_m ∈ [1,5]`, and population size `N_total ∈ [20,100]`. Successful agents provide `K_demo=10` demonstrations. Each demonstration adds `R_opt=1` for optimal, `R_myo=0.5` for myopic, plus independent noise; the summed noise has distribution `epsilon_eff ~ Normal(0, epsilon^2*K_demo)`. Discovery uses `k_opt ~ Binomial(k,q_opt)`, `k_myo=k-k_opt`; optimal is prioritized if both strategies are discovered. Social transmission is Bernoulli with probability `lambda`; integration is `max(s_ind,s_social)`. The main text identifies `lambda=0.9` as a high-learnability case.

The branch's `src/paper_abm.rs` uses the exact terminal categorical probabilities implied by the Binomial allocation and independent per-trial discovery rates. This is distribution-equivalent for final strategy state, while avoiding up to 100,000 explicit trial draws; it does not reproduce the authors' exact random-number stream. The supplement says learners sample `k` candidate teachers from the preceding generation but the extracted description does not state the numeric `k`. CivBench exposes this unknown as an explicit sensitivity parameter (1/5/10/20), not an author-reported value. Machine-placement labels in the Rust model (clustered prefix/uniform random) are also explicit CivBench scenarios; the source's broader placement treatments are preserved as separate no-machine/seed-only/all-generation availability conditions.

The `paper-abm-sweep` binary runs 50 fixed-seed replications per condition across 167 conditions in four families: (A) machine discovery capacity/bias and representative log-spaced `d_opt`; (B) no/seed-only/all-generation machine availability crossed with `lambda`, `tau` and `epsilon`; (C) `N_m=1..5`, `N_total=20..100` and two declared placement schemes; (D) teacher candidate-set size 1/5/10/20 crossed with the published `tau` values. It reports total, human-only and machine-only optimal prevalence separately, machine-origin lineage among humans, reward and transmission rates, and 95% simulation-replicate confidence half-widths. This is a source-informed sensitivity suite, not a byte-for-byte reproduction of the full author grid or the original RNG stream.

Do not use the human experiment's 30 populations × five generations × eight positions as the ABM configuration. Conversely, do not present ABM trajectories as human participant results. Exact candidate sampling, full parameter grid, random-number implementation and author-code equivalence require Zenodo v2.0 and supplementary-source verification before reproduction is claimed.

## Baseline ladder

Run on identical frozen networks and move budgets:
1. seeded random legal move;
2. immediate-reward greedy;
3. loss-seeking (-50 edges preferred, otherwise greedy);
4. loss-then-greedy (declared loss count, then greedy);
5. full-information finite-horizon oracle (upper bound only; not an admissible participant policy);
6. reproduced paper DQN, after exact artifacts are ingested;
7. Symthaea cognitive/HDC/LTC contenders under the same task and splits.

## Required metrics

Separate total reward, regret to oracle, move-level agreement, strategy discovery, social acquisition, written articulation, demonstrator selection, inter-generation retention, correction after misinformation, transfer to held-out networks, and compute/time cost. Reward achievement alone does not prove conceptual understanding; written articulation alone does not prove executable skill.

## Paradigm-shift experiments

- **Discovery vs transmission:** isolate machine discovery from learner adoption and transfer.
- **Strategy compression:** compare full trajectories, action rules, executable programs, natural-language explanations and HDC representations at equal information budgets.
- **Cumulative improvement:** reward only strategies that survive transmission and improve held-out task outcomes.
- **Error-correcting culture:** inject corrupted demonstrations, contradictory teachers and later corrections; measure adoption and recovery.
- **Plural knowledge:** test context-dependent strategy portfolios rather than forcing one universal strategy.
- **Autonomy and provenance:** preserve authorship, evidence, uncertainty, challenge and the learner's option to reject a machine-originated strategy.

## Controls and data gates

Match the accumulated individual-learning time/trials in social-learning comparisons. A 2024 PNAS experiment found repeated individual learning outperformed chain/group conditions in one combinatorial task, illustrating why this control is important: https://pmc.ncbi.nlm.nih.gov/articles/PMC11621820/

Before analysis, download Zenodo v2.0, record file digest and record/version metadata, inspect its README/data dictionary, preserve raw and processed tables separately, identify population/generation/participant/network/move/demo IDs, and reproduce the authors' exclusions and derived-variable pipeline. The Zenodo record says it contains de-identified participant- and move-level raw/processed data, strategy codings, networks, solutions, model weights and ABM outputs.

The archive has not been ingested into this branch. No reanalysis or reproduction result is claimed. A small synthetic test fixture below must never be mistaken for one of the paper's original networks.

## Implementation assumptions that require source verification

The initial Rust V0 represents an action as a directed transition (from node to node) and resolves equal-reward/equal-loss choices by stable edge-index order. This is an explicit implementation choice for deterministic tests, not yet a verified property of the original experiment interface or network serialization. Before original-data replay, inspect the v2.0 network files and experiment code to establish whether source edges are directed or undirected, whether reverse traversal is available, how visited nodes/edges are treated, the legal-move rules, and the exact tie-breaking/randomization behavior. The importer must preserve the source representation and must not silently coerce undirected edges to directed arcs or vice versa.


## Three-axis cultural frontier implemented in this crate

The paper argues that machine-originated cultural change requires discoveries that are non-trivial, learnable, and have a recognisable selective advantage. The reference ABM exposes these as separate controls:

- **Discovery difficulty:** `optimal_discovery_probability_per_trial`, combined with human/machine exploration bias and their individual-learning budgets.
- **Learnability:** `social_observation_probability`, `social_strategy_acquisition_probability`, `demonstration_corruption_probability`, and `strategy_integration_probability`.
- **Selective advantage:** `optimal_demonstration_reward - myopic_demonstration_reward`, evaluated through noisy demonstrations and the configured teacher-selection mechanism.

The new `cultural-frontier-sweep` binary evaluates a 3×3×3 grid (optimal discovery probability 0.0001/0.001/0.01, social acquisition probability 0.25/0.60/0.95, selective advantage 0/0.25/0.50) with 256 fixed seeds per condition. It reports independent optimal discovery, integrated prevalence, social-only optimal acquisition and integration, top-teacher selection, and cross-generation persistence. The same seed range is reused across conditions to support paired sensitivity comparisons; the model remains stochastic and the results are simulation-replicate estimates.

This is a mechanism-level frontier exploration, not a fit to participant data and not a reproduction of the authors' ABM. Freeze the exact grid, seed range, code digest, model version and output digest before reporting results. The social-only metric counts optimal strategies acquired/integrated by agents that did not individually discover them; it avoids misclassifying independent rediscovery as cultural transmission.

## External AI cultural-transmission reference

A second important comparison target is DeepMind's cultural-general-intelligence work:
- Platonov et al. (2023), *Learning few-shot imitation as cultural transmission*, Nature Communications 14, 7536: https://www.nature.com/articles/s41467-023-42875-2
- Research summary: https://deepmind.google/blog/learning-robust-real-time-cultural-transmission-without-human-data/

That work studies real-time imitation from demonstrations on novel contexts and reports a training recipe (MEDAL-ADR) involving memory, expert dropout, attentional bias toward the expert and automatic domain randomization. Its authors report outperformance over ablations, including ME-AL, on held-out tasks. This is a different task family from Reward Network III and should be treated as a separate transfer benchmark, not as a directly comparable score.

CivBench should compare at least three axes:
- **Human-population transmission:** Reward Network III (does the strategy spread, become understood and persist in populations?).
- **Agent-to-agent few-shot transfer:** DeepMind-style demonstration imitation (can the learner infer, recall and generalize a behaviour after few examples?).
- **Mechanism-based social learning:** transparent diffusion, prestige/performance selection, Bayesian/evidence-weighted learning and Symthaea cognitive mechanisms (which mechanism improves correctness, retention, correction and transfer under controlled budgets?).

The 2024 PNAS control-condition study above is a further warning: social transmission is not automatically superior to individual learning. Comparisons must equalize accumulated learning opportunity and measure acquisition cost, fatigue and redundancy.


### Multi-seed sensitivity experiment

The `cultural-transmission-sweep` binary in this crate runs 256 deterministic seeds for each of seven preregistered reference-model conditions: no social transmission; uniform teacher choice with 0%, 5% or 20% corruption; and performance-softmax teacher choice with the same corruption levels. It emits per-generation optimal-strategy prevalence, demonstration observation, acquisition conditional on observation, optimal acquisition, integration conditional on acquisition, and selection of the top-scoring candidate. The prevalence interval is a normal-approximation 95% confidence half-width over simulation replicates; it is not a confidence interval for human population effects.

This experiment should be used to verify expected directions and interactions in the simplified mechanism model. It cannot establish human cultural universality, paper replication, or superiority over the archived DQN.

## Simplified executable generational model in this crate

`src/cultural_evolution.rs` is an independent reference ABM, not the paper's original agent-based model. It deliberately exposes separate state transitions:

1. individual discovery, with distinct first-generation/later-generation trial budgets and human/machine search biases;
2. teacher-candidate sampling;
3. teacher choice by uniform selection or performance softmax;
4. demonstration observation;
5. strategy acquisition conditional on observation;
6. optional transmission corruption (optimal → myopic → none);
7. integration into the learner's active strategy, independently parameterized from acquisition.

The model records the selected teacher and their strategy, what the learner acquired, whether integration occurred, and whether the selected teacher was the best-scoring member of the candidate set. Metrics separate individual discovery, demonstration observation, strategy acquisition, optimal-strategy acquisition, active integration and top-teacher selection.

This decomposition helps localize failure: a strategy may fail to spread because it was not discovered, because the wrong teacher was selected, because the learner did not observe the demonstration, because it was not acquired, because transmission corrupted it, or because the acquired strategy was not integrated. V0 still simplifies the task to a three-strategy repertoire, one teacher per learner, scalar demonstration performance, one-step corruption and a Bernoulli integration gate. It does not reproduce the authors' experiment or ABM; the archive and supplementary parameters must be ingested before any reproduction claim.

The sweep reports **individual optimal discovery** separately from **integrated optimal prevalence**. It also separates demonstration observation, content/strategy acquisition, corruption, and active integration. These are different denominators and must not be collapsed into an adoption score. Its reported 95% intervals are simulation-replicate uncertainty under the reference ABM, not confidence intervals for human participants.


The Rust task engine now preserves the article-level paper profile explicitly and a separate older-repository compatibility profile. Unit fixtures for each are synthetic. A structural pass on `validate_published_profile()` checks article-described invariants only; it does not prove archive origin, split membership, or exact reproduction of the v2.0 generator.


## LiveCultureBench adapter boundary

Pham et al. (ACL 2026) introduce LiveCultureBench as a multi-cultural dynamic-town simulation for goal completion and socio-cultural norm adherence, with a separate LLM verifier whose reliability is also evaluated. CivBench's `src/cultural_situation.rs` module now supplies reusable per-context metrics and verifier calibration (Brier, log loss, fixed-bin ECE, confusion matrix, audit coverage and abstention) plus Pareto comparisons that minimize norm violation without collapsing it into task completion.

The official repository currently contains only a README placeholder saying the data-download link is still to be added: https://github.com/thanhpv2102/LiveCultureBench. This module is not an official simulator port or a reproduction. Follow [LIVECULTUREBENCH_ADAPTER_PROTOCOL_V0_1.md](LIVECULTUREBENCH_ADAPTER_PROTOCOL_V0_1.md) and issue [#10038](https://github.com/Luminous-Dynamics/symthaea/issues/10038) for upstream pinning, data/split ingestion, verifier audits and exact reproduction gates.
