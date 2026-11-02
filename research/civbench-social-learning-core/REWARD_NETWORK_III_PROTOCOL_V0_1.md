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

The task network has 12 nodes and 30 edges. Rewards are in `{-50, 0, 100, 200, 400}`; nodes have hidden levels 0–3. Network-generation constraints ensure any path from a level-0 node to a level-3 node includes at least three negative-reward edges. +400 rewards are available only from level 3; level-0 transitions include +200 rewards. Participants begin at a predetermined node and initially see only its outgoing edges; further edges are revealed as nodes are visited. Each trial has ten moves.

The reported best score is 2,650 (three -50 moves followed by seven +400 moves). The myopic ceiling is 2,000 (ten +200 moves). The paper generated separate 1,000-network training, validation and experiment sets; the latter was held out. About 9% of generated networks were excluded because the myopic rule outperformed the loss-seeking rule.

Human participants in Generation 0 completed six individual-learning networks; later generations did two individual-learning networks plus four social-learning networks, including demonstrator selection from five candidates, observation/replay, replication, own attempt and feedback. Demonstration trials supplied content for later generations.

The published DQN used a GRU with two 15-unit ReLU linear layers, 20,000 training episodes, and a 1,000-network training set. Reproduce the paper's actual DQN only from its pinned code, data, weights and configuration; the transparent heuristics in this crate are not the DQN.

Reported outcomes: machine-discovered strategies were preserved across every generation in 9/15 Human-Machine populations; six additional groups had temporary adoption. Fourteen of fifteen Human-Only populations never discovered the strategy. 86.7% selected the highest-scoring demonstrator; 90% of Generation-1 learners in Human-Machine groups selected a machine. These are empirical reference results, not hard-coded expected outputs.

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
