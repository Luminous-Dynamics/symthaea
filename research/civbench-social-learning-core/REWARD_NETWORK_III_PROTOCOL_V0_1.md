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
