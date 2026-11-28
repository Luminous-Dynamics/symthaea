# Symthaea Game Intelligence Benchmark v0.1

**Tracking issue:** [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137)  
**Status:** Research specification; implementation and qualification pending  
**Date:** 2026-11-28  
**Scope:** Interactive game-playing as a controlled test of reusable cognitive capability

## Executive thesis

Games are valuable to Symthaea not because a game score is a proxy for intelligence, but because interactive environments expose an agent to a compact world with state, dynamics, goals, constraints, uncertainty, actions, and consequences. A well-designed game campaign can test whether Symthaea can construct a useful state and world model, distinguish known rules from hypotheses, choose actions and information-gathering interventions, plan through delayed consequences and irreversible actions, remember events and transfer reusable skills, detect world changes, recover, and produce independently replayable trajectories.

The proposed contribution is **mechanism-grounded interactive evaluation**: every task outcome is paired with evidence about the predictions, beliefs, interventions, memory use, actions, and recovery that produced it. This is stronger than a leaderboard score and narrower than a claim of general intelligence.

No Symthaea game-playing result is claimed in this document.

## 1. Research landscape and the bar to clear

### 1.1 Existing suites answer different questions

| Suite | What it is good for | Main limitation for this programme | Planned use |
|---|---|---|---|
| [BALROG](https://balrogai.com/) | Agentic reasoning across BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack, and NetHack; common entry point for LLM/VLM agents | Aggregate progress does not identify the cognitive mechanism responsible for success or failure | First external interactive-reasoning reproduction |
| [Procgen](https://github.com/openai/procgen) | Procedural variation, sample efficiency, and generalization to unseen levels across 16 game-like environments | Primarily an RL generalization protocol; not a complete memory / active-experiment / evidence-integrity benchmark | Held-out level and mechanics-generalization track |
| [Crafter](https://arxiv.org/abs/2109.06780) | Long-horizon exploration, survival, resource acquisition, crafting, and semantic achievements | Broad outcomes need controlled interventions for cognitive attribution | Long-horizon exploration subtrack |
| [GameWorld](https://gameworld-project.github.io/) | 34 browser games / 170 tasks across runner, arcade, platformer, puzzle, and simulation genres; state-verifiable success and progress | Visual/browser action interface and real-time control are separate competencies | Later visual and real-time track |
| [MiniHack](https://github.com/facebookresearch/minihack) | Composable NetHack-style tasks with configurable mechanics and controlled difficulty | Needs Symthaea-specific mechanism ablations and causal transfer tests | Selected task source via BALROG or direct pinned adapter |

Procgen was explicitly designed to measure sample efficiency and generalization using diverse procedural levels and separate train/test seeds. It is useful because a policy that memorizes a narrow level distribution should not score well on held-out levels. See Cobbe et al., [ICML 2020](https://proceedings.mlr.press/v119/cobbe20a.html).

Crafter introduced a visual open-world survival game with semantically meaningful achievements spanning exploration and crafting. It provides a compact test of long-horizon behaviour rather than a single narrow task. See Hafner, [Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780).

BALROG's ICLR 2025 paper identifies long-horizon planning, spatial reasoning, exploration, and interaction as recurring weaknesses in language and vision-language agents. Its live leaderboard contains post-paper entries. At the research snapshot for this specification, the public page showed a Claude Opus 5 Max entry at **63.4 ± 1.8 overall progress**, dated **2026-09-20**, with uneven per-environment results (for example, 100.0 BabyAI, 68.2 Crafter, 95.8 Baba Is AI, 37.5 MiniHack, and 7.3 NetHack). This is a dated public leaderboard reference, **not** a locally reproduced baseline or universal SOTA claim. Freeze a dated copy and protocol identity for every comparison. See the [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html) and [live leaderboard](https://balrogai.com/).

GameWorld reports 34 games and 170 tasks, two agent interfaces (computer-use actions and semantic action parsing), and direct state-based verification rather than relying only on a vision-language judge. Its paused protocol and real-time variant must remain separate because the latter includes inference-time interaction constraints. See the [paper](https://arxiv.org/abs/2604.07429), [project page](https://gameworld-project.github.io/), and [repository](https://github.com/gameworld-project/gameworld).

### 1.2 Industry direction and opportunity

The industry is moving toward agents that reason and act inside persistent interactive environments, not only answer questions. Google DeepMind's [SIMA 2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) combines language instruction, reasoning, and interaction across 3D worlds. DeepMind's October 2026 [Genie 4 announcement](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/) pushes interactive, coherent world simulation further.

These are adjacent but different capabilities: a world generator is not a game-playing agent, and a game-playing agent is not necessarily a good world model. The SIMA 2 research announcement itself identifies long-horizon task completion, limited interaction memory, precise low-level control, and robust visual understanding as ongoing challenges. These are useful hypotheses for Symthaea's test design, not proof that any one architecture has solved them.

**Opportunity:** Symthaea should not try to win by imitating the largest multimodal model on every visual task. It should test whether its cognitive core offers a different advantage: reliable state abstraction, explicit memory, causal mechanics discovery, active information gathering, transfer across rules, uncertainty-aware action, and lower inclusive cost for verified decisions.

## 2. What is genuinely new in the proposed programme?

The benchmark should not duplicate BALROG, Procgen, Crafter, or GameWorld. Its novel layer is a set of controlled cognitive experiments wrapped around compatible external environments.

### 2.1 World-model-first, outcome-second

For each episode, separate these questions:

- **State estimation:** What does the agent believe the current state is?
- **Dynamics prediction:** What does it predict will happen after each legal action?
- **Counterfactual ranking:** Which action does it predict will best advance the objective?
- **Closed-loop outcome:** What actually happens after the selected action?
- **Calibration:** Was confidence commensurate with observed correctness?
- **Cost:** What did the complete decision process consume?

A high game score with poor counterfactual predictions may reflect memorized action patterns. Low prediction error with poor task success may indicate a planner, objective, or action-selection failure. Report both.

### 2.2 Epistemic gameplay: play, probe, wait, or abstain

Traditional game agents are often evaluated as if every turn must produce a game action. Symthaea should also be allowed to choose a legal information-gathering action, request a permitted observation, wait for a delayed effect, or abstain when uncertainty is too high.

Examples include inspecting an unexplored tile before entering it, testing whether a switch changes a door before committing to a one-way passage, comparing candidate actions when mechanics are uncertain, and declining a speculative move when the expected cost of error is high.

Measure value-of-information regret against an exact evaluator oracle on small tasks. Reward useful information only when it can improve a later decision; do not reward curiosity that consumes budget without changing a prediction or action.

### 2.3 Mechanics interventions: same appearance, different law

Create paired tasks where the goal, map, and visual appearance remain fixed while one transition rule changes. Examples:

- a key opens one door in world A but toggles all doors in world B;
- a conveyor pushes in one direction in A and reverses in B;
- a resource is renewable in A but exhaustible in B;
- a hazard advances every action in A but every second action in B.

The candidate should predict the difference, choose a discriminating probe where appropriate, and update only the affected rule. This directly tests memorization and overfitting to superficial appearance.

### 2.4 Compositional transfer, not level memorization

Expose the system to primitive skills separately (navigation, key/door, pushing, resource dependency, hazard avoidance). Evaluate unseen combinations and held-out rule families. A transfer claim requires a matched from-scratch baseline and a predeclared cost budget. Measure how many interactions, model updates, and compute units it saves—not merely whether the agent succeeds.

### 2.5 Cognitive causality through controlled ablation

Use matched episodes to compare full Symthaea, HDC representation ablation, temporal model ablation, memory ablation, planner ablation, active-exploration / value-of-information ablation, and simple heuristic or search substitutes.

Where practical, use a small factorial design rather than only one-component-at-a-time ablations. This makes interactions visible (for example, whether temporal prediction helps only when episodic memory is available). Match compute and interaction budgets. Removing a component must not remove evaluator safeguards.

### 2.6 Counterfactual replay and mutation of the evaluator

For each episode, preserve a replayable event stream. Re-run controlled counterfactuals: replace an observation with an aliased or shuffled observation; remove one memory event; alter a single game rule; delay or duplicate an action result; change one irrelevant visual texture while preserving mechanics.

The independent evaluator should also be mutation-tested. It must reject tampered seeds, reordered or duplicated events, altered transition rules, candidate access to hidden state, and scoring that depends on post-outcome labels.

### 2.7 Self-generated challenges, with an independent referee

Symthaea may propose levels, counterexamples, or hidden mechanics that expose its weaknesses. But it must not decide whether those tasks count, choose which generated tasks are scored, see hidden evaluation outcomes, or grade itself. Self-generated challenges are a supplemental discovery channel; sealed externally generated tasks remain the qualification cohort.

## 3. Benchmark architecture and evidence boundary

### 3.1 Three observation tracks

| Track | Observation | What it isolates | Claim ceiling |
|---|---|---|---|
| O1: structured state | Typed, explicit game state | Reasoning, memory, dynamics, planning | Not visual competence |
| O2: partial observation | Limited local view plus declared observation history | Belief-state estimation, memory, active sensing | Not pixel-level perception |
| O3: visual | Rendered frames, with measured image/action path | Visual grounding plus closed-loop reasoning | Only after the vision adapter is validated |

Run paired task instances across tracks where semantics permit. Do not compare O1 results to O3 leaderboard scores as if interfaces were equivalent.

### 3.2 Two timing protocols

- **Paused decision protocol:** the world waits while the agent reasons. This isolates decision quality.
- **Real-time protocol:** the world continues during inference. This tests latency, stale observations, action duration, and closed-loop control.

The two protocols require distinct run IDs and separate score tables. Never combine their results.

### 3.3 Candidate / evaluator separation

The agent receives only the declared observation, legal-action contract, permitted history, and budget state. It must not receive hidden map cells or mechanics, evaluator state or reward decomposition not exposed by the task, task-family or holdout labels, exact oracle plans or optimal action values, future events, random generator state, or test seeds.

The evaluator owns the true state, transition oracle, task acceptance, hidden split, and final outcome. The candidate's claims about state, prediction, confidence, and intent are diagnostic artifacts; the evaluator independently computes correctness from the underlying environment state and replay.

### 3.4 Run identity and receipts

Reuse #9735 for the canonical benchmark run manifest and its inclusive resource boundary. Reuse established sealed-task and evaluator custody work (#9332 / #10756) rather than introducing another generic authority.

The game-specific episode record should bind at least:
- benchmark and task-family version;
- environment, ruleset, and adapter source digests;
- candidate runtime and configuration digests;
- declared observation/action interface and action-duration semantics;
- episode seed commitment and split identity (hidden from the candidate);
- ordered observation/action/outcome event digests;
- interaction, simulator-call, rollout, compute, latency, memory, and energy accounting;
- independent evaluator result and replay status.

Hashes provide integrity commitments, not by themselves an authenticated signature or proof of independent custody. If a required receipt is missing, stale, duplicated, or inconsistent, the result is INCOMPLETE or INVALID, not a pass.

## 4. V0.1 Rust-native fixture suite

The first tranche should be a small, deterministic, standalone research crate, isolated from the production runtime. Target **24 frozen instances across six families** (four difficulty variants per family is a starting design, not a fixed statistical sample size).

| Family | Core mechanics | Primary capability |
|---|---|---|
| F1: navigation | Grid, walls, moving or changing obstacles | Spatial representation and route planning |
| F2: pushing | Sokoban-like crates, target cells, irreversible deadlocks | Multi-step planning and deadlock detection |
| F3: prerequisites | Keys, doors, switches, ordered dependencies | Causal state tracking and planning |
| F4: partial map | Local observations, aliased cells, memory-dependent routes | Belief maintenance and active sensing |
| F5: resource chain | Gather, consume, craft, limited inventory, delayed payoff | Long-horizon planning and resource memory |
| F6: hidden rule | Two or more candidate mechanics, informative and uninformative probes | Hypothesis formation, experiment choice, adaptation |

For each family include trivial positive controls that a correct implementation must solve; nontrivial held-out instances; at least one negative control that defeats a shallow heuristic; explicit maximum action / observation / compute budgets; a deterministic state-transition oracle; exact BFS/A*/dynamic-programming or exhaustive optimum where tractable; a replay fixture and expected evaluator outputs.

The initial suite is a protocol and implementation testbed, not evidence that 24 instances are enough to estimate broad generalization. Statistical claims require a larger sealed cohort, enough independent task families, paired seeds, and cluster-aware uncertainty.

### 4.1 Keep oracle access evaluator-only

For small deterministic fixtures, exact search can calculate reachability and solvability, shortest legal path or minimum action cost, exact deadlock / failure outcome, optimal action value for the declared objective, and optimal value of a candidate observation or intervention.

The candidate cannot query these exact quantities. They are diagnostic upper bounds and scoring references. Do not compare the exact oracle's cost or performance as if it were a learned agent.

### 4.2 Initial complexity progression

1. Full map, static mechanics, one goal.
2. Obstacles and multiple goal constraints.
3. Irreversible actions and delayed consequences.
4. Hidden cells and aliased observations.
5. Unknown mechanics and active probes.
6. Rule changes after initial learning.
7. Held-out compositions of familiar skills with unfamiliar layouts.

This progression allows failure diagnosis without conflating every difficulty dimension at once.

## 5. Baselines and fair comparison

Every comparison must state the task version, observation track, action interface, interaction budget, training data, prior exposure, model/runtime, and inclusive resource budget.

| Baseline | Role | Comparison rule |
|---|---|---|
| Random legal policy | Lower-bound sanity check | Same legal-action and budget contract |
| Reactive heuristic | Simple competent baseline | Handwritten rules disclosed and fixed before test |
| BFS/A*/Dijkstra/exact DP | Optimal small-world diagnostic | Evaluator-only; never candidate-accessible |
| Belief-state planner / POMCP | Partial-observation planning baseline | Same observation and interaction budget |
| Small tabular RL / PPO where suitable | Learning baseline | Training environment count and steps reported |
| HDC-only | Representation contribution | Same task and resource budget |
| Temporal-model-only (CfC/LTC) | Dynamics contribution | Same task and resource budget |
| HDC + temporal model | Coupled representation/dynamics | Same task and resource budget |
| Planner-only / heuristic world model | Planning contribution | Same candidate actions and rollout budget |
| Full Symthaea | Integrated candidate | Exact runtime and configuration digest |
| Public benchmark agents | External reference | Only compare when environment, interface, and protocol match |

Large LLM/VLM agents may be included as reference systems, but must be labeled by pretraining and tool access. Report model family/version, prompts, context budget, retries, tool calls, inference hardware/service, latency, and cost. Do not compare a small unpretrained cognitive core against a proprietary model score without explaining this asymmetry.

### 5.1 External tournament order

1. **BALROG slice:** select one simple and one difficult environment (for example BabyAI plus MiniHack) and pin the official benchmark revision and task configuration. Then expand to Crafter, TextWorld, Baba Is AI, and NetHack.
2. **Procgen:** first reproduce a single environment with its original training/evaluation split, then evaluate a frozen multi-environment policy on held-out level seeds.
3. **GameWorld:** feasibility and license review first; then a small puzzle/platformer slice under the official paused protocol. Only later attempt fine-control and real-time tracks.
4. **Crafter standalone:** add a separate reproduction only if its chosen BALROG integration does not already provide the needed protocol and outcome metrics.

Do not vendor benchmark implementations until licenses and dependencies are checked. Keep upstream Python/browser stacks in pinned adapter environments; keep Symthaea's cognitive core and benchmark-domain logic Rust-first.

## 6. Metrics: no single scalar hides a failure

### 6.1 Primary task metrics

- **Success rate:** completed objective / assigned episodes.
- **Normalized progress:** partial achievement on an explicit, independently computed scale.
- **Action efficiency:** actions or path cost relative to the exact optimum.
- **Budget curve:** success and progress versus interaction budget.
- **Generalization:** performance on held-out level seeds, mechanic families, and unseen combinations.
- **Adaptation cost:** interactions and compute required after a rule change.
- **Transfer gain:** relative improvement in sample/compute efficiency over a matched from-scratch baseline.
- **Retention / interference:** preservation of previously qualified skills after sequential task learning.

### 6.2 Cognitive mechanism diagnostics

- state-estimation accuracy and belief calibration;
- one-step and multi-step transition prediction error;
- counterfactual prediction error under held-out interventions;
- action-ranking regret under matched candidate sets;
- memory retrieval, temporal ordering, and source-of-observation accuracy;
- observation / probe value and regret relative to the exact small-world oracle;
- recovery latency after model surprise or invalid action;
- explanation-to-trace consistency (diagnostic only, never a correctness oracle).

### 6.3 Calibration and risk

Report confidence calibration, Brier score or log score where probabilistic outputs are well-defined, selective risk versus coverage, and reason-specific abstention. A system that always abstains must fail a minimum-coverage requirement; a system that always acts must be scored for avoidable failures. Preserve invalid or missing episodes in the denominator according to the preregistered protocol.

### 6.4 Systems and economic efficiency

Report total wall-clock time and p50/p95/p99 decision latency; CPU/GPU time, peak memory, model queries and simulator calls; candidate action count, rollouts, and actual interaction count; energy as metered, device telemetry, or estimate (never conflate these); retries, failed runs, and incomplete episodes; cost per success and cost per independently verified success.

The primary efficiency question is useful capability per inclusive unit cost, not merely model-only tokens or inference time.

## 7. Adversarial controls and evaluator mutation tests

Required task controls:
- shuffle observations and action/outcome history;
- provide aliased observations whose distinction requires memory;
- change textures or symbols while keeping mechanics invariant;
- change one mechanics rule while keeping appearance and objective invariant;
- include misleading, irrelevant, and no-op observations;
- create irreversible dead ends and delayed consequences;
- inject dropped, duplicated, reordered, and delayed events;
- test post-training level-family and mechanic-family holdouts;
- include both always-act and always-abstain controls.

Required evaluator mutations:
- alter the hidden seed after candidate execution;
- swap environment rule digests;
- duplicate or omit event records;
- reorder events where order is semantically relevant;
- alter candidate-visible fields to leak hidden state;
- tamper with the score or task acceptance receipt;
- replay against a different adapter/runtime digest.

Every mutation must either be detected or explicitly marked outside the current guarantee. Mutation adequacy should be reported as a denominator (detected / tested), not a vague green check.

## 8. Prospective targets and claims ladder

The following are **prospective research targets**, not observed results or published SOTA claims:

| Gate | Prospective target | What it establishes |
|---|---:|---|
| Frozen deterministic controls | 100% exact agreement with evaluator oracle on designated known-answer fixtures | Correctness of the small task and evaluator contract |
| Exact-equivalence mechanics controls | 0 forced unique-mechanic claims when evidence is non-identifying | Proper uncertainty handling on the declared fixture |
| Held-out procedural/compositional cohort | ≥80% success on a pre-registered bounded cohort | Initial generalization target; not general game competence |
| Cross-task transfer | ≥20% lower interaction count or inclusive cost than matched from-scratch baseline, with no worse validity | Reusable skill benefit on the tested cohort |
| Sequential retention | ≥90% retention of previously qualified task skills | Bounded resistance to interference |
| Efficiency | Full quality / cost / latency Pareto frontier | Whether gains are economically meaningful |

Before claiming an external SOTA result: reproduce the selected benchmark protocol and baseline locally; freeze exact environment and evaluator revisions; match observation and action interfaces; match or disclose training exposure and compute; use paired seeds and multiple independent task clusters; publish uncertainty, all failed/incomplete episodes, and raw receipts; obtain independent replay.

For BALROG, the dated 63.4 ± 1.8 page entry is a reference point only. It is not a target to beat with a different model or interface. Establish the current official leaderboard snapshot on the exact benchmark revision first. For GameWorld, distinguish success rate from progress, computer-use from semantic-action interfaces, and paused from real-time execution. For Procgen, report the original train/test protocol and environment-specific scores rather than a cherry-picked mean.

## 9. Execution plan

### M0 — specification and baseline archaeology (this document)

- Freeze task taxonomy, claim boundaries, interfaces, and the staged external portfolio.
- Confirm the exact existing owner of generic manifests, sealing, world-model evaluation, and statistical reporting.
- Capture dated external benchmark references.

### M1 — standalone Rust-native core

- Implement deterministic game state and transition rules for the six fixture families.
- Implement evaluator-only exact search for the subset that is tractable.
- Add deterministic seed/replay behavior, typed legal-action checks, and event-log verification.
- Keep the crate outside the production runtime until it passes independent tests.

### M2 — benchmark receipts and mutation tests

- Bind game-specific task and episode records to the existing #9735 run manifest.
- Reuse existing sealed-task and evaluator custody.
- Add candidate/evaluator separation tests, replay determinism checks, seed substitution tests, and evaluator mutations.
- Never report a synthetic fixture gate as agent capability.

### M3 — Symthaea adapters and matched component baselines

- Start with a structured-state adapter; O1 is the least ambiguous integration.
- Add partial-observation adapter only after hidden-state leakage tests pass.
- Add HDC-only, temporal-only, planner-only, and full-agent configurations.
- Add matched compute and interaction budgets, then run factorial ablations.

### M4 — external environment reproduction

- BALROG first, then Procgen, then GameWorld.
- Pin licenses, versions, action contracts, and evaluator revisions.
- Keep paused and real-time game operation separate.
- Visual track is blocked until the actual image pathway and its complete cost are validated.

### M5 — transfer and research publication

- Freeze held-out mechanics and task-composition cohort.
- Compare reuse against from-scratch learning.
- Test whether internal capability gains transfer to held-out Symtropy/Sol-Atlas tasks; treat any such transfer as a new experiment, not an assumed consequence of game success.
- Publish scorecard, failures, ablations, resource receipts, and independent replay results.

## 10. Falsification criteria

The programme should be willing to conclude that Symthaea's current architecture does not offer a game-agent advantage.

- If HDC / temporal / memory ablations do not change relevant outcomes under matched budgets, do not attribute capability to those modules.
- If cross-task reuse does not beat from-scratch learning, report no transfer.
- If a mechanics change causes the same policy to fail and the system does not detect surprise, the causal world-model hypothesis is weakened.
- If performance appears only under privileged state, do not claim perception.
- If structured-state performance is high but visual performance is weak, report the modality gap.
- If score gains come from additional interactions, larger budgets, or evaluator leakage, they are not capability improvements.
- If benchmark improvement does not survive held-out task families and independent replay, do not claim robust generalization.

## 11. Ownership and non-duplication map

| Existing owner | Reuse boundary |
|---|---|
| #9735 | Canonical run manifest and resource accounting |
| #9332 / #10756 | Sealed generation and independent evaluator research |
| #9094 and its CMAQ/SAM dependencies | Prediction-to-action world-model comparison |
| #13691 | Active experiment design, identifiability, and evidence-aware update concepts |
| #13018; #3054 / #3095 | Legacy / institutional game-theory qualification, not game-playing runtime |
| Existing psych-bench | Component-level cognitive measures; not an end-to-end interactive game score |

This programme must not create another generic benchmark manifest, task-sealing authority, score aggregator, or world-model decision auditor.

## 12. References

1. Paglieri et al. (ICLR 2025), [BALROG: Benchmarking Agentic LLM and VLM Reasoning on Games](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html); [leaderboard](https://balrogai.com/).
2. Cobbe et al. (ICML 2020), [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://proceedings.mlr.press/v119/cobbe20a.html); [official Procgen repository](https://github.com/openai/procgen).
3. Hafner (2021), [Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780) (Crafter).
4. Ouyang et al. (2026), [GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents](https://arxiv.org/abs/2604.07429); [project page](https://gameworld-project.github.io/); [repository](https://github.com/gameworld-project/gameworld).
5. Google DeepMind (2025), [SIMA 2: An Agent that Plays, Reasons, and Learns With You in Virtual 3D Worlds](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/).
6. Google DeepMind (2026-10-27), [Genie 4: A New Generation of World Simulation](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/).
7. Facebook Research, [MiniHack repository](https://github.com/facebookresearch/minihack).

---

**Qualification statement:** This file is a research specification. It does not contain a completed Rust implementation, executed task campaign, local external-benchmark reproduction, CI pass, or Symthaea performance result.

## 13. Updated industry snapshot — 28 November 2026

This addendum supersedes the earlier BALROG example in §1.1 where it described Claude Opus 5 Max as the top visible result. A fresh read of the **official BALROG leaderboard** now shows:

| BALROG LLM entry | Overall progress | BabyAI | Crafter | TextWorld | Baba Is AI | MiniHack | NetHack | Result date |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GPT-6-Astra-Max** | **68.3 ± 2.0** | 100.0 | 76.8 | 54.5 | 100.0 | 65.0 | 13.2 | 2026-09-18 |
| Claude-Opus-5-Max | 63.4 ± 1.8 | 100.0 | 68.2 | 71.6 | 95.8 | 37.5 | 7.3 | 2026-09-20 |
| GPT-5.6-Terra-Max | 53.2 ± 2.0 | 98.0 | 55.0 | 27.8 | 97.5 | 37.5 | 3.3 | 2026-09-20 |

Source: official [BALROG leaderboard](https://balrogai.com/), read 2026-11-28. The environment-level pattern matters: the overall leader still scores 13.2 on NetHack, while achieving 100 on BabyAI and Baba Is AI. Do not convert progress into success probability or claim broad game competence from the aggregate.

A third-party result page is unnecessary for this snapshot: GPT-6-Astra-Max appears directly on the official leaderboard. The dated leaderboard remains a snapshot, not a locally reproduced Symthaea baseline.

### 13.1 GameWorld: keep standard and fine-control protocols separate

The GameWorld project page currently describes 34 browser games and 170 tasks, with five genres and direct state-derived success/progress metrics. Its visible generalist podium lists Gemini-3-Flash-Preview at **41.9% progress / 21.2% success**, GPT-5.2 at **40.6% / 20.6%**, and Claude-Sonnet-4.6 at **39.3% / 20.6%**. The visible novice-human progress baseline is **64.1%**, and expert-human progress is **82.6%**.

A November 27, 2026 Benchmark Almanac read of the **GameWorld official leaderboard** reports a newer **Fine-Control Protocol, 400-step** result: GPT-6.1 Terra at **76.8%**, Gemini 3.8 Flash at **63.5%**, and Qwen3.8-Flash-Next at **61.5%**. The new score is protocol-specific; it must not be compared as though it were the same run as the standard-protocol public podium. The Almanac is a secondary snapshot of the official leaderboard, so preserve its provenance as such until a raw official result capsule is downloaded and independently inspected.

Sources:
- GameWorld project results and human baselines: https://gameworld-project.github.io/
- Benchmark Almanac, retrieved 2026-11-27: https://www.benchmarkalmanac.com/benchmarks/gameworld/

The programme should record at least four distinct columns when reproducing GameWorld:
1. standard paused protocol progress and success;
2. fine-control paused protocol progress and success;
3. GameWorld-RT progress and success (real-time, separate);
4. human novice/expert reference, with the exact interface and action budget.

### 13.2 SIMA 2: compare training-environment and held-out-environment capability

Google DeepMind's SIMA 2 technical report (2025-12-05) reports training-environment average success of **65% human-evaluated / 66% automatically evaluated**, compared with human **76% / 78%** under the same timeout and human-no-timeout **86%**. However, its entirely held-out environment success is much lower: roughly **14% on ASKA** and **13% on MineDojo** (versus approximately 2.5% and 0% for SIMA 1). It completed tasks in 26 of 50 MineDojo task categories.

This is an important methodological lesson: report familiar-environment skill, held-out task generalization, and truly held-out environment generalization separately. A system can approach the human baseline on familiar environments while retaining a large gap under more severe distribution shift.

Source: SIMA 2 technical report, https://arxiv.org/abs/2512.04797 ; downloadable report: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf

### 13.3 Revised SOTA strategy

Do not set a single cross-benchmark target. Use protocol-matched goals:

| Track | Current external reference | Symthaea target sequence |
|---|---|---|
| BALROG LLM | GPT-6-Astra-Max 68.3 ± 2.0 overall progress | First reproduce a small, exact BALROG slice; then aim to exceed 68.3 only under the same benchmark release, interface, model/tool budget, and episode protocol |
| GameWorld standard | Gemini-3-Flash-Preview 41.9 progress / 21.2 success on visible podium | First reproduce one puzzle and one platformer family; report progress and success separately |
| GameWorld fine-control | GPT-6.1 Terra 76.8 progress, 400-step Fine-Control Protocol (secondary snapshot of official board) | Protocol-match the exact fine-control action rate and 400-step boundary before claiming a challenge; target >76.8 only after official raw result and local baseline are verified |
| SIMA 2 held-out environments | About 14% ASKA / 13% MineDojo | Build a genuinely unseen mechanics/world split and compare success, adaptation cost, and skill transfer; do not compare unlike games directly |
| Rust-native GameBench | Exact oracle exists only for small deterministic fixtures | 100% on frozen known-answer controls; then test held-out maps, rules and compositions with statistical uncertainty |

For a GameWorld or BALROG SOTA claim, Symthaea will eventually require a real visual/action route. The current F1–F6 structured environment fixtures do not qualify pixel perception, computer-use, browser interaction, or multimodal performance.

## 14. New research direction: skill contracts and cognitive control

The strongest next paradigm shift is not another monolithic policy. It is a **composable skill system with independently verifiable contracts**.

A reusable skill should have:
- a typed precondition (what must be true before execution);
- an action policy or planner;
- a postcondition (what the skill guarantees when it reports success);
- invariants (what it must never violate);
- an uncertainty / failure condition;
- an interaction and compute budget;
- a trace proving what actions occurred.

For example, a `collect_key` skill would not report success merely because the agent moved near a key. It would report success only when the evaluator verifies the inventory transition. An `unlock_door` skill would require possession of the matching key and return a proof that the correct door state changed.

Compose these contracts into a hierarchical task graph. Measure:
1. primitive skill success;
2. composition success on unseen task graphs;
3. recovery when a skill postcondition fails;
4. transfer savings versus matched from-scratch planning;
5. invalid or falsely reported skill successes.

This creates a possible bridge to robotics and Symtropy: the same skill contract can later describe a physical action, with different evidence and safety requirements. Game success alone never qualifies a physical capability.

### 14.1 Surprise-triggered model revision

Add a second explicit mechanism test. Before an action, the candidate predicts a distribution over the next state. After the evaluator returns the observed state, compute an independently defined prediction residual / surprise event. The agent may propose a local model revision, but must:
- identify which transition rule changed;
- distinguish sensor noise from a mechanics change;
- update only the affected skill/model component;
- demonstrate improved held-out prediction and action quality;
- preserve unaffected skills.

Report false alarms, missed changes, detection delay, model-revision cost, post-change regret, and forgetting. This is more informative than reporting only the final score after a game changes its rules.

### 14.2 Active sensing under a decision budget

The F4 scan fixture creates the first explicit sensing-cost experiment. Extend this across tasks by comparing:
- no scan / passive observation;
- fixed scan schedule;
- matched-cost random scan;
- greedy entropy reduction;
- decision-value / EVSI probe choice;
- exact evaluator-only probe oracle on small worlds.

A scan that reduces uncertainty but cannot change any optimal decision should not be called useful exploration. The primary quantity should be decision regret avoided per sensing cost, accompanied by coverage and failure severity.

### 14.3 Self-improvement without self-certification

Permit Symthaea to propose new levels, skill candidates and adversarial counterexamples. Freeze generated tasks before the candidate sees evaluation outcomes. Have an independent generator/evaluator select the final hidden cohort. Score the improvement *process* on new tasks created after the system's proposal policy is frozen. This composes with the existing sealed-evaluator work; it does not grant the agent authority over its own score.


## 15. Implementation checkpoint — 28 November 2026

The standalone Rust research crate now contains deterministic environment fixtures for all six initial task families:

| Family | Implemented environment contract | Important controls |
|---|---|---|
| F1 Navigation | Rectangular grid, bounded moves, full and local observation, ordered trace/replay, private exact shortest-path oracle | Invalid geometry, blocked movement, action limit, tampered transition and non-contiguous event sequence |
| F2 Pushing | Single-box Sokoban-like movement and push mechanics, private exact reachability oracle | Blocked push, wall-trapped box, duplicate walls, overlapping objects |
| F3 Prerequisites | Single-use keys, explicit collected-key and current-inventory state, persistent unlocked doors, private exact planner | No-key locked door, no key respawn, duplicate IDs, unknown key reference, key/door overlap |
| F4 Partial map | Current field-of-view observations and paid one-step scan; environment does not retain observation history | Hidden distant cells, scan cost, previous cells absent from later views, blocked action costs |
| F5 Resource chain | Scarce gatherable piles, typed recipes, workbench, acyclic recipe dependency validation, depletion and private exact planner | Zero resource pile, recipe cycle, missing station/input, depleted pile cannot be gathered |
| F6 Hidden mechanics | Two rule variants with identical initial observations; paid lever intervention; route determined by the revealed gate state | No-probe negative control, action budget, single-use switch, hidden rule omitted from observations and debug output |

The current branch contains **41 authored Rust unit tests** across F1–F6. Test count is a source inventory only: it is not a test pass. A dedicated GitHub Actions workflow is configured for formatting, tests, and strict Clippy. No local Rust toolchain is available in the current work environment, and no exact-head CI result is claimed until that workflow completes.

### 15.1 Important implementation boundary

These are environment contracts and evaluator diagnostics, not a game-playing policy. The hidden-rule constructor receives the evaluator-selected mechanism; it does not infer it. The partial-observation environment discards prior views; it does not implement candidate memory. The exact planners are crate-private and are intended only for known-answer / optimality references on small fixtures.

The current milestone is therefore:

- **Implemented:** six deterministic task families and their source-authored controls.
- **Structural:** skill-contract direction, causal surprise / model revision protocol, active-sensing value accounting, sealed external task portfolio.
- **CI-pending:** rustfmt, cargo test, strict Clippy on the exact PR head.
- **Not implemented:** Symthaea's actual cognitive-policy adapter, candidate-side episodic map memory, skill acquisition/transfer, signed evaluator receipts, procedural hidden cohorts, and public BALROG/Procgen/Crafter/GameWorld reproductions.
- **Not qualified:** autonomous discovery, visual game play, SOTA performance, or transfer to physical systems.

### 15.2 Immediate next tranche

The next productive step is not another environment family. It is the candidate interface and cognitive instrumentation:

1. Define an observation/action-only candidate trait in a separate adapter boundary. Do not provide Game, Spec, hidden rule, remaining pile vector, or exact planner to the candidate.
2. Add a candidate episode loop which receives only the current observation and emits one legal action plus a typed prediction/uncertainty record.
3. Add an independently replayed outcome trace for F1–F6; keep evaluator truth and task scoring outside candidate memory.
4. Add matched candidate variants: random legal, deterministic heuristic, BFS only as evaluator diagnostic, HDC-only, temporal-only, memory-only, planner-only, and coupled variants where available.
5. For F4, score memory retention by requiring the candidate to navigate from accumulated observations after earlier cells disappear from the current view.
6. For F6, compare no-probe, fixed probe, random probe, entropy-greedy, decision-value probe, and exact evaluator-only oracle under equal interaction budgets.
7. Only after these controls pass should an external benchmark adapter or performance campaign be treated as a capability result.
