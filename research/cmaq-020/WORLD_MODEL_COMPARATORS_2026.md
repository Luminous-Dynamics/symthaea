# World-model and Physical-AI comparator map (26 November 2026)

This file separates current industry/research systems by the claims their evaluations support. It is a comparison plan, not a report of Symthaea performance.

## 1. Comparator families

| Family | Representative system | What it demonstrates / measures | How CMAQ-020 should compare |
|---|---|---|---|
| Video/world simulation | NVIDIA Cosmos Predict 2.5 | Multimodal video prediction; action-conditioned robotics model variants; PAI-Bench domain and visual quality scores | Use as a visual/temporal prediction reference where observation modalities and prompts match. Do not infer causal state sufficiency or control utility from video quality. |
| World-action models | DreamZero, Cosmos Policy, LingBot-VA, Fast-WAM | Video and action prediction jointly or through inverse/representation-only action pathways | Same robot/task, same action interface, same controller budget; report action fidelity, executed success and false-success under failure separately. |
| Multi-embodiment latent world models | RoboJEPA (8B) | JEPA latent prediction over 12 robot embodiments; reports compute scaling of imagination error and downstream planning | Include as a high-capacity latent-prediction arm on compatible robot tasks; compare scaling and planning, not raw latent loss against HDC similarity. |
| Contact-rich visual/tactile models | ContactWorld | Twelve manipulation tasks, six sensing modalities, planning and real-world contact tasks | Compare representation structure and cross-modal compatibility; point cloud / tactile force-field arms must be matched to the same task and planner. |
| Failure fidelity / diagnosis | WorldFailBench | Failure reason, execution stage and success↔failure boundary under scene changes; LIBERO, RoboTwin and physical G1 manipulation | Score false-success separately from reason/stage fidelity and two-sided boundary localization; preserve matched initial state, realized action and environment cluster. |
| Failure-conditioned world-model training | AMD Learning to Fail / BWM on RoboTwin adjust_bottle | Tests whether failure rollouts repair fake success and whether in-world RL survives replay in the simulator | Reproduce only on the same task/model/policy interface; include success-only control, multiple seeds, shared-start action forks and closed-loop replay mismatch. |
| Recovery after failure | LIBERO-RECOVER | Retry, action adaptation, object-state recovery and environmental recovery across 1,000+ scenarios | Evaluate recovery as downstream policy utility, separately from whether the world model predicted the failure correctly. |
| Action-conditioned reliability | MiraBench | Physics adherence, action-following fidelity and optimism-bias/failure calibration, with human judgments | Add failure-inducing actions, missed contacts, partial execution and false-success rate to our action-response panel. |
| Executable consequence evaluation | ScratchWorld | Replay-verified changed fields and values, causal traces and counterfactual outcomes in executable programs | Include no-change/copy controls and value-aware changed-field correctness so persistence overlap cannot masquerade as consequence prediction. |
| Closed-loop world-model utility | World-In-World, WorldArena 2.0 | Interactive task success, model use as an environment, and embodied/visuo-tactile evaluation | Treat as later closed-loop tracks after simulator, action interface and custody are reproduced. |
| Model-based control baseline | TD-MPC2 / RSSM-style recurrent state-space models | Strong model-based planning/control in continuous-control settings | Reproduce only on matched continuous-control tasks, with identical action and compute budgets. |
| Direct policy control | VLA / direct policy baseline, e.g. Cosmos Policy or compatible robot policies | Maps observations directly to actions, without the same explicit state-rollout interface | Mandatory same-policy/no-world-model comparison: isolate the causal contribution of adding a world model. |
| Planner exploitation diagnostics | MRS.Q | Demonstrates that search/optimization can exploit systematic model error | Increase search budget, include controlled overestimation and measure selected-action error and realized regret. |

## 2. Useful published reference points (not cross-benchmark rankings)

- **Cosmos Policy (ICLR 2026 / NVIDIA Cosmos Lab):** reports 98.5% average success on LIBERO and 67.1% on RoboCasa in its stated evaluation setup. This is a strong embodied-policy reference, not a directly comparable number for a synthetic CMAQ evaluator. https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/
- **DreamZero:** NVIDIA's World-Action Models overview reports 1,750 Elo on the April 2026 RoboArena leaderboard versus 1,622 for Pi-0.5 in that snapshot. Elo depends on that particular arena's task mix and evaluator. https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
- **RoboJEPA:** the October 2026 paper reports an 8B-parameter multi-embodiment JEPA trained across 12 robotic embodiments, compute scaling of imagination error, and zero-shot long-horizon planning. Reproduce its public checkpoint/protocol where hardware and task interfaces permit. https://arxiv.org/abs/2610.10515
- **ContactWorld:** reports 12 contact-rich tasks and six visual/tactile modalities. Its paper reports mean planning success of 32.1% for point cloud and 36.1% for PointCloud+TacFF in its own testbed, compared with 20.7% wrist RGB and 22.0% front RGB. These are ContactWorld-specific measurements. https://contact-world.github.io/
- **MiraBench:** evaluates 12 configurations and more than 16,000 human judgments around physical consistency, action-following and optimism bias. https://arxiv.org/abs/2605.29360
- **ScratchWorld:** uses replay-verified executable worlds and emphasizes value-aware changed-field scoring. It is a useful guard against copy-the-input and no-change baselines. https://arxiv.org/abs/2606.31689
- **World-model benchmark survey:** its November 2026 corpus covers 130 benchmarks and separates target, protocol, metric and data. It records 98 open-loop tracks and 13 closed-loop tracks; 99 use prediction-level metrics and 14 use downstream outcome metrics, with overlaps across categories. https://world-model-benchmarks.github.io/World-Model-Benchmarks/

## 3. Failure fidelity and recovery

- **WorldFailBench (October 2026):** evaluates observable failure reasons, execution stages and success/failure boundaries in LIBERO, RoboTwin and physical G1 manipulation; seven tested world models show substantial failure-fidelity gaps. This is a failure-diagnostic benchmark, not just a task-success leaderboard. https://arxiv.org/abs/2610.33565
- **AMD Learning to Fail (20 October 2026):** on RoboTwin 2.0 adjust_bottle, BWM's base checkpoint rendered 82.5% of simulator failures as successes; failure-rollout fine-tuning reduced the reported held-out fake-success rate to 26.3%. This is one task/model/policy setup; the paper also finds same-start action-fork outcomes remain difficult. https://www.amd.com/en/developer/resources/technical-articles/2026/rl-inside-world-models.html
- **LIBERO-RECOVER (September 2026):** separates retry, action adaptation, object-state recovery and environmental recovery across 1,000+ failure-recovery scenarios. It measures a downstream policy capability and must not be confused with failure-prediction fidelity. https://arxiv.org/abs/2609.05178

Use [FAILURE_FIDELITY_V0_1.md](FAILURE_FIDELITY_V0_1.md) for the executable scorecard. It separates false success, false failure, reason accuracy, stage accuracy, boundary-flip detection and exact two-sided boundary outcome accuracy. It requires evaluator-owned reference outcomes and cluster-linked replay evidence.

These results show why a world model can have a plausible final success rate while still misleading an RL policy or a deployment reviewer: it may invent a lift after missed contact, place a failure at the wrong execution stage, or move the success/failure boundary under the wrong scene perturbation. The current NixWard transition candidate's typed episode path is intended to preserve realized execution outcomes and failure dispositions; it has not yet been evaluated on any of these external benchmarks.

## 4. Matched experiment protocol

Every claim of superiority must pin and report:

1. **Task/data:** official split, dataset revision, source commit, file inventory, checksum, license and environment versions.
2. **Model:** exact checkpoint digest, architecture/size, tokenizer/encoder, prompts, adaptation or training data and frozen/adaptive status.
3. **Interface:** observation modality, action semantics, sampling frequency, rollout horizon, reset rules and success definition.
4. **Budget:** simulator interactions, real robot trials, model calls, search expansions, wall time, p50/p95/p99 latency, peak memory, CPU/GPU and measured energy.
5. **Policy:** same downstream policy with and without the world model; direct policy/VLA; no-model, copy/no-change and oracle-only controls.
6. **Statistics:** independent environment/mechanism clusters; paired cluster-level uncertainty; all failures, abstentions, timeouts and missing cells; simultaneous intervals for confirmatory families.
7. **Authority/evidence:** model predictions and counterfactuals remain hypotheses. A separate policy/safety/authority layer decides whether an action may execute; only an independently observed execution result is promoted to outcome evidence. Unknown execution outcomes require reconciliation before any follow-on mutation.

Do not pool LIBERO success, RoboCasa success, RoboArena Elo, ContactWorld planning rates, MiraBench reliability and CMAQ response-distortion metrics into one score.

## 5. Symthaea-specific architecture tournament

Start with an identical source history and action budget. Candidate arms:

- Full history/full-state reference (charge all state, replay and checkpoint costs).
- Per-action HDC mean-delta baseline (the currently implemented NixWard `NixWorldModel` transition baseline; it must not be described as CfC).
- Context-conditioned symbolic/SCM reference.
- Linear/low-rank state projection.
- Matched RSSM/recurrent latent state.
- HDC-only.
- CfC/LTC-only, when an actual transition adapter is wired and qualified.
- HDC-LTC / HDC+CfC factorized state, as a distinct arm.
- Retrieval-backed event/state references with provenance resolution.

For each arm, preserve the same registered `(U,Q,T,I,D)` contract. Measure response fidelity, false merges, false splits, invariant violations, selective uncertainty, causal-effect contrast, false-success optimism, closed-loop task success, recovery, all-in storage and inference cost.

## 6. What would count as beating the field?

A credible Symthaea advantage should be a preregistered Pareto improvement, not a cherry-picked score:

- Equal or better held-out executed task utility at materially lower state/checkpoint/memory and inference cost; or
- Better weak-cell performance, uncertainty calibration and recovery at the same all-in cost; or
- Better causal/intervention response on held-out mechanisms while retaining a significantly smaller and more auditable state.

Proposed stretch gate for the state-abstraction lane: at least 25% lower all-in state cost than full-history at a held-out operating point, with the upper paired cluster-level bound on additional decision regret within a pre-registered one-percentage-point margin; zero false merges on deterministic witnesses; no invariant violations. This is a target, not an achieved result.

A simple baseline is allowed to win. If HDC/CfC does not outperform the per-action delta or symbolic/recurrent baselines under matched cost, the correct result is that the more complex representation has not earned its complexity.

## 7. Current Symthaea claim boundary

CMAQ-020's Rust evaluator currently measures a finite response-grid abstraction and selects cost-aware counterexample witnesses. It does not yet run HDC, CfC/LTC, a robot policy, ScratchWorld, MiraBench, ContactWorld, Cosmos Policy or RoboJEPA. The HDC-delta NixWard model is only a source-inspected candidate baseline. External benchmark reproduction and source-data admission remain separate work.
