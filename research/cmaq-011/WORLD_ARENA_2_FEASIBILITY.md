# CMAQ-011: External Manipulation Benchmark Feasibility — LIBERO, RoboWM-Bench and WorldArena 2.0

**Status:** source-audited feasibility note; no adapter or benchmark result yet  
**Related:** [Issue #9094](https://github.com/Luminous-Dynamics/symthaea/issues/9094), [protocol](PROTOCOL.md), [matched predictive-advantage design](PREDICTIVE_ADVANTAGE.md)

## Decision

Prioritize the first manipulation adapter as:

1. **LIBERO-Spatial** (10 tasks) for geometric/spatial reasoning.
2. **LIBERO-Long** (10 tasks) for multi-step dependencies and state persistence.
3. **RoboWM-Bench** for video-to-action executable manipulation evaluation and reconstructed-simulator outcome checks.
4. **WorldArena 2.0 / RoboTwin 2.0** for visuo-tactile contact, online policy improvement and eventual sim-to-real.
5. **World-In-World** as a separate visual closed-loop track for active recognition, embodied QA, image-goal navigation and manipulation.

Do not start with a physical ALOHA run. First establish a reproducible simulator adapter, paired D0/D1/W1/W2/W3 comparisons, and a robot-specific safety case.

## Source audit (30 October 2026)

| Source | What it establishes | What remains unverified |
|---|---|---|
| [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/libero) | Five suites, 130 tasks total; Linux/MuJoCo requirement; installation via `.[libero]`; EGL headless rendering instructions | Exact pinned task package, physics version, wrappers, observation/action adapters and all dependency hashes for our run |
| [WorldArena 2.0 project page](https://world-arena.ai/) | Published protocol spans vision → visuo-tactile, offline → online RL, simulator → real robot; names RoboTwin 2.0, LIBERO and AgileX split-type ALOHA | Whether all 2.0 implementation code, data, evaluator and licenses are released as a coherent version |
| [WorldArena GitHub](https://github.com/tsinghua-fib-lab/WorldArena) | Public repository with `video_quality` and `embodied_task` source trees | Latest inspected main commit is `2da2ae253b8637ba9de3afc7bea4e087f778ee4d` (2026-07-22); GitHub API returns no detected license. This is not enough to infer WorldArena 2.0's full code/data licensing or release correspondence |
| [World-In-World ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html) | Closed-loop task platform with standardized action API and four visual/embodied task families | Confirm current upstream environment lockfile and runnable adapter before a local reproduction |
| [MRS.Q repository](https://github.com/facebookresearch/MRSQ) | README shows Gym-HalfCheetah-v4 and Dmc-quadruped-walk invocation; README states CC BY-NC 4.0 | Need pin task source/release, dependency lock, baseline configs and data/seed protocol; GitHub API reports SPDX `NOASSERTION`, so confirm license text directly |

WorldArena 2.0's published site illustrates why capability-level reporting matters. Its displayed task table contains strongly asymmetric per-task success (for example, one entry has 100% on Insert HDMI and 0% on Lift Bottle). Such a profile must not be compressed into a single average without preserving per-task rows and confidence intervals. These website figures are project-reported references, not independent reproductions.

## First adapter: LIBERO-Spatial and LIBERO-Long

### Observation boundary

Only provide the policy/model with observations available to the deployed agent. The adapter must explicitly separate:

- camera images and any observation-derived features;
- robot joint/gripper state actually exposed by the task;
- language instruction;
- action-space specification and scaling;
- reward/termination used by the environment.

Privileged object poses, simulator contacts, hidden task state and future environment transitions are evaluator-only. They may be used to compute the oracle W3 arm and counterfactual ground truth, never passed into D0/D1/W1/W2.

### Paired action protocol

For every state/seed block, freeze and hash one candidate action-sequence set. Execute the same sequences under:

- D0 frozen direct controller;
- D1 compute/latency-matched direct controller refinement;
- W1 Symthaea action-conditioned prediction/planner;
- W2 shuffled futures across candidate sequences, stratified by task/state;
- W3 exact simulator dynamics for the same candidate set/horizon.

Keep reward, reset distribution, action horizon, replanning cadence, query budget and evaluation seeds identical. Measure the four paired estimands from `PREDICTIVE_ADVANTAGE.md`: prediction advantage, deployment gain, prediction mediation and oracle gap.

### Capability slices

| Slice | Initial tasks | Primary question |
|---|---|---|
| Spatial relation | LIBERO-Spatial | Does prediction improve action ranking when object geometry/relation matters? |
| Long-horizon | LIBERO-Long | Does prediction reduce irreversible mistakes across multi-step tasks? |
| Occlusion/hidden state | Select tasks only after confirming occlusion in the policy-visible observations | Does the model help infer unobserved state rather than leak simulator state? |
| Contact / tactile | WorldArena 2.0 / RoboTwin after versioned source validation | Does visuo-tactile prediction improve contact timing, slip handling and manipulation success? |
| Sim-to-real | AgileX ALOHA after simulator-stage success | Does the measured predictive advantage survive embodiment transfer? |

Do not claim a capability is tested unless at least one task actually exercises it and the observation/action adapter exposes the corresponding signal correctly.

## Engineering boundary

- Keep orchestration, manifest validation, statistics, report generation and provenance receipts Rust-first.
- Use the benchmark's external Python/MuJoCo runner only as an isolated adapter process if unavoidable; do not make it a Symthaea production dependency.
- Pin the upstream commit, package versions, physics engine, model checkpoints, task config, wrappers, camera transforms, action scaling and seed manifest.
- Record wall-clock p50/p95, peak memory, model queries, simulator queries, inference FLOPs where measurable, and any GPU utilization.
- Preserve all crashes, timeouts, invalid actions, environment resets and safety interventions in the denominator and raw ledger.
- If the exact upstream version/license cannot be confirmed, stop at a feasibility note; do not vendor code or claim reproduction.

## Acceptance gates

1. **Adapter integrity:** smoke tests validate observation/action shapes, reset determinism, reward/termination and no privileged-state leakage.
2. **Policy parity:** direct and model-assisted arms execute the same action proposal set and controller checkpoint.
3. **Causal test:** W1 > D1 with paired confidence interval excluding zero on at least one preregistered capability, while W1 > W2 establishes action/future correspondence matters.
4. **Robustness:** report held-out task variants, contact/occlusion/shift stress where applicable, and CVaR95/failure non-inferiority.
5. **Independent reproducibility:** a second run from pinned artifacts reproduces the aggregate within preregistered tolerance.

Until those gates pass, this remains an adapter plan, not a result or SOTA claim.
