# Real-time cognition: charge planning for a world that keeps moving

Owner: [ROB-REALTIME-COGNITION-001 / #11561](https://github.com/Luminous-Dynamics/symthaea/issues/11561).

Composes with #11475, #11496, #9010, #11122, #11338 and Symtropy #4119/#4382/#4549/#4555/#4588. The issue owns evaluation semantics; it does not add another planner, world model, controller, scheduler, energy ledger, actuator or safety authority.

## Core paradigm

A benchmark that pauses the world while the agent reasons is an upper bound, not a real-time robot test.

While the controller observes, infers, simulates, chooses a plan and projects an action, the physical environment may move; observations age; actuators continue tracking their last command; thermal and battery state evolve; and action/certificate validity can expire.

Therefore compute is not just a cost measured after a task. It is part of the task's physical transition.

For decision candidate i, report separately:
- predicted decision benefit and its calibration;
- wall-clock time from observation cutoff through action application;
- compute electrical energy and temperature impact, where measured;
- sensing and communication costs;
- physical state change during the delay;
- proposal expiry/staleness and independent action admission;
- resulting useful service, safety and recoverability.

Do not convert FLOPs, tokens, model size or operation counts into physical joules without an identified, calibrated hardware model. Do not equate simulation clock with host wall-clock or actuator/device time.

## Environment timing modes

1. Frozen-world evaluator oracle: pause the world during compute; upper bound only.
2. Wall-clock real-time: advance environment during all inference and decision stages.
3. Deadline-limited: each task/action has a frozen deadline and expiry semantics.
4. Compute contention/throttling: declare CPU/GPU contention, power/thermal throttling and scheduling load.
5. Stale observation: delay/reorder samples while the physical world advances.
6. Anytime planning: permit multiple proposals, each independently validated at its actual application time; later plans do not retroactively replace prior actions.

## Baseline matrix

Compare deterministic reflex, direct learned policy, fixed-rate sensing/fixed-depth planning, Bayesian/set-membership + EVSI, corrected action-conditioned FEP + EVSI, thermal-aware MPC/HQP, robust min–max MPC/viability filter, dual control / simultaneous system identification and MPC, and anytime/interruptible planning.

GenAIF diffusion/CVAE/flow-matching and HDC/CfC/LTC ablations must use the same task, observation, action, hardware/plant and wall-clock/compute budgets. Keep manipulation, locomotion, aerial manipulation and laboratory tasks in distinct strata.

## External benchmark context

- [RoboDyna](https://robodyna.github.io/robodyna/), NeurIPS 2026 workshop: physically realistic dynamic manipulation where freezing the scene during inference hides the effect of computation time.
- [SafeLab](https://proceedings.mlr.press/v306/bai26e.html), ICML 2026: 64 laboratory tasks across 9 categories, trajectory-level safety, unsafe-but-successful outcomes and 50 open-loop physical replays.
- [AM-Bench](https://arxiv.org/abs/2609.00641), 2026: modular comparison across policy, low-level control and embodiment for aerial manipulation.
- [Simultaneous system identification and MPC with no dynamic regret](https://doi.org/10.1109/TRO.2025.3576969): online model adaptation during control with simulation and hardware experiments.
- [Long-horizon measured-input system identification on KUKA](https://doi.org/10.1016/j.trpro.2026.04.071): recommends rollout evaluation up to 1000 steps and ID/OOD splits rather than one-step RMSE alone.

These are different task and embodiment classes; they motivate separate strata, not a shared leaderboard.

## Metrics and prospective gates

Report service success and quality; safe-success; regret at action-application time; state prediction error at action time; p50/p95/p99/worst latency by stage; deadline misses; stale actions; RTA rejection; compute/sensing/cooling energy; temperatures; reserve before/after cognition; safe-stop/recovery feasibility; and abstention/Unknown rates.

Targets (not achieved):
- Zero missed deadlines and stale/clock-Unknown action applications in finite known-answer tests.
- p99 decision-to-actuation latency <= 50% of a declared deadline on measured target hardware.
- No hard thermal/voltage/current/safe-stop violation caused by extra reasoning.
- A model/FEP/active-sensing component is retained only if it improves the real-time service/viability Pareto frontier over a matched direct or MPC/HQP baseline after charging elapsed time and measured resource costs.
- Report null results; simpler policies may be better when additional reasoning cannot change the decision.

No benchmark score confers actuator authority. RTA/HAL remains independent.


## Synthetic known-answer corpus

The wall-clock / stale-action extension has ten machine-readable cases:
- [fixtures/realtime_cognition_v0_1.json](./fixtures/realtime_cognition_v0_1.json)
- [fixtures/realtime_cognition_v0_1.schema.json](./fixtures/realtime_cognition_v0_1.schema.json)

The corpus distinguishes frozen-world upper bounds, stale proposals, valid-but-not-yet-authorized proposals, missed deadlines, direct-policy wins after latency costs, anytime-plan expiry, compute thermal throttling, clock uncertainty, no model value-add, and simulation-time-only evidence. It is synthetic and does not itself qualify a runtime clock, hardware deadline or controller.


## New external real-time references (11 November 2026)

The original real-time cognition protocol should be evaluated against three separate external lanes, not one pooled score:

- **RT-SAFE** (October 2026 preprint): navigation in which the world evolves during reasoning. Its matched static/real-time comparison reports 91.3%/94.1% task completion but 19.8%/0.7% safe completion, with 12.3× more collisions in real-time. Its snapshot has exploratory/single-seed and incomplete-run caveats. https://arxiv.org/abs/2610.09294
- **LIBERO-Dynamic**: 40 LIBERO tasks, 1,800 episodes, latency-calibrated inference, physical object motion, synchronous/async/RTC/VLASH arms, and real-robot conditions. Its π0.5 PyTorch results differ across sync hold (43.1%), async queueing (48.4%) and RTC (37.8%) at different best measured action horizons; compare the exact model and horizon strata. https://github.com/renshaojie233/LIBERO-Dynamic
- **RoboDyna**: 30 dynamic manipulation tasks, paired sync/async evaluation, physics-governed object motion and human baselines. https://robodyna.github.io/robodyna/

These public results motivate the action-chunk and temporal coherence contract in [action_chunk_temporal_alignment.md](./action_chunk_temporal_alignment.md).

## Distributed placement and edge/cloud inference

The real-time work now includes [Symthaea #11573](https://github.com/Luminous-Dynamics/symthaea/issues/11573), a matched benchmark for onboard, edge, cloud, hybrid and fleet-shared inference. Microsoft Research's 2026 mobile-manipulation measurement study motivates the comparison: offloading can improve task performance and robot battery life, but latency, bandwidth and remote GPU availability affect the result.

The test charges full capture→encoding→uplink→queue→inference→downlink→local verification/projection→admission latency. It measures robot/edge/cloud/network energy separately and requires local low-level control, RTA and safe-stop to remain functional under remote outage. The 15-case synthetic corpus is linked from the parent README and the issue.


## Strong asynchronous WAM baselines

The real-time benchmark should explicitly compare:
- **Streaming-WAM** (Huang et al., arXiv:2609.28927): action-conditioned world prediction that accounts for the committed prefix while the next chunk is prepared; the project reports 41 ms LIBERO chunk time and 98.35% LIBERO average success in its H100 experiment.
- **AHA-WAM** (arXiv:2606.09811): slow low-frequency world context with a high-frequency action expert; reports 24.17 Hz and 92.80% RoboTwin average success in the authors' setup.
- **StreamingWAM** (Ren et al., arXiv:2610.15585): separate chunk-wise staircase-flow method; reports 57.3 ms real-robot replanning latency.

These are distinct papers and training/robot setups. Compare on the same task/robot/hardware when feasible; otherwise keep results as external reference strata. A Luminous ablation should separate the value of multi-timescale context, action-prefix conditioning and the independent temporal-assurance gate.


## Actuation-side freshness and foreground compute contention

The real-time comparison now includes actuation-side AoI rather than stopping at inference return or message receipt. It also includes EdgePace as a foreground-deadline / unified-memory-contention baseline. See [decision_sensitive_freshness.md](./decision_sensitive_freshness.md) for equations, comparison arms and external hardware references.


## Physical actuation-side AoI reference

The timing measurement owner is [Symtropy #4713](https://github.com/Luminous-Dynamics/symtropy/issues/4713), with implementation in [draft PR #4715](https://github.com/Luminous-Dynamics/symtropy/pull/4715). Real-time task evaluation should use the source observation lineage at actual actuator-command application; result-receive age alone is insufficient.


## Task-conditioned information sufficiency

The real-time benchmark now distinguishes chronological age from decision sufficiency. See [decision_sufficiency_horizon.md](./decision_sufficiency_horizon.md). The benchmark asks whether the old action remains within a frozen task-loss regret tolerance and robustly admissible, not simply whether a timestamp is under a fixed number of milliseconds.
