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
