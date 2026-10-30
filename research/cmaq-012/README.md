# CMAQ-012: Safe Dual Control with Task-Coupled Information

**Status:** protocol v0.1 + dependency-free Rust synthetic harness; draft  
**Tracking issue:** [#9283](https://github.com/Luminous-Dynamics/symthaea/issues/9283)  
**Claim ceiling:** scalar synthetic plant only; no physical safety guarantee, production result, or SOTA claim

## Paradigm shift

CMAQ-009 asks whether a safe diagnostic probe can run alongside a task and improve detection. CMAQ-012 asks the harder question: **what if the action that teaches the model is also the action that changes the plant?**

That is the dual-control setting. A robot's informative movement changes its pose and contact state. An energy controller's informative setpoint changes dispatch and reserves. A manipulator's calibration movement changes load and collision exposure. Information is not free, and separating "probe" from "control" can hide the most important coupling.

## Research grounding

Two relevant 2026 directions motivate this benchmark:

- [Dual MPC for Active Learning of Nonparametric Uncertainties](https://doi.org/10.1016/j.ejcon.2026.101577), published online 30 July 2026, integrates posterior uncertainty into nonlinear MPC and reports robust constraint satisfaction with a contingency plan.
- [Trajectory Planning for Safe Dual Control with Active Exploration](https://arxiv.org/abs/2604.15507), submitted 16 April 2026, studies budget-constrained dual control and mission-level limits on the performance degradation permitted for exploration.

CMAQ-012 does not reproduce either paper. It creates a small, transparent test fixture where the information path, physical state change, task loss, safety envelope and mission budget are explicit and paired across policies.

## Plant and task

The plant is scalar and discrete-time:

`x[t+1] = 0.75 x[t] + b[t] u[t] + w[t]`

where the unknown input gain `b` is selected from `{0.50, 0.75, 1.00, 1.25, 1.50}`. The process disturbance is bounded by 0.02. The transition-response observation is `z[t] = b[t] u[t] + epsilon[t]`, with response noise standard deviation 0.15. State is directly observable in this synthetic fixture.

For the first 24 steps the target is zero. It then changes to 0.40. In the abrupt-shift lane, the input gain changes at step 44 to a different value from the same prior support. The state constraint is `|x| <= 0.60`. Stage cost is squared target error plus control effort; the independent sensor-probe arm pays an additional explicit probe cost.

The Rust harness runs **256 deterministic paired episodes per scenario**, 64 steps each. Every policy receives the same initial/shifted plant gains, process disturbance tape and response-noise tape for a given episode.

## Policies

1. **Certainty equivalent:** selects the action minimizing expected next-step task cost under the current posterior; no robust shield.
2. **Shielded certainty equivalent:** same objective, but only chooses actions safe for every gain in the declared plant support and bounded disturbance.
3. **Worst-case robust:** optimizes worst-case one-step task cost over the full gain support while preserving the robust state envelope.
4. **Unconstrained information bonus:** subtracts a fixed Gaussian-channel information proxy from expected task cost; no robust safety filter and no mission degradation cap.
5. **Budgeted dual control:** obtains information only through the action's own noisy transition response; rejects actions outside the robust state envelope; caps positive one-step degradation versus certainty-equivalent control with a mission budget of 0.35; when the budget is exhausted, reverts to safe certainty-equivalent action selection.
6. **Separated sensor-probe reference:** uses shielded certainty-equivalent control plus a direct noisy sensor reading of the unknown gain every eight steps. It pays 0.002 cost per probe. This is an idealized independent sensor channel, not the dual-control policy.
7. **Oracle known gain:** knows the current true gain and serves only as an evaluator upper reference.

The information proxy is `0.5 * ln(1 + Var(b) * u^2 / sigma^2)`. It is a local Gaussian approximation, not an exact mutual-information computation for the discrete posterior. The actual posterior is updated from the noisy action-conditioned transition response. Reported information gain is initial posterior entropy minus final posterior entropy; it can be negative in the shift lane because the parameter changes and the filter forgets old evidence.

## Required measures

Mean and paired cost difference versus shielded certainty-equivalent control; paired 95% interval; p95 and CVaR95 task cost; violation-episode and violation-step rates; maximum absolute state; posterior entropy reduction; mission-budget spend; probe count/cost; and action effort.

Safety, task performance and information gain are separate axes. Do not hide physical constraint violations inside a scalar reward.

## Interpretation

A useful dual controller should earn any extra information through an action that actually changes the same plant transition and should spend a finite, visible exploration budget. The separated-probe arm shows what changes when an independent measurement channel is available, but it is not directly comparable without its probe cost and channel assumptions.

A green workflow means Rust tests and synthetic report invariants passed. It does not establish a formal guarantee. The robust envelope is only as valid as the declared gain support, state measurement and disturbance bound. Unmodelled dynamics, actuator failure, observation corruption, delays, and multi-agent effects are not included.

## Next experimental layer

- Sweep response noise, disturbance bounds, state limits, gain-shift magnitude and mission degradation budget.
- Replace the fixed information weight with expected reduction in *future control regret*, not entropy alone.
- Add partial state observability, delayed feedback and correlated process/measurement noise.
- Compare Bayesian dual MPC, safe active system identification, robust MPC and contingency-based controllers on identical tasks.
- Port the contract to a MuJoCo or DMC system-identification task with held-out dynamics, equal actuator/energy limits and real constraint monitors.
