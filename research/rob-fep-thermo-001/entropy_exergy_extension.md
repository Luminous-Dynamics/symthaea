# Entropy and exergy in thermodynamic cognitive robotics

Status: research / benchmark design addendum. This document defines no deployed objective, physical controller, qualified entropy estimator or hardware result.

Parent: [Symthaea #11496 — ROB-FEP-THERMO-001](https://github.com/Luminous-Dynamics/symthaea/issues/11496).  
Physical-accounting owner: [Symtropy #4631 — LOCO-ENTROPY-ACCOUNT-001](https://github.com/Luminous-Dynamics/symtropy/issues/4631).  
Composes with [Symtropy #4044](https://github.com/Luminous-Dynamics/symtropy/issues/4044), [#4179](https://github.com/Luminous-Dynamics/symtropy/issues/4179), [#4610](https://github.com/Luminous-Dynamics/symtropy/issues/4610), and [#4588](https://github.com/Luminous-Dynamics/symtropy/issues/4588).

## Decision

Robots should manage uncertainty, entropy generation and exergy destruction, but should not minimize entropy universally.

The proposed objective is **useful future capability subject to physical and informational constraints**. Entropy / exergy are separately reported physical diagnostics and possible secondary optimization criteria only after their boundaries, models and uncertainty are qualified.

## Four separate quantities

| Quantity | Meaning | Unit / interpretation | Correct use |
|---|---|---|---|
| Belief entropy | Uncertainty over a declared probability distribution | bits or nats | Calibrated inference diagnostic |
| Expected information gain | Expected belief change about a named target | bits or nats | Compare information acquisition |
| Decision value (EVSI) | Expected reduction in downstream Bayes risk / decision loss | Declared task-loss units | Decide whether an observation is worth its costs |
| Physical entropy | State entropy, entropy flow and entropy generation in a declared physical system | J/K | Thermodynamic accounting |

Belief entropy is not an estimate of physical entropy. Information gain is not a thermodynamic resource. A Shannon entropy decrease does not establish physical bit erasure. A Markov-chain path irreversibility score remains dimensionless and model-relative unless the physical transition mechanism, heat/work assignments and local-detailed-balance assumptions justify a J/K interpretation.

## Open-system entropy accounting

For a declared control volume, use a consistent entropy balance:

dS_CV/dt = Σ_k(Q̇_k / T_b,k) + Σ_in(ṁ s) - Σ_out(ṁ s) + Ṡ_gen

The sign convention is heat and matter entering the control volume as positive entropy flow; T_b,k is the local absolute boundary temperature at heat port k. Matter-flow terms may be omitted only when the selected boundary and process justify a closed-mass approximation. Electrical and mechanical work are energy transfers, not entropy-flow terms merely because they carry energy.

The balance requires a frozen boundary, time window, heat-port manifest, boundary temperatures, material-flow assumptions, unit convention, sign convention, state identity and uncertainty. Missing ports or stale/misaligned measurements produce Unknown / incomplete—not zero entropy generation.

For an idealized quantity Q transferred from a hot reservoir at T_h to a cold reservoir at T_c:

S_gen = Q(1/T_c - 1/T_h) ≥ 0

where Q is positive from hot to cold and temperatures are absolute kelvin. This is an analytic two-reservoir control only. It does not by itself quantify all entropy generation in a robot, including electrical resistance, friction, battery chemistry, electronics or contact events.

For a constant-heat-capacity body with fixed C over the declared interval, its entropy-state change is:

ΔS = C ln(T_1/T_0)

This is a state-change formula under the stated constant-C model, not a universal estimate of entropy production. It must not be substituted for a full open-system balance.

Exergy destruction may be calculated as:

X_destroyed = T_amb S_gen

only when the reference environment is defined, the exergy balance is valid for the selected system and the thermodynamic assumptions are satisfied. If the environment varies materially or the energy/material streams cannot be accounted for, report the entropy result and exergy status as unavailable rather than inventing a reference.

## What the robot should optimize

Use a constrained, multi-objective hierarchy:

1. **Admissibility:** independent safety limits, runtime assurance, actuator authority, safe-stop and recovery constraints.
2. **Service:** complete the declared task and deliver useful work.
3. **Future options:** preserve observation-constrained continuations under all physically possible successor states and required evidence.
4. **Resource performance:** compare battery energy, actuator losses, cooling, compute, wear, thermal margins, entropy generation and exergy destruction as separate dimensions.
5. **Efficiency refinement:** among matched policies with non-inferior service, safety and future options, test whether avoidable exergy destruction can be reduced without moving losses or risk into another subsystem.

Do not let a favorable entropy score compensate for an overtemperature, a lost retreat path, a degraded safe-stop or an unacceptable fall/contact risk. Do not compress battery reserve, thermal headroom, actuator authority, mechanical condition, sensor observability and recovery topology into one unqualified resilience number.

A candidate action can use more immediate energy yet preserve more future mobility. Conversely, a probe can reduce belief uncertainty while consuming thermal headroom and battery reserve. Its information is valuable only when expected decision improvement justifies the physical cost and the probe remains independently admissible.

## Research hypothesis: thermodynamic option preservation

The research hypothesis is that the best long-horizon controller will not simply minimize energy or entropy production. It will maintain a viable set of future actions while choosing among currently admissible actions on a Pareto frontier of:

- delivered service / useful mechanical work;
- net battery energy and electrical / mechanical losses;
- per-node temperature, time-to-derating and cooldown;
- actuator and transmission health;
- cooling and compute demand;
- entropy state change, entropy transfer and entropy generation;
- exergy destruction, where qualified;
- observation-constrained future continuations and terminal safe-stop;
- risk, uncertainty, model discrepancy and adverse events.

The value of an additional computation, observation or physical probe should be priced in its expected effect on the downstream decision, compute/latency budget and physical state—not in raw entropy reduction alone.

## Minimum analytical corpus

The physical-accounting work item #4631 should implement exact known-answer tests before simulator or hardware integration:

1. Equal-temperature ideal heat transfer: zero entropy generation.
2. Finite hot-to-cold transfer: exact positive result from Q(1/T_c - 1/T_h).
3. Invalid heat-flow sign: fail closed; never silently emit negative entropy generation for the declared passive transfer.
4. Invalid absolute temperature: reject non-positive and non-finite values.
5. Missing heat port or boundary temperature: Unknown / incomplete.
6. Multi-node transfer: internal heat transfer cancels from the whole-control-volume energy balance but retains the local entropy-generation contribution.
7. Uncertain T_h, T_c and Q: propagate declared intervals / distributions; do not silently substitute a point estimate.
8. Markov-path irreversibility without local detailed balance: retain dimensionless, model-relative labeling and do not emit J/K.
9. Changed boundary, units, sign convention, calibration, predecessor state or model digest: reject the stale receipt.

Energy closure, entropy balance, numerical stability and empirical model validity are separate acceptance gates.

## Matched benchmark design

Compare conventional cost-of-transport / energy accounting, qualified actuator loss maps, thermal-aware MPC/HQP, thermal-aware RL, Bayesian/set-membership EVSI, corrected action-conditioned FEP + EVSI, and an optional entropy/exergy secondary optimizer. Use the same frozen plant, action limits, sensing budget, compute budget and physical resource boundary in each matched comparison.

The primary test is whether entropy/exergy metrics provide held-out predictive or decision value after conditioning on already available energy, actuator-loss and per-node thermal evidence. If they do not, retain them as diagnostics and do not make them policy objectives.

The stronger test is whether an entropy/exergy-aware secondary optimizer improves the service / future-mobility Pareto frontier without increasing adverse outcomes. Compare paired sequential missions with persistent battery, temperature, actuator-health and recovery state; do not reset between faults. Hold out terrain, payload, ambient temperature, wear and sensor-bias regimes. Preserve failed, aborted, Unknown and incomplete runs in the denominator.

Report separate results for A1, Go2, ANYmal, humanoid, manipulator and simulation platforms. Do not compare raw endurance across unlike payloads, energy boundaries, gaits, batteries or environments.

## External reference strata

- Qian et al., 2026, A1 thermal-aware locomotion: over 27 minutes versus roughly 7 minutes under that study's 3 kg payload condition ([arXiv:2603.01631](https://arxiv.org/abs/2603.01631)).
- Wan et al., 2026, A1 residual thermal policy: over 13 minutes versus roughly 5 minutes in a separate experiment ([arXiv:2605.27046](https://arxiv.org/abs/2605.27046)).
- De Benedittis et al., 2026, temperature-constrained HQP/MPC and redundant-manipulator validation ([DOI: 10.1109/TCST.2026.3721130](https://doi.org/10.1109/TCST.2026.3721130)).
- Bjelonic et al., 2026, physics-grounded motor-loss model and sim-to-real transfer, reporting 32% lower full CoT for ANYmal and transfer across additional robots ([DOI: 10.1177/02783649261459628](https://doi.org/10.1177/02783649261459628)).
- Lee et al., 2026, RAIBO2 marathon, reporting 42.195 km and total CoT 0.25 ([Nature DOI: 10.1038/s41586-026-11102-5](https://doi.org/10.1038/s41586-026-11102-5)).

These are different platforms, task profiles and measurement boundaries—not one leaderboard. They are strong reference strata for energy and thermal engineering, not evidence that any competitor has or has not optimized a whole-robot entropy-production objective.

## Claim ceiling

This document and synthetic arithmetic can establish only the stated equations and finite semantic contracts. They do not qualify Symtropy's thermal model, prove that entropy/exergy metrics improve decisions, demonstrate a physical robot result, establish FEP superiority or authorize actuation.

No entropy-minimization policy should enter the motor-control path from this work.