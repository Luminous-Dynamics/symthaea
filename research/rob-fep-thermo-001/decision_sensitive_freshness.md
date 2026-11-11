# Decision-sensitive freshness and actuation-side Age of Information

Parent: [ROB-DISTRIBUTED-INFERENCE-001 / #11573](https://github.com/Luminous-Dynamics/symthaea/issues/11573).

Physical measurement owner: [Symtropy #4713](https://github.com/Luminous-Dynamics/symtropy/issues/4713); initial Rust reference: [Symtropy PR #4715](https://github.com/Luminous-Dynamics/symtropy/pull/4715).

Composes with:
- Symthaea #11122 EVSI and #11338 robust active sensing;
- Symthaea #11561 real-time cognition, #11581 adaptive commitment horizon and #5468/#3351 metareasoning;
- Symtropy #4117 state-dependent freshness/certificate validity, #4382 action projection, #2836 controller dispatch, #4044 energy boundaries and #4610 thermal state;
- Symtropy #4689 asynchronous thermal locomotion experiment.

This extension owns the comparison between *observation age* and *decision value of freshness* in inference placement. It does not create a second AoI monitor, EVSI planner, scheduler, controller or certificate authority.

## 1. Why transaction latency is not enough

A remote inference result may be fast after it starts but still be based on an old observation. Conversely, a result can arrive promptly and then sit in a local action queue, wait for projection, or wait for a controller slot before the command reaches the motors.

Therefore maintain separate clocks and events:

1. observation generation / sensor exposure;
2. observation cutoff and encoding;
3. request transmission and remote queue;
4. remote inference;
5. result reception;
6. local verification and physical projection;
7. controller dispatch;
8. command application / actuator response.

**The primary freshness measure is age at the physical destination that uses the information.** For control, that destination is the actuator, not the inference server or message receive callback.

## 2. Actuation-side Age of Information (AoI)

Let u(t) be the generation timestamp of the freshest source observation underlying the command currently being applied at the actuator. In the single-robot-clock reference case:

`AoI_act(t) = t_robot_monotonic - u_robot_monotonic(t)`

The observation identifier and generation timestamp must be carried through the complete command/decision receipt so the actuator-side command can be linked to its exact source observation.

Report time-weighted mean, p50, p95, p99 and worst observed actuation-side AoI. Do not compute percentiles by treating irregular update samples as equally weighted; use elapsed-time weighting over the AoI process. Report the source update interval and skipped-frame rate separately.

Use one robot monotonic clock for both observation generation and command application where possible. If an external camera/edge/cloud clock must be involved, consume the bounded clock mapping and uncertainty semantics from #4117. Unknown mapping means Unknown freshness; raw timestamps from unrelated clocks must never be subtracted.

Keep distinct:
- inference duration;
- network RTT;
- result age at receipt;
- queue/verification/projection/dispatch delay;
- actuation-side AoI;
- actuator command-to-motion delay.

A receipt-age pass does not guarantee an actuation-age pass. The new distributed-inference corpus includes DI20 specifically for this case.

## 3. Freshness is not decision value

AoI tells us how old the information driving the actuator is. It does not tell us whether a newer update changes the decision.

For a declared current belief b_t and frozen task loss L, define an evaluator-side stale-decision regret:

`R_stale(t) = E_{s~b_t}[L(a_stale,s)] - min_{a in A_adm(b_t)} E_{s~b_t}[L(a,s)]`

where a_stale is the action selected from the older belief/observation and A_adm(b_t) is the independently admissible action set for the current belief/evidence. If the old action is no longer admissible, record an independent admission failure; do not hide it inside a finite regret value.

For a candidate new observation o, define decision-focused expected value of sample information using the existing EVSI owner:

`EVSI(o) = rho(b_t) - E_{o'}[rho(b_t | o')]`

where rho is the minimum expected task loss under the frozen decision problem. EVSI is in the declared task-loss units. It is not Shannon entropy and is not joules.

Keep the following dimensions separate:
- actuation-side AoI and tail freshness;
- EVSI and realized decision-regret reduction;
- network/compute/sensor energy;
- thermal and battery reserve;
- action admissibility and recovery viability.

Do not subtract joules from EVSI unless a separate, explicit and validated utility conversion has been declared. Prefer a Pareto comparison.

A low-AoI update with zero EVSI may not be worth its transmission, encoding, compute, queue and thermal cost. A moderately aged observation can remain decision-sufficient if all relevant states still share the same admissible action. Conversely, a small age increase can be catastrophic near a contact transition, obstacle crossing, actuator derating boundary or recovery deadline.

## 4. Required scheduler baselines

Compare on a matched robot/task/clock/network/compute budget:
1. fixed-period updates;
2. minimum-AoI or age-only scheduler;
3. freshness/deadline-aware inference scheduler;
4. value-of-information / EVSI scheduler;
5. AoI + EVSI without physical viability gating;
6. AoI + EVSI + #4117 current certificate validity + #4588 observation-constrained recovery;
7. direct local policy / no remote update;
8. evaluator-only latest-state oracle, never deployable.

For on-device unified-memory contention, include **EdgePace** as a strong runtime baseline. EdgePace (arXiv:2610.37307, October 2026) schedules optional LLM prefill/decode around foreground memory-bandwidth pressure and deadline slack. On a OnePlus 13 and three Jetson platforms, the authors report 24.1%–96.1% fewer foreground deadline misses while retaining 64.9%–98.5% of LLM throughput; their OnePlus 13 example reports 91.3% fewer camera deadline misses while retaining 90.4% throughput. These are results from that study's hardware/workloads, not Luminous measurements.

## 5. Payload compression and freshness saturation

Jeon, Mtowe and Kim (Frontiers in Robotics and AI, 4 November 2026) provide a particularly useful physical reference. On a camera-based mobile robot, the authors link each applied motor command to its source camera frame and reconstruct actuation-side AoI on one robot monotonic clock.

In their constrained-channel experiment:
- raw-image PUPPET payload: 61,570 B;
- first edge cut E1 contour payload: 2,694 B;
- E2 feature tuple: 390 B;
- E3 scalar feature: 4 B;
- mean actuation-side AoI: 82.73 ms (PUPPET) to 44.75 ms (E3);
- p95 actuation-side AoI: 208.73 ms to 68.73 ms.

The first cut reduces payload by 95.6% and captures most of the measured freshness improvement; reducing E1 to E3 by a further factor of 674 yields only about 0.5 ms more mean AoI in that testbed. Their shared-medium analytical extension shows a different saturation point for robot density: 30-fold budget-count improvement from PUPPET to E1, then about 1.8-fold from E1 to E3. Thus byte minimization, actuation freshness and team capacity do not share one optimum.

This motivates a **payload–AoI–decision-value frontier**, not a "smallest packet wins" rule. Test multiple compression levels and preserve task-specific accuracy, EVSI, tail AoI, network utilization and fleet admission as separate measures.

Source: Jeon et al., https://doi.org/10.3389/frobt.2026.1999591

## 6. Asynchronous action-chunk and real-time policy baselines

The following 2026 manipulation methods should be represented in the compatible manipulation lane, while the locomotion lane remains separate:

- **FASTER (NeurIPS 2026):** horizon-aware flow sampling; official implementation reports 10× acceleration for immediate action generation on tested pi0.5/X-VLA policies and a streaming client/server path. It provides a strong near-term reaction-time/action-horizon baseline. Its repository lists >8 GB inference memory, so hardware feasibility must be reported rather than assumed. https://github.com/hku-sail/FASTER
- **PACT-VLA (arXiv:2610.28019, 20 October 2026):** generates candidate chunks from an inference-start observation, then uses a later pre-commit observation to select a complete chunk. At four-step delay, authors report +15.0 percentage points over RTC on Kinetix, +5.2 points over single-candidate async GR00T N1.7 on LIBERO, and 60% to 82% mean success over naive asynchronous execution on three real-robot tasks. https://arxiv.org/abs/2610.28019
- **REMAC (ICLR 2026):** addresses intra-chunk inconsistency as well as prefix-preserved inter-chunk continuity; it is a required comparison where the policy/task implementation is compatible. https://proceedings.iclr.cc/paper_files/paper/2026/hash/c35834443b7881e782e52b3519fe27c7-Abstract-Conference.html

These methods are not all apples-to-apples. Keep weights, task sets, training recipes, action frequency, compute hardware and horizon explicit. Their relevance is that action freshness, intra-chunk alignment and reaction time need distinct measurement.

## 7. Known-answer corpus updates

The 22-case distributed-inference corpus now includes:
- DI20: the remote output is fresh at receipt but stale by the time its command would be applied at the actuator; reject it.
- DI21: the update is very fresh, but EVSI is zero; do not pay the update cost solely to reduce AoI.
- DI22: local LLM inference competes with foreground camera processing for unified-memory bandwidth; deadline-aware suspension preserves the foreground path while retaining useful inference throughput.

DI18 and DI19 were also corrected so they isolate compression-quality loss and fleet-priority admission rather than accidentally failing the earlier freshness gate.

## 8. Claim boundary

The AoI and regret calculations qualify only the declared timestamp/decision model. They do not validate robot clocks, sensor calibration, belief accuracy, task-loss choice, whole-system energy, physical dynamics or safety. An actuation-AoI pass does not confer action authority. #4117 certificate validity, #4382 projection and #2836 controller dispatch remain independent.


## Age of Actuation and Probability of Missing Actuation

A 2026 IEEE Transactions on Communications paper formalizes **Age of Actuation (AoA)** and **Probability of Missing Actuation (PoMA)** alongside Age of Information. Its abstract defines AoA as a timeliness measure for status updates that are used to actuate; it characterizes average AoA, AoA violation probability and PoMA, including the cost of missed actions. The model is a wireless-power-transfer system, not a robot, so these concepts should be treated as a useful communication/control comparator rather than copied without adapting the actuation semantics.

- Nikkhah, Ephremides & Pappas, “Age of Actuation and Timeliness: Semantics in a Wireless Power Transfer System,” IEEE Transactions on Communications, 2026, DOI: https://doi.org/10.1109/TCOMM.2026.3663227

Luminous should report, as separate metrics:
- actuation-side AoI from exact source observation to command application;
- actuation deadline violation probability (PoMA-style, with the event and denominator declared);
- probability that the command is physically admitted/applied before its task deadline;
- task-loss impact of a missed actuation, in fixed task-loss units;
- decision-focused EVSI of requesting a newer observation;
- physical resource and recovery consequences.

Do not identify AoA with the AoI measurement implemented in Symtropy #4713. AoA, actuator-side AoI, command-to-motion delay, EVSI and missed-actuation cost are related but distinct. A lower AoI does not imply lower probability of missing a critical action if the actuation deadline or physical feasibility changed.
