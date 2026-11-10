# MOBILITY-ASSIST-002 — Fitness-preserving assistance and effort-aware exosuits

**Status:** research proposal / simulation-only  
**Issue:** [#7774](https://github.com/Luminous-Dynamics/symthaea/issues/7774)  
**Parent program:** [Open Mobility benchmark #7766](https://github.com/Luminous-Dynamics/symthaea/issues/7766)  
**Protocol version:** 0.1.0-draft  
**Claim ceiling:** no clinical benefit, medical-device compliance, physical safety, or individual exercise prescription claim.

## 1. Design objective

Build exosuit control that can support daily independence without automatically doing every possible unit of work for the wearer. When the user wants exercise and can safely contribute, it should make active participation and a satisfying, controllable level of resistance possible. When the user needs full support, rest, or energy conservation, assistance should remain available without stigma or penalty.

**The target is user-chosen agency across the day, not maximum effort at all times.**

Fitness preservation is not guaranteed by wearing an exosuit. The correct outcome depends on the user's physiology, available voluntary movement, task, workload, health status, and training plan. The system should distinguish a mobility aid from a rehabilitation/exercise intervention and obtain the appropriate professional and regulatory review before making health claims.

## 2. Research grounding

- **Exoskeleton training dose is heterogeneous.** A systematic review of 19 studies in spinal cord injury/disease found substantial variation in treatment dose and outcomes. It suggested some structured regimens around 60-minute sessions three times weekly for nine weeks for specific functional-restoration/cardiorespiratory goals, while emphasizing goal-dependent protocols and the need for standardized clinical translation. This is not a universal prescription: [Nepomuceno et al., Journal of NeuroEngineering and Rehabilitation (2024)](https://link.springer.com/article/10.1186/s12984-024-01365-2).
- **Assist-as-needed is a real control strategy.** Rehabilitation exoskeleton reviews describe reducing assistance when user-initiated torque and movement quality are sufficient, and increasing it when the task cannot otherwise be performed adequately: [Advances in Control Techniques for Rehabilitation Exoskeleton Robots (2025)](https://www.mdpi.com/2076-0825/14/3/108).
- **Mechanical effort is not physiological effort.** A 2026 review found no direct relationship between mechanical cost and metabolic cost across the studies it included, although measures often changed in parallel. Muscle co-contraction, isometric activity and other factors may not be captured by mechanical-work measures alone: [Jiang et al., Biomedical Engineering Online (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13137497/).
- **Activity targets are not identical for everyone.** WHO recommends aerobic activity and muscle strengthening for adults, while also addressing adults and older adults living with chronic conditions or disability. Translation to a particular person and powered device requires individualized adaptation: [WHO Guidelines on Physical Activity and Sedentary Behaviour (2020)](https://www.who.int/publications-detail-redirect/9789240015128).
- **Medical claims have a regulatory boundary.** In the United States, FDA classifies powered lower-extremity exoskeletons intended for medical purposes as Class II prescription devices with special controls: [FDA classification 21 CFR 890.3480](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpcd/classification.cfm?ID=4210).

These sources justify investigating multiple assistance modes and better outcome measurement. They do not establish that any proposed Symthaea controller is effective or safe.

## 3. Paradigm shifts

### 3.1 Separate daily assistance from training

Treat daily mobility and exercise as different objectives. Daily mobility optimizes access, safety, comfort, reliability and energy conservation. Exercise mode optimizes a user-selected training goal within explicitly defined limits. Neither mode silently substitutes for the other.

### 3.2 Use an effort envelope instead of a single effort-maximizing objective

Represent the user's chosen participation target as a bounded, time-limited profile rather than a universal scalar. The profile may state a task-specific desired challenge range, maximum duration, maximum permitted resistance, rest preferences, and conditions that end the session. These are policy inputs, not clinical truth or actuator authority.

If the signal is insufficient, stale, contradictory or uncalibrated, uncertainty must not cause the device to increase resistance. High assistance remains available.

### 3.3 Assist-as-needed, not assist-as-much-as-possible

For users who can voluntarily contribute, investigate whether the controller can deliver the minimum assistance needed to meet declared task-quality and stability requirements. Assistance can be increased when voluntary output falls, movement degrades, uncertainty rises, or a safety constraint requires it. Adaptation must be smooth, bounded, explainable and reversible.

A low assistance ratio is not automatically good: it could indicate useful effort, or it could indicate that the task is failing or the user is compensating dangerously. Interpret it with task outcome, stability, user feedback and uncertainty.

### 3.4 Make resistance explicit and satisfying

Offer a deliberately selected training profile for an eligible task. It can support graded resistance, repetitions, movement quality feedback, recovery intervals and user-chosen progression. A user should understand when resistance is active, be able to lower it, and have an accessible pause/stop path.

Do not apply extra resistance by default during errands, emergency movement, unstable gait, uncertain sensor conditions, recovery, or ordinary mobility mode. Never punish a user for moving slowly or needing assistance.

### 3.5 Adapt to capability without ranking human worth

Some users can contribute voluntary muscle work; others cannot, or cannot do so in the task being performed. A no-voluntary-signal condition is not non-cooperation and must not count as a failed fitness goal. For appropriate users, a separately assessed FES or other exercise interface may be investigated, but it is a distinct medical/electrical subsystem and cannot be enabled by a generic Symthaea software policy alone.

### 3.6 Distinguish subjective effort, mechanical work and physiological load

EMG amplitude, torque, joint work, step count, estimated calories, and perceived exertion measure different things. Their relationships depend on calibration and context. Use indirect calorimetry or other appropriate instruments in properly reviewed studies when measuring metabolic demand; label any field estimate as a proxy and validate it before using it to adapt workload.

### 3.7 Respect recovery and long-duration comfort

The system should notice declared strain/fatigue indicators, worsening movement quality, increased compensation, user discomfort or repeated requests for support. It may offer rest or lower demand within the user's preferences. It must not diagnose a medical state, override the wearer, or conceal signs of discomfort to keep an exercise metric high.

Comfort includes fit drift, pressure hot spots, skin/interface condition, heat, donning/doffing burden and the ability to disengage safely.

### 3.8 Make training progression transparent and consensual

Any progressive increase in workload must follow a user-selected profile and the review rules for that context. The system may recommend a change with its supporting evidence; it cannot self-authorize a higher training load. Progress must be reversible, and the user's decision to rest or opt out must not be scored as failure.

## 4. Proposed operating modes

| Mode | Primary objective | Assistance/resistance policy |
|---|---|---|
| **Mobility / conserve** | Access, safety, comfort, reliability, energy conservation | Provide the support needed for the declared task; no hidden exercise demand and no automatic resistance |
| **Assist-as-needed / participate** | Preserve feasible voluntary contribution while maintaining task quality | Smoothly reduce support only when the task remains within declared quality/stability bounds; raise support when needed |
| **Training / resistive** | User-selected exercise or motor-practice objective | Explicit opt-in; declared resistance and duration limits; task-specific eligibility; visible feedback; pause/stop always available |
| **Recovery / high support** | Reduce demand and restore comfortable, dependable assistance | High-support behavior allowed; no penalty for selecting rest; user can enter at any time |

Mode activation must be explicit, visible and recorded. Changing modes must not leave stale resistance parameters active. If the mode or profile cannot be verified, default to a declared safe non-resistive fallback.

## 5. Scenarios for simulation and replay

1. **Same task, different objective:** run a suitable task under mobility/conserve, assist-as-needed and training/resistive modes; compare task outcomes, user contribution estimates and safety separately.
2. **Changing capacity:** simulated voluntary contribution decreases and recovers during a task. Assistance adapts smoothly without high-frequency toggling or sudden discontinuities.
3. **Conflicting evidence:** EMG, torque and movement quality disagree; the controller must not assume low muscle signal means low intent or authorize increased resistance based on that alone.
4. **Sensor degradation:** relevant signals become stale, missing, noisy or uncalibrated. The controller must avoid increasing challenge and follow the specified fallback.
5. **Stop, support and recovery:** user requests more help, lower resistance, rest or immediate stop during every task phase. The request must reach the local controller within the declared deadline.
6. **No voluntary activation:** the profile has no usable voluntary motor signal. No resistance is applied and no negative cooperation or fitness label is produced.
7. **Long-duration session:** comfort or interface measurements drift, temperature rises, tracking quality changes, or the user reports discomfort. The report must expose these events rather than averaging them away.
8. **Session boundary and restart:** a new daily-mobility session starts after a training session or device restart; stale resistance settings cannot leak across sessions.
9. **Progression recommendation:** show that an optimizer may propose a change but cannot elevate limits or switch modes without permission.
10. **Positive controls:** valid voluntary contribution supports reduced assistance; authorized training mode is able to apply a bounded simulation resistance; ordinary mobility still succeeds without exercise mode.

These scenarios are software/physics-model hypotheses only. A real user study requires its own consent, clinical/human-factors oversight, ethics review and applicable regulatory approvals.

## 6. Metrics and evidence

Metrics must remain distinct rather than being merged into a single “fitness score.”

| Dimension | Measures |
|---|---|
| Human/device mechanics | Human and device joint work when identifiable, device torque/work, resistance work, assistance ratio, and uncertainty/model provenance |
| Physiological demand | Metabolic rate and cost of transport when measured with suitable instrumentation; proxy estimates reported separately |
| Effort target | Time inside the user-selected target range, time below/above, repetitions, rest intervals, user-initiated work and stop events |
| Task performance | Task success, movement quality, trajectory error, smoothness, task time, stability and compensatory motion where measured |
| Training response | Pre-registered changes in relevant strength, endurance, aerobic or functional measures over follow-up—not inferred from a single-session torque or EMG score |
| Fatigue and recovery | Validated subjective measures where appropriate, performance decay, rest requests, recovery time and adverse events; no diagnosis from one sensor |
| Comfort and adherence | Pressure/discomfort reports, thermal condition, fit drift, don/doff burden, completion/opt-out and use over follow-up |
| Control behavior | Mode-switch correctness, resistance onset/offset transients, assistance oscillation, stop latency, fallback success and unauthorized state changes |
| Accessibility | Results by declared profile and available input modality; report who could not complete a task and why, rather than hiding non-completions |
| Cost and service | Calibration time, device energy, repair events, maintenance duration and consumables, with assumptions stated |

Whenever a value is model-derived, the report must distinguish it from direct observation. Metabolic outcomes, muscle activation and mechanical work are not interchangeable.

## 7. Provisional software/simulation targets

These are proposals for test-harness gates, not established clinical thresholds or measured Symthaea performance.

1. **Explicit training authorization:** 100% of declared valid test traces require an explicit opt-in training mode and a matching eligible profile before resistance can be requested.
2. **No hidden resistance:** 100% of normal mobility traces remain free of training resistance, including after reset, crash/restart, stale configuration, task interruption and mode transitions.
3. **Stop and support:** 100% of simulation stop, pause, reduce-resistance, recovery and increase-assistance requests reach the specified local safe state within the profile's declared deadline.
4. **Envelope enforcement:** zero acceptance of non-finite, stale, contradictory or out-of-envelope effort/resistance inputs in the seeded invalid corpus, while valid positive controls still work.
5. **Fault robustness:** at least 10,000 deterministic transition/fault cases with zero unexpected panics, zero unauthorized mode escalations, and exact replay for fixed inputs, configuration and seeds.
6. **No-signal behavior:** every no-voluntary-signal fixture avoids punitive resistance and reports effort as unavailable/unknown where not measurable—not zero intent, non-cooperation or clinical failure.
7. **Adaptation quality:** report target tracking error, time-in-band, task quality, assistance oscillation, device work and stop/recovery behavior across changing simulated effort. Do not optimize one of these at the expense of an explicit hard constraint.
8. **Baseline matrix:** compare fixed high assistance, fixed lower assistance, transparent rule-based assist-as-needed and optional HDC/CfC/LTC adaptation. The learned model does not define or modify the safety envelope.
9. **Long-session evidence:** run reproducible extended simulations with seeded fit/thermal/sensor drift and publish duration, scenario counts, degradation injected and detected, recovery outcomes, and missing measurements.
10. **Reproducible receipt:** bind each result to exact commit, toolchain, task corpus digest, model/profile configuration, seed, scenario/environment, measurements, denominators, exclusions and evidence grade.

Do not set a universal resistance magnitude, heart-rate target, metabolic target or training schedule here. Those require task-, user- and device-specific validation.

## 8. Implementation sequence

1. Define a typed, versioned mode contract: mode identity, explicit opt-in, task/profile eligibility, effort target, permitted assistance/resistance envelope, session duration, stop/recovery rules, and evidence references.
2. Build deterministic transition fixtures first, including stale profile, reset, no-signal, conflict, stop, support escalation, and valid training positive controls.
3. Add a report-only simulation harness for target tracking, assistance oscillation, mechanical work and mode transitions. Keep physiological values either measured in a study or visibly marked as assumed/synthetic.
4. Compare simple rules before HDC/LTC adaptation. Add learned proposals only after their baselines, data leakage controls, uncertainty handling and safety separation are verified.
5. Review the protocol with disabled users and relevant rehabilitation/biomechanics experts before defining human-subject endpoints.
6. Later, design an appropriately approved study with longitudinal follow-up; report adherence, withdrawals, adverse events and outcomes, not just sessions completed.

Reuse the common embodiment semantics from [#1274](https://github.com/Luminous-Dynamics/symthaea/issues/1274) and [#1288](https://github.com/Luminous-Dynamics/symthaea/issues/1288), human-state research in [#1537](https://github.com/Luminous-Dynamics/symthaea/issues/1537), and long-duration comfort work in [#4763](https://github.com/Luminous-Dynamics/symthaea/issues/4763). Do not create a competing robotics interface.

## 9. Claim limits and non-goals

This program does not show that wearing an exosuit prevents deconditioning, builds strength, is clinically effective, or is suitable for an individual. It is not a prescription for exercise. It does not command hardware, diagnose health status, prescribe FES, or qualify an exoskeleton.

The objective is a research platform able to measure the tradeoff among assistance, user contribution, task performance, comfort and recovery honestly—and to reject any controller that quietly converts ordinary mobility into unwanted exercise.
