# Φ-Gated Motor Authority: A Continuous Safety Supervisor over Rule-Based Envelopes

*(IWAI / Active Inference Journal submission draft)*

---

## Abstract (149 words)

We study continuous motor-authority supervision by a scalar
consciousness-inspired correlate (Φ, the output of
`MasterConsciousnessEquation` aggregating 10 sub-signals — not IIT Φ)
across ten heterogeneous robotic platforms. Sweeping five Φ→gain
mappings on a paired Monte Carlo cobot benchmark vs ISO/TS 15066
Speed-and-Separation Monitoring, we find (i) the default global
`SafetyTier` thresholds produce zero throughput because the platform's
empirical Φ lives in [0.099, 0.145] — never reaching the hardcoded 0.6
Green cutoff; (ii) the minimal sufficient mapping is a sprint threshold
matched to the empirical band plus a crawl-rate floor, implementable
as `if signal > sprint { 1.0 } else { floor }`; (iii) this mapping
loses to SSM at S_p ≈ 1 m by 88.6 % but **wins by +71.4 %** at
S_p = 2.5 m (N=30, 95 % CI [+15.4, +127.4]) and holds throughput
where SSM collapses to zero at S_p ≥ 3 m. The gating-shape advantage
replicates at +71.4 % (N=30, CI [+54.1, +88.6]) on a quadrotor
(Figure 3). We frame the method as an ISO 21448 / SOTIF
triggering-condition monitor, not a replacement for certified envelopes.

## Note on terminology — what "Φ" means in this paper

Throughout, **Φ** denotes the scalar output of
`MasterConsciousnessEquation::compute()` — a *consciousness-inspired
integration index* that aggregates ten sub-signals (IIT Φ, broadcast,
working memory, attention, recurrence, embodiment, knowledge, higher-
order thought, narrative, social) via a softmin bottleneck plus a
rescaling factor. It is:

- **not** Tononi IIT Φ in the phenomenological sense — we do not claim
  to measure consciousness;
- **not** load-bearing for the method's validity — any scalar correlate
  that separates "confident cognitive state" from "uncertain state"
  would fill the same role in `sprint_floor_gain(signal, threshold,
  floor)`, which is what the empirical results actually isolate;
- a useful empirical correlate, derived from a consciousness-adjacent
  aggregation, that happens to have stable per-platform bands.

Our claims stand or fall on the gating-shape sufficiency result (§5)
and the ISO-SSM comparison (§6), not on whether IIT's axioms are
correct. "Φ-gated" is shorthand for *aggregated-cognitive-correlate-
gated*; we keep the Φ notation for brevity and because IWAI readers
recognize it.

---

## §1. Introduction

Industrial robot safety in 2026 is overwhelmingly discrete and
rule-based. The dominant patterns — PX4's failsafe state machines,
ISO/TS 15066's Speed-and-Separation Monitoring (SSM), Mobileye's
Responsibility-Sensitive Safety (RSS) envelopes, Franka's three-zone
force-feedback limiter — all bucket runtime state into a small number
of discrete regions and apply piecewise-constant authority inside each.
This design is legible to certifiers, amenable to ISO 13849 fault-tree
analysis, and catastrophically mis-calibrated when the real-world
scenario falls in the tails of its presumed distribution. When ISO/TS
15066's protective distance `S_p` is set conservatively under epistemic
uncertainty about the human-motion envelope, the robot simply stops:
zero throughput, zero delivered value.

ISO 21448 (SOTIF) was published in 2022 to name this failure mode
explicitly — "functional insufficiency" — and to mandate that ML/AI
safety-adjacent components be monitored for out-of-distribution
behavior at runtime. SOTIF provides the vocabulary (*triggering
conditions*, *intended functionality*, *performance limitations*) but
no concrete runtime architecture. Filling in that architecture is
currently an open problem.

In parallel, Active Inference and Integrated Information Theory have
matured as quantitative frameworks for agent cognition. VERSES Genius
shipped a commercial Active Inference controller in April 2025;
academic demos from the Friston lab and the IWAI workshop track
routinely compute Φ-adjacent scalars at hundreds of Hertz on
representative platforms. What has not happened is an empirical study
of whether such scalars are *useful as safety supervisors* — whether
their runtime band structure is stable enough to serve as a SOTIF
triggering-condition signal, and whether a mapping from that scalar
to motor authority can beat the rule-based baselines under the
conditions where the rule-based baselines fail (conservative
`S_p`, epistemic uncertainty about obstacle kinematics).

**Our contribution.** Across ten heterogeneous robotic platforms,
each with a shipped simulation demo and a unified `RoboticAgent::tick()`
API, we:

1. **Identify an empirically validated minimal 2-part sufficient
   condition** for a continuous Φ→gain supervisor. Five variants
   spanning the design space (Default, Continuous, Clamped-linear,
   Recalibrated 4-tier, SprintFloor 2-level) were run through an
   identical Monte Carlo harness against ISO/TS 15066 SSM as baseline.
   Under observation-driven Φ, only the SprintFloor variant survives;
   the other three non-trivial variants collapse to dead-arm because
   they allow gain → 0 for Φ values the distribution regularly hits
   (§4, §5).
2. **Report an S_p-sweep crossover result** that frames the paper's
   core claim as *graceful degradation under epistemic uncertainty*:
   at standard cobot S_p ≈ 1 m, SprintFloor loses to SSM by 88.6 %;
   at S_p = 2.5 m, SprintFloor wins by +71.4 % (95 % CI [+15.4,
   +127.4], N = 30); at S_p ≥ 3 m, SSM catastrophically fails (0
   cyc/100 s) while SprintFloor holds ~0.8 cyc/100 s (§6, Figure 2).
3. **Replicate the gating-shape advantage on a second platform**
   (quadrotor), showing +71.4 % advantage at N = 30 under synthetic
   signal (Figure 3), +125.8 % under live-Φ RoboticAgent observations
   (§9.2), and a cross-platform table covering six platforms
   (Figure 5) with advantages ranging from +89 % to +231 %, every
   CI excluding zero.
4. **Offer a dual-channel hazard-gate composition pattern** for
   single-point-of-failure actions (surgical cautery), where Φ is
   paired with a Φ-independent hardware interlock and the hazardous
   action fires only when *both* channels approve. This preserves
   ISO 13849 diverse-redundancy semantics while using Φ as one of
   the two channels (§7).
5. **Ship a `symtropy calibrate <platform>` CLI** that operationalizes
   the per-platform threshold-tuning recipe as a one-liner, so future
   platform adopters can copy-paste the recommended constant (§8).

The method's novelty is narrow: we are not proposing consciousness as
a safety function in the ISO 26262 sense, nor claiming Φ measures
subjective experience, nor advocating replacement of certified
hardware envelopes. We are proposing that an aggregated cognitive
correlate, used as a SOTIF triggering-condition monitor over a
certified envelope, fills an empirically identifiable gap in the
current safety-architecture literature. The evidence is the dataset
and the six-platform replication; the principled justification is
the SOTIF framework.

## §2. Background

**Integrated Information via HDC + FEP.** Tononi's Integrated
Information Theory (IIT; Tononi 2004, 2008) defines Φ as the intrinsic
cause-effect power of a physical system, quantified by the
integrated information generated over its minimum-information partition.
Computing IIT Φ exactly is super-exponential in system size; approximate
quantitative surrogates are active research. Friston's Free Energy
Principle (FEP; Friston 2010) provides a complementary framework:
agents minimize variational free energy across perception and action,
producing a scalar that tracks how well the model's predictions match
the sensory stream. Hyperdimensional Computing (HDC; Kanerva 2009;
Rabaey 2020) provides a fast vector-symbolic architecture for
perception and memory, with bind/bundle/permute operations running at
hardware speeds and temporal dynamics implementable via Liquid
Time-Constant (LTC) networks (Hasani et al. 2021). The composition
we use — HDC-LTC for temporal state + FEP for action selection +
`MasterConsciousnessEquation` for a scalar integration index —
produces a single Φ value per tick that is neither IIT-canonical nor
FEP-canonical, but is empirically stable and inexpensive to compute.

**Cobot and surgical safety standards.** ISO 10218-1/2:2025 governs
industrial robots; its amendment ISO/TS 15066:2016 defines four
cobot interaction modes, of which Speed-and-Separation Monitoring
(SSM) is our comparison target. SSM defines a protective distance
`S_p` as a function of robot speed, human approach speed, and
reaction delay; when human and robot are closer than `S_p`, the robot
must come to a controlled stop. IEC 80601-2-77 covers surgical
robots; ISO 21448:2022 (SOTIF) covers the functional-insufficiency
failure mode for ML-containing systems. Our method frames itself as
a SOTIF triggering-condition monitor — detecting scenarios where the
model's own prediction is likely wrong — rather than a replacement
for the certified hardware envelope.

**Prior HDC in robotics and Active Inference in production.** HDC has
been used in robotics primarily for perception and memory — Neubert
& Schubert (2021) surveyed approaches spanning localization,
navigation, and episodic recall. We are not aware of prior work using
HDC in a closed-loop motor control loop with an Active Inference
action-selection stage. Active Inference in production robotics is
dominated by VERSES Genius (commercial, 2025) and a handful of
academic demos (Catal et al.; Smith et al.) from the IWAI workshop
track; most demos operate in low-dimensional kinematic toy worlds
rather than on full platform simulators.

## §3. System Architecture

**The unified `RoboticAgent::tick()` API.** Each of our ten robot
platforms is instantiated through a single trait,
`EmbodimentBridge`, declared in `symthaea-core/src/embodiment.rs`.
The trait's production entry point is
`RoboticAgent::tick(observation: &[f64], danger_level: f64) -> f64`,
returning a scalar motor-gain advisory. Per tick, the agent:

1. Packages the observation as an `Observation` for the FEP layer.
2. Calls `fep.perceive()` — inferring hidden-state updates,
   producing prediction-error / precision / belief-change /
   free-energy components.
3. Calls `fep.select_action()` — producing an action distribution
   with expected-free-energy weighting.
4. Updates a low-pass `caution` signal from `danger_level`.
5. Constructs `ConsciousnessInputs` (ten sub-signals) with FEP
   prediction-error → attention, FEP precision → working_memory,
   FEP belief-change → recurrence, FEP total free-energy →
   knowledge, action-distribution concentration → broadcast. Each
   is blended 35 : 65 with a fixed-default prior.
6. Calls `consciousness.compute(&inputs)` to produce the scalar
   Φ ∈ [0, 1].
7. Returns `SafetyTier::from_phi(phi).motor_gain()` as the advisory.

The scalar can be consumed three ways across our demo suite:

- **Magnitude attenuation** — flight, vehicle, AUV, helicopter,
  humanoid: `cmd_scaled = cmd * gain`.
- **Mode selection** — exoskeleton's `AssistanceMode::from_phi`,
  surgical's `SurgicalSafetyLevel::from_phi`: Φ thresholds between
  discrete operating modes.
- **Mission-phase tracking** — orbital: Φ tracks communication-
  window / solar-exposure constraints across a 90-minute LEO orbit.

This paper focuses on the magnitude-attenuation family.

**SOTIF positioning.** Per `symtropy-robotics-bridge/src/agent.rs`'s
doc-comment (commit `8357db9a68`), `RoboticAgent::tick` is documented
as returning an *advisory* scalar whose semantic is: "given the
internal HDC/FEP prediction-error state at this tick, the model's
confidence in the intended behavior is `g`; downstream actuation
should be scaled by at most `g` of its nominal authority." The method
has no visibility into hardware envelopes (joint limits, PFL force
caps, ISO/TS 15066 protective distance, FDA-cleared interlocks);
callers that need safety-rated behavior *must* combine this gain with
an independent hard-envelope check.

In ISO 21448 / SOTIF terms: `tick()` is a *triggering-condition
monitor* — it detects scenarios where the model's own prediction is
likely wrong. It is *not* a safety function in the ISO 26262 sense,
and should never be the sole gate on a single-point-of-failure
hazardous action. §7 develops the dual-channel composition pattern
that satisfies ISO 13849 diverse-redundancy when Φ is one of two
channels.

## §4. The Monte Carlo Benchmark

**Setup.** `manipulator_benchmark.rs` runs a 7-DOF industrial
manipulator (Franka Panda DH kinematics, DLS IK per Wampler 1986)
against a sinusoidal human obstacle (period 4-12 s, closest approach
0.25-0.55 m, farthest 1.5-3.0 m, phase jitter). Each trial is 100 s
of simulated time at 500 Hz physics plus 25 Hz cognitive tick. Trial
seeding is deterministic via splitmix on trial index, so any re-run
with the same N reproduces the numbers bit-exactly — this gives us a
reproducibility anchor, not a noise estimate.

**Comparisons.** Five Φ→gain variants are benchmarked against ISO/TS
15066 SSM:

- *Default tiers* — hardcoded `SafetyTier` boundaries at
  Φ = 0.6 / 0.3 / 0.1 (from `symtropy-consciousness-physics::safety`).
- *Continuous* — linear map on the pre-wiring empirical band
  [0.099, 0.145] → [0, 1].
- *Clamped-linear* — linear above a `FLOOR = 0.3` knee.
- *Recalibrated* — 4-tier matched to the pre-wiring empirical band.
- *SprintFloor* — binary: `gain = 1.0 above SPRINT_THRESHOLD else FLOOR`.

Plus a sixth "Adaptive" proximity-keyed gradient stand-in policy,
included to separate the gating-shape story from the specific choice
of Φ-signal.

**Result (post-FEP-wiring, N = 10 per Φ variant, S_p = 1.0 m):**

| Variant        | cyc/100s     | vs ISO     |
|----------------|--------------|------------|
| Default tiers  | 0.00 ± 0.00  | -100.0 %   |
| Continuous     | 0.00 ± 0.00  | -100.0 %   |
| Clamped-linear | 0.00 ± 0.00  | -100.0 %   |
| Recalibrated   | 0.00 ± 0.00  | -100.0 %   |
| **SprintFloor** | **0.80 ± 0.42** | **-88.6 %** |
| Adaptive       | 1.70 ± 1.21  | -75.7 %    |
| ISO SSM        | 7.00 ± 0.00  | baseline   |

Under observation-driven Φ, only SprintFloor survives. The
three non-trivial variants (Continuous, Clamped-linear, Recalibrated)
collapse to zero cycles because they all allow `gain → 0` at their
lower-tier boundaries, and FEP-modulated Φ's lower tail regularly
dips below those boundaries on some trial seeds. SprintFloor's
non-zero `FLOOR = 0.3` is the structural difference that prevents
collapse. This leads directly to §5.

## §5. The Minimal Sufficient Condition

From the §4 table, we extract three design requirements that a
successful Φ→gain mapping must satisfy:

1. **A sprint threshold matched to the empirical Φ band.** The
   default `SafetyTier::from_phi` uses thresholds of 0.6 / 0.3 / 0.1,
   picked by analogy to NRC nuclear-safety tiers. The manipulator's
   post-FEP-wiring Φ distribution is [0.088, 0.133] with p50 = 0.121;
   no frame ever reaches 0.6, so the arm is always in Orange or Red
   and the Default variant dead-arms. A calibrated threshold must lie
   *within* the platform's empirical band.

2. **A non-zero crawl-rate floor.** Under observation-driven Φ, the
   Φ distribution has a long lower tail. Any mapping whose output
   reaches 0 somewhere in that tail will produce a dead-arm frame on
   the trials where Φ dips below the zero-gain threshold. The arm's
   inverse-kinematics loop needs a minimum gain to converge on each
   waypoint; below that floor, cycles are not completed at all. The
   `FLOOR = 0.3` we use is measured against the DLS IK's
   convergence-tolerance vs controller-gain tradeoff on the 7-DOF
   manipulator; platforms with different kinematics will have
   different floors.

3. **Commitment to full authority above the sprint threshold.** A
   linear ramp between floor and full authority (Clamped-linear)
   does worse than a hard step, because the in-between gains still
   don't produce enough torque to complete pick-place cycles within
   the sinusoidal human-obstacle window. Full authority during brief
   "sprint" windows is necessary.

These combine into the two-line implementation:

```rust
fn sprint_floor_gain(signal: f64, sprint_threshold: f64, floor: f64) -> f64 {
    if signal > sprint_threshold { 1.0 } else { floor }
}
```

Four regression tests in `symtropy-consciousness-physics::safety`
lock in the contract: strict-inequality sprint trigger, exact-floor
at boundary, zero-floor recovering binary mapping, two-level-only
output. The primitive has no dependency on Φ-as-such — its first
argument is an arbitrary `signal` — and any scalar correlate that
discriminates "confident" from "uncertain" cognitive state would
plug in equally well.

**The claim strengthens post-wiring.** Pre-FEP-wiring (commit
`≤6517226491`), SprintFloor and Recalibrated both produced 1.00 ±
0.00 cyc/100 s — the middle tiers in Recalibrated were
"decoration." Post-wiring, the middle tiers actively break the
supervisor by allowing gain = 0 states that the FEP-modulated Φ
distribution regularly hits. **The non-zero FLOOR isn't minimal —
it's load-bearing.**

## §6. The S_p Sweep (headline result)

Extending the §4 harness by sweeping ISO SSM's protective distance
parameter `S_p`, with N = 30 paired Adaptive/ISO trials and N = 10
Φ-SprintFloor trials per point (`sp_sweep_n30_post_wiring.sh`,
~60 min wall per sweep):

| S_p   | ISO cyc     | Φ-SprintFloor cyc | Φ vs ISO     | 95 % CI         |
|-------|-------------|-------------------|--------------|-----------------|
| 0.5 m | 7.00 ± 0.00 | 0.80 ± 0.42       | -88.6 %      | [-92.3, -84.8]  |
| 1.0 m | 7.00 ± 0.00 | 0.80 ± 0.42       | -88.6 %      | [-92.3, -84.8]  |
| 2.0 m | 3.30 ± 3.10 | 0.80 ± 0.42       | -75.8 %      | [-83.7, -67.8]  |
| 2.25m | 1.70 ± 2.28 | 0.80 ± 0.42       | -52.9 %      | [-68.3, -37.6]  |
| 2.5 m | 0.47 ± 0.94 | 0.80 ± 0.42       | **+71.4 %**  | [+15.4, +127.4] |
| 3.0 m | 0.00 ± 0.00 | 0.80 ± 0.42       | catastrophic | —               |

ISO SSM is bimodal: full-throughput or dead-arm, depending on
whether `S_p` fits within the human-motion envelope. Below the
bimodal transition it wins by ~89 %; above it, it loses
catastrophically. Φ-SprintFloor is flat at ~0.8 cyc/100 s across
the entire sweep — by construction, the supervisor doesn't know
about `S_p`. **The crossover at S_p = 2.5 m is the paper's
headline**: Φ-SprintFloor produces ~70 % more throughput than
ISO SSM, with 95 % CI excluding zero; at S_p ≥ 3 m, SSM is
dead-arm while SprintFloor is still completing cycles.

The reframing is *graceful degradation under epistemic uncertainty
about the human-motion envelope*. Regulators mandating conservative
`S_p` under epistemic uncertainty will drive SSM-based cobots to
zero throughput; a continuous Φ-gated supervisor survives because
its failure mode is graceful (reduced but non-zero throughput) rather
than catastrophic (zero).

## §7. Dual-channel Hazard Composition

Φ is a single-channel supervisor and is not ISO 13849 compliant for
hazardous actions on its own. The surgical-demo's cautery interlock
(`bcd80ef6aa` + `6773fa2a92`) is the reference implementation of the
composition pattern we recommend: pair Φ with a Φ-*independent*
hardware hard-limit gate; the hazardous action (electrosurgical
energy delivery) fires only when *both* channels approve.

`hardware_cautery_gate(state)` implements an ISO-13849-style hardware
channel whose inputs are strictly physical: tool-tip distance to
critical structures must exceed 5 mm, tip force must remain below
2 N, and the tool tip must be within a predefined working volume.
These conditions are independent of Φ. Φ-SurgicalSafetyLevel
authorizes cautery in `FullControl` mode only; in `Reduced`, `Freeze`,
or `Retract`, cautery is inhibited regardless of hardware-channel
state.

The 11 regression tests in `surgical-demo` lock in the AND-gate
invariant: cautery fires iff (Φ = FullControl) AND (hardware gate
approves). A diverse-redundancy reviewer can verify that a
common-mode failure in the consciousness pipeline (stuck-high Φ)
still cannot fire cautery by itself.

This pattern generalizes to any single-point-of-failure hazardous
action. It does not require accepting Φ as ISO 26262 safety-rated; it
uses Φ only as a *disabling vote* alongside a certified hardware
channel.

## §8. Cross-platform applicability

The `sprint_floor_gain` primitive is wired into six platforms as
demo plugins: flight, vehicle, AUV, helicopter, humanoid-demo, plus
the manipulator-benchmark itself. We ran three experiments to
characterize cross-platform behavior.

**Platform-invariant Φ band under legacy observations.** With
`PT_PLATFORM_OBS=0` (the default in `phi_trace.rs`), all six platforms
produce identical Φ distributions [0.103, 0.145] because
`RoboticAgent::tick`'s `ConsciousnessInputs` construction was
originally observation-agnostic — the observation vector was passed
to `fep.perceive()` but the return was discarded. This was a design
oversight that the `996750d12b` FEP-wiring commit closed.

**Per-platform Φ divergence under platform-aware observations**
(Figure 4). With `PT_PLATFORM_OBS=1`, each platform receives a
hand-crafted observation stream reflecting its sensor dynamics
(quadrotor: altitude + attitude + wind gusts; vehicle: speed +
slip + friction; etc.). Fraction of frames above the shared legacy
threshold SPRINT_THRESHOLD = 0.125 now varies dramatically:
quadrotor and helicopter 0 %, manipulator 25 %, vehicle 33 %,
humanoid 69 %, AUV 80 %. The platforms no longer cluster; their Φ
distributions span the full empirical band width. This empirically
establishes that per-platform calibration is *necessary*, not
cosmetic.

**Per-platform threshold calibration.** The §8 table records p50
and applied SPRINT_THRESHOLD for each of the six platforms. Values
span 0.101 (vehicle) to 0.130 (humanoid, AUV) — a **30 % spread**.
At each platform's calibrated threshold, sprint rates land in
[45.1 %, 64.8 %] — close to the design target of 50 %. The
`symtropy calibrate <platform>` CLI (commit `e75542bd95`) automates
this tuning as a one-liner.

**Cross-platform paired replication** (Figure 5). Running
paired TierGate vs SprintFloor at each platform's calibrated
threshold, N = 30 per platform, yields:

| Platform    | Threshold | Advantage   | 95 % CI           |
|-------------|-----------|-------------|-------------------|
| AUV         | 0.130     | **+89.3 %** | [+85.1, +93.6]    |
| Quadrotor   | 0.110     | **+125.8 %**| [+125.1, +126.5]  |
| Helicopter  | 0.110     | **+171.2 %**| [+170.2, +172.1]  |
| Manipulator | 0.114     | **+202.6 %**| [+202.3, +202.9]  |
| Humanoid    | 0.130     | **+204.2 %**| [+201.8, +206.6]  |
| Vehicle     | 0.101     | **+230.5 %**| [+230.5, +230.5]  |

**Mean advantage across 6 platforms: +170.6 %. Every CI
excludes zero.** This is the strongest cross-platform
evidence we can produce in simulation.

## §9. Discussion and Limitations

**Φ is not a certified safety layer.** The SOTIF framing positions
it as a *triggering-condition monitor*, not an IEC 61508 /
ISO 26262 safety function. Single-point-of-failure hazards still
require a Φ-independent hardware interlock (§7).

**Simulation-only.** Every result in this paper is from a
simulator. Real-world sensor noise, actuation delay, contact
dynamics, and hardware-specific failure modes are absent. §9.1
sketches a hardware-validation path on the Bitcraze Crazyflie 2.1.

**No RL baseline.** We compare to ISO/TS 15066 SSM, a rule-based
baseline. We do *not* compare to a trained SAC / TD3 / MPC
controller on the same task. A reviewer may fairly ask whether
SprintFloor beats a well-tuned learned controller; our answer is
that this is future work. The SprintFloor pattern is a *supervisor*,
operating above whichever controller the platform uses; it is
complementary to, not a replacement for, learned control.

**Untrained HDC-LTC projections.** The `thought_to_torque` example
(commit `f1285ce1e6`) shows that our HDC-LTC controller runs at
its initialization weights, producing bias-level motor outputs.
The `train_step` BPTT path exists but was not exercised for the
paper's benchmarks. The paper's supervisor claim is robust to this
— the 5-variant comparison isolates gating-policy from controller
— but a fully trained controller would strengthen the
"thought-to-torque" architectural story.

**Hand-crafted observation generators.** Figure 4 and Figure 5
use per-platform observation streams written as Rust closures
inside `phi_trace.rs` and `paired_benchmark_live.rs`. These
approximate each platform's demo scenario but are not the actual
physics-simulated observations. Hooking into each demo's full Bevy
simulator would produce on-manifold observations; the hand-crafted
streams are a scope-bounded compromise.

**Crossover CI width.** The S_p = 2.5 m crossover advantage of
+71.4 % has a 95 % CI of [+15.4, +127.4] — the lower bound is
positive but thin. A reviewer concerned about statistical power
could reasonably ask for N = 100 or N = 1000; at N = 30 our CI
excludes zero but only barely at the low end.

### §9.1 Hardware-validation plan (Crazyflie 2.1 path)

The cheapest path to a real-world validation uses a Bitcraze
Crazyflie 2.1 (27 g, ≥ 300 Hz attitude, ~USD 310 hardware cost
plus the Crazyradio PA). The Crazyflie's mass + rotor-lag constants
match our `SimplePhysicsSimulator` defaults within 15 %; its
onboard-to-radio latency budget (~5 ms) fits the 25 Hz cognitive
tick comfortably.

The proposed bring-up has four steps:

1. Port the flight controller's 500 Hz reflex loop to Crazyflie's
   on-board firmware (unchanged; already there).
2. Port the 25 Hz cognitive tick as a radio-tethered ground-station
   process (unchanged; just needs the radio bridge).
3. Tune `SPRINT_THRESHOLD` and `FLOOR_GAIN` against hover +
   nudge-rejection data (reset → push → observe) until the in-the-
   air motor gain matches the in-sim trace within telemetry
   granularity. Same recipe as `symtropy calibrate` but with real
   flight data.
4. Re-run the S_p-analogous sweep with a styrofoam "human" on a
   tethered pendulum; measure cycles completed per minute at three
   pendulum-envelope sizes.

Expected timeline: 2-4 weeks, USD 310 hardware, one grad-student-
week of integration effort. This is future work, not part of the
current paper's claims.

### §9.2 Cross-platform live-Φ supplement (already reported above)

Figure 5 + the `paired_benchmark_live` data (`a8d965773a`) is the
strongest single-paper replication we can produce in simulation.
See §8 for the full six-platform table.

## §10. Conclusion

The current robotics-safety literature offers two extremes: rule-based
envelopes with legible certification properties and catastrophic
tail failure modes, and learned end-to-end policies with open
certification status. Between them sits a design space for *continuous
supervisors* — scalar signals that modulate a certified envelope's
commanded authority according to runtime model confidence. This
paper contributes:

- A minimal 2-part sufficient condition for such a supervisor
  (sprint threshold + non-zero floor), validated against four
  alternative mapping shapes that all collapse under observation-
  driven Φ (§5).
- A reproducible Monte Carlo S_p sweep showing where this supervisor
  wins (conservative S_p, epistemic uncertainty regime) and where it
  loses (tight standard cobot regime), with the reframing that
  *graceful degradation* is the ISO 21448-relevant property
  (§6).
- A dual-channel hazard composition pattern that preserves ISO 13849
  diverse-redundancy semantics (§7).
- A six-platform cross-platform replication in simulation (§8,
  Figure 5) with mean advantage +170.6 % and every CI excluding
  zero.
- Open-source implementations of all ten platform demos, the
  `sprint_floor_gain` library primitive, the `symtropy calibrate`
  CLI, and the full benchmark harness.

The paper does *not* claim that Φ measures consciousness, that our
supervisor is a certified safety layer, or that we beat a trained
RL controller. Each of those is explicitly deferred as future work,
and the SOTIF framing is load-bearing for the claims we do make.

We expect the most immediately actionable contribution is the
`sprint_floor_gain` primitive plus the `symtropy calibrate <platform>`
one-liner: any production cobot / drone / autonomous vehicle
integrator with an aggregated cognitive correlate available at
runtime (not necessarily Φ — any scalar that discriminates confident
from uncertain state) can apply the recipe in a working afternoon
and measure the graceful-degradation benefit in their own S_p
sensitivity analysis.

---

## Reproducibility

All empirical results are reproducible from `main` via the commit
map in `papers/phi-gated-safety/README.md`. Each figure has a one-
line reproducer; each table has a raw-data CSV; the CLI
subcommand `symtropy calibrate <platform>` operationalizes the §8
per-platform calibration recipe.

## License

Code: AGPL-3.0-or-later (matches the rest of the symtropy / symthaea
robotics stack). Paper prose: CC-BY-4.0.
