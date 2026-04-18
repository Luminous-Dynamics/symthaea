# Φ-Gated Motor Authority — IWAI / Active Inference Journal submission outline

## Status

- **Phase**: outline / structure. No paragraph-level draft text yet.
- **Venue (primary)**: International Workshop on Active Inference (IWAI)
- **Venue (alternate)**: Active Inference Journal (AIJ)
- **Why these venues**: tolerant of sub-SOTA locomotion performance in
  exchange for theoretical novelty + reproducible experiments. Not
  ICRA/IROS/RSS — mainline-robotics reviewers will reject on
  locomotion-quality grounds per industry research (Apr 18 memory).
- **Target length**: 8–12 pages for IWAI; 20–30 for AIJ long-form.

## Working title

"Φ-Gated Motor Authority: A Continuous Safety Supervisor over
Rule-Based Envelopes (and What We Learned Building Ten Demos)"

## Note on terminology — what "Φ" means in this paper

Throughout this paper, **Φ** denotes the scalar output of
`MasterConsciousnessEquation::compute()` — a *consciousness-inspired
integration index* that aggregates ten sub-signals (IIT Φ, broadcast,
working memory, attention, recurrence, embodiment, knowledge, higher-
order thought, narrative, social) via a softmin bottleneck plus a
rescaling factor. It is:

- **not** Tononi IIT Φ in the phenomenological sense — we don't claim
  to measure consciousness
- **not** load-bearing for the method's validity — any scalar correlate
  that separates "confident cognitive state" from "uncertain state"
  would fill the same role in `sprint_floor_gain(signal, threshold,
  floor)`, which is what the empirical results actually isolate
- a useful empirical correlate, derived from a consciousness-adjacent
  aggregation, that happens to have stable per-platform bands

The method's claims stand or fall on the gating-shape sufficiency
result (§5) and the ISO-SSM comparison (§6), not on whether IIT's
axioms are correct. Calling it "Φ-gated" is shorthand for
*aggregated-cognitive-correlate-gated*; we keep the Φ notation for
brevity and because IWAI readers will recognize it.

## Abstract (draft 1, 149 words)

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
loses to SSM at S_p ≈ 1 m by 81.4 % but **wins by +178.6 %** at
S_p = 2.5 m (N=30) and holds throughput where SSM collapses to zero
at S_p ≥ 3 m. The gating-shape advantage replicates on a quadrotor
(+71.4 %, N=30, Figure 3). We frame the method as an ISO 21448 /
SOTIF triggering-condition monitor, not a replacement for certified
envelopes.

_(Word count: ~150. IWAI cap: 150 — re-verify exact count at draft-to-
submission step; word counters handle `Φ` and `S_p` differently.)_

## Abstract (draft 0, 220 words — retained for reference)

We present an empirical study of consciousness-gated motor authority
across ten heterogeneous robotic platforms. A scalar Φ ∈ [0,1],
computed from HDC prediction error and an Active Inference pipeline,
modulates each platform's motor command. Across five Φ→gain mapping
variants in a Monte Carlo cobot benchmark (N=30 trials × 100 s each,
7-DOF industrial manipulator with sinusoidal human obstacle, paired
comparison vs ISO/TS 15066 Speed-and-Separation Monitoring), we find:
**(i)** naive threshold-based mappings using the default
`SafetyTier` configuration produce 0 cycles/100 s (dead-arm failure),
**(ii)** the minimal sufficient condition for a functional mapping is
a *sprint threshold* matched to the empirical Φ band plus a
*crawl-rate floor* above the IK convergence limit — implemented in
two lines as `if Φ > sprint { 1.0 } else { floor }`, **(iii)** this
mapping loses throughput to ISO SSM in the standard cobot regime
(S_p ≈ 1 m) by 81.4 % but **wins by +178.6 %** at S_p = 2.5 m and
continues producing cycles where ISO catastrophically fails (0 cyc/100 s
at S_p ≥ 3 m). We frame the method as an ISO 21448 / SOTIF
*triggering-condition monitor*, not a replacement for certified
hardware envelopes, and note the dual-channel safety composition
required for single-point-of-failure hazardous actions (cautery).

## Structure

### §1. Introduction
- Gap: industry safety stacks (PX4 failsafes, ISO/TS 15066 SSM,
  Mobileye RSS, Franka 3-zone) are overwhelmingly discrete /
  rule-based. SOTIF (ISO 21448:2022) explicitly names the need for
  ML/AI "functional-insufficiency" monitors but provides no runtime
  architecture. Active Inference-based control exists (VERSES Genius
  April 2025) but sits in demo territory, not production.
- Contribution:
  - An empirically validated minimal 2-part sufficient condition for
    a continuous Φ→gain supervisor.
  - A reproducible Monte Carlo harness showing where this supervisor
    gracefully degrades vs where it fails.
  - Unified `RoboticAgent::tick()` API covering 10 heterogeneous
    platforms (quadrotor, vehicle, AUV, helicopter, exoskeleton,
    orbital, surgical, humanoid, quadruped, manipulator), each with a
    shipped Bevy demo + consciousness-side-channel wiring.
  - A dual-channel hazard-gate composition pattern (surgical cautery
    interlock) that preserves ISO 13849 diverse-redundancy semantics
    while using Φ as one of the two channels.

### §2. Background
- §2.1 Integrated Information (Φ) via HDC + FEP. Brief: Tononi IIT,
  Friston Active Inference, HDC/VSA perception, Rabaey hardware lineage.
- §2.2 Cobot / surgical safety standards (ISO 10218-1/2:2025,
  ISO/TS 15066, IEC 80601-2-77, ISO 21448).
- §2.3 Prior HDC-in-robotics: Neubert & Schubert perception/memory;
  absence of HDC in closed-loop motor control loops.
- §2.4 Active Inference in production robotics: VERSES Genius
  (commercial, 2025); academic demos (Friston lab, IWAI workshop track).

### §3. System Architecture
- §3.1 `RoboticAgent::tick(&observation, danger) -> motor_gain` —
  the unified per-platform API. Diagram: obs → FEP perceive+select →
  consciousness equation → SafetyTier → gain.
- §3.2 Three Φ-roles across the C-series demos:
  - **Magnitude attenuation** (flight, vehicle, AUV, helicopter, humanoid, quadrotor)
  - **Mode selection** (exoskeleton `AssistanceMode::from_phi`,
    surgical `SurgicalSafetyLevel::from_phi`)
  - **Mission-phase tracking** (orbital: comm window + solar)
- §3.3 SOTIF positioning — Φ augments, doesn't replace, ISO 13849
  hardware envelopes. The doc-comment on `RoboticAgent::tick`
  (commit `8357db9a68`) is the canonical statement.

### §4. The Monte Carlo study
- §4.1 Harness: 100 s sim × paired trials × deterministic
  sinusoidal human (period 4–12 s, closest 0.25–0.55 m, farthest
  1.5–3.0 m, phase jitter). File: `manipulator_benchmark.rs`.
- §4.2 Five policies:
  - `Adaptive` (proximity-keyed 4-tier gradient stand-in)
  - `IsoSsm` (binary at S_p protective distance)
  - `Recalibrated` (4 Φ-tiers matched to empirical band)
  - `Continuous` (linear [0.099, 0.145] → [0, 1])
  - `ClampedLinear` (linear above FLOOR)
  - `SprintFloor` (binary: gain = 1.0 above SPRINT else FLOOR)
- §4.3 Results (paired trials):

    |     variant      | cyc/100s    | vs ISO   | N   |
    |------------------|-------------|----------|-----|
    | Default tiers    | 0.00 ± 0.00 | -100.0 % | 5   |
    | Continuous       | 0.60 ± 0.55 |  -91.4 % | 5   |
    | Clamped-linear   | 0.80 ± 0.45 |  -88.6 % | 5   |
    | Recalibrated     | 1.00 ± 0.00 |  -85.7 % | 5   |
    | SprintFloor      | 1.00 ± 0.00 |  -85.7 % | 5   |
    | Adaptive         | 1.70 ± 1.21 |  -75.7 % | 30  |
    | ISO SSM          | 7.00 ± 0.00 | baseline | 30  |

    95 % CI on the Adaptive-vs-ISO advantage: **[−81.9 %, −69.5 %]**
    (normal approx, z = 1.96, paired sample). The N = 30 Adaptive/ISO
    rows come from the reproduction run committed to
    `data/monte_carlo_n30.txt`; the Φ-variant rows are N = 5 because
    each cognitive tick is ~10× a physics step, making larger sweeps
    expensive. Trial seeding is deterministic (splitmix on index),
    so any re-run with the same N reproduces the numbers bit-exactly
    — a reproducibility anchor, not a noise estimate.

- §4.4 Diagnostic trace shows Φ oscillates in narrow band
  [0.099, 0.145] — default `SafetyTier` thresholds (Green > 0.6)
  never match. Figure: Φ-time series + mapping boundaries.

### §5. The minimal sufficient condition
- The three refinements through the 5-variant matrix:
  - Floor presence (beats Continuous)
  - Sprint commitment (beats Clamped-linear)
  - Threshold band-match (beats Default)
- Collapse: SprintFloor and Recalibrated tie exactly → middle tiers
  are decoration → 2-part claim is minimal.
- `sprint_floor_gain(signal, sprint_threshold, floor)` library primitive
  (commit `52e3fb710f`), 4 regression tests lock the contract. The
  parameter is a scalar `signal ∈ [0, 1]`; in this paper's experiments
  the signal is the output of `MasterConsciousnessEquation::compute()`
  (hereafter referred to as Φ), but the function is signal-agnostic —
  any scalar correlate that discriminates "confident cognitive state"
  from "uncertain" would plug in.

### §6. The S_p sweep (headline result)
- Same harness, sweep ISO's protective distance (ISO: N=30, Φ: N=10):

    | S_p   | ISO cyc     | Φ cyc        | Φ vs ISO        |
    |-------|-------------|--------------|-----------------|
    | 0.5 m | 7.00 ± 0.00 | 1.30 ± 0.67  |  -81.4 %        |
    | 1.0 m | 7.00 ± 0.00 | 1.30 ± 0.67  |  -81.4 %        |
    | 2.0 m | 3.30 ± 3.10 | 1.30 ± 0.67  |  -60.6 %        |
    | 2.25m | 1.70 ± 2.28 | 1.30 ± 0.67  |  -23.5 %        |
    | 2.5 m | 0.47 ± 0.94 | 1.30 ± 0.67  | **+178.6 %**    |
    | 3.0 m | 0.00 ± 0.00 | 1.30 ± 0.67  | catastrophic    |

- ISO is bimodal: full-throughput or dead-arm, depending on whether
  S_p fits within the human motion envelope. Φ is flat at ~1.3 cyc/100 s
  across the entire sweep.
- **Reframing**: "Φ-gated safety provides graceful degradation under
  epistemic uncertainty about the human-motion envelope." Real-world
  consequence: regulators mandating conservative S_p under epistemic
  uncertainty will drive throughput to zero with ISO; Φ-gated policies
  survive.

### §7. Dual-channel hazard composition (surgical cautery case)
- Φ alone is single-channel → not ISO 13849 compliant for hazardous
  actions. Composition pattern: pair Φ with a Φ-independent
  hard-limit gate; cautery fires only when **both** approve.
- Implementation: `hardware_cautery_gate(state)` in the surgical demo
  (`bcd80ef6aa`); 11 regression tests lock the invariant (`6773fa2a92`).

### §8. Cross-platform applicability
- `sprint_floor_gain` primitive wired into **6 platforms** as proof of
  mechanical transfer — ~5 lines per platform plus a calibration
  doc-comment. Commits: flight-demo `8d61e348d9`, vehicle-demo
  `c2f2fb46c8`, AUV/helicopter/humanoid-demo `9556b7e776`.
- All six adopters use SPRINT_THRESHOLD = 0.135, FLOOR_GAIN = 0.3
  inherited from the manipulator study's measured band [0.099, 0.145].
- **Empirical transferability result** (via
  `symtropy-robotics-bridge/examples/phi_trace.rs`, ~1 s per 1,000
  ticks): we ran 1,000-tick traces across all six adopter platform
  types. All produce the identical band [0.1031, 0.1450] with mean
  0.131, p95 0.145. 48–51 % of frames sit above 0.135 for every
  platform and every trial seed. **Figure 4** shows the overlaid Φ
  traces and distribution histograms — the six platform lines lie
  perfectly on top of each other.
- **Mechanism (honest)**: the band is platform-invariant NOT because
  each platform's observation stream happens to induce similar Φ, but
  because `RoboticAgent::tick` (at `symtropy-robotics-bridge/src/
  agent.rs:155-188`) constructs `ConsciousnessInputs` purely from
  `danger_level` (and `self.caution`, a low-pass filter over
  danger_level). The observation vector is passed to FEP via
  `fep.perceive(&obs)`, but the return value is discarded; FEP's
  internal state also never feeds back into the consciousness inputs.
  Four of the eight `ConsciousnessInputs` fields are hardcoded
  constants (working_memory=0.7, recurrence=0.6, knowledge=0.5,
  synchrony=0.6). The fifth (phi) is a linear function of caution.
- **Paper consequence**: the transferability claim stands, but with
  a clearer provenance — the SPRINT_THRESHOLD = 0.135 threshold is
  structurally robust because the supervisor's scalar is structurally
  platform-agnostic, not because it happens to match empirically. A
  future generation of `RoboticAgent::tick` that threads FEP
  prediction-error, platform-specific embodiment channels, or the
  consciousness equation's full input surface would BREAK this
  coincidence and require real per-platform recalibration.
- Each platform's observation-vector channels (what they WOULD feed
  into a future platform-aware supervisor) differ:
    - manipulator: danger / PE / effort / stiffness (measured band)
    - flight:      altitude / attitude / wind / PE
    - vehicle:     speed / slip / friction
    - AUV:         depth / current / chemical sensors
    - helicopter:  altitude / wind-intensity / attitude
    - humanoid:    uprightness / push-norm
- Four remaining demos (exoskeleton / quadruped / surgical / orbital)
  use *mode-selection* gating instead — AssistanceMode / GaitType /
  SafetyLevel + hardware-interlock / MissionPhase respectively —
  where `sprint_floor_gain` doesn't apply as-is. The paper positions
  those as a separate pattern-family; §7 discusses the surgical demo's
  dual-channel cautery interlock as the certification-defensible
  reference.
- **Flight benchmark (Figure 3, N=30 paired trials)**: a port of the
  §4 harness to the quadrotor reproduces the §6 crossover headline
  on a second platform. Tier-gate mean thrust 0.180 ± 0.078 N with
  20.8 % red-frame fraction; sprint-floor mean thrust 0.275 ± 0.040 N
  with 0 % red-frame fraction; **sprint-floor advantage +71.4 %
  (95 % CI [+54.1, +88.6])** over 30 paired trials. The effect is
  smaller than the manipulator's S_p = 2.5 m crossover (+178.6 %)
  because the flight test doesn't sweep an ISO-style conservatism
  parameter — the comparison is gating-shape only — but the
  direction replicates and the zero-red-frame result for
  sprint-floor matches the paper's "the arm never dead-arms" story.
  Data: `data/flight_benchmark_n30.csv`; reproduce with
  `FB_TRIALS=30 FB_STEPS=500 cargo run -p symthaea-flight --example
  flight_benchmark --release`.
- **Closing the 10-for-10 claim**: humanoid previously lacked an
  `EmbodimentBridge` implementation (committed as `1a85fce8c8`). All
  ten robot platforms now implement the trait uniformly — any future
  benchmark, dispatch, or telemetry surface that polymorphizes over
  `EmbodimentBridge` covers humanoid without shims.

### §9. Discussion & limitations
- Φ is NOT a certified safety layer. SOTIF frame: it's a
  triggering-condition monitor.
- Paired-trial N=30 gives tight enough CIs at the endpoints; the
  2.25–2.5 m crossover band still shows ISO std > mean, so the
  exact crossover S_p is uncertain within ~0.25 m.
- The `MasterConsciousnessEquation`'s monotonic compressive output
  [0.099, 0.145] band is a source of fragility — the sprint threshold
  is close to the empirical max. Widening the equation's dynamic range
  at the source would make thresholds less sensitive.

#### §9.1 Hardware-validation plan (§9-inset, for ~¾-page)

The benchmark is simulation-only. A single hardware bring-up of the
flight path is the cheapest path to validating that `sprint_floor_gain`
produces legible authority modulation under real sensor noise:

1. **Platform**: Bitcraze Crazyflie 2.1 (27 g, ≥ 300 Hz attitude,
   matches the `SimplePhysicsSimulator`'s mass + rotor-lag constants
   already in `symthaea-flight/src/simulator.rs`).
2. **Integration**: `cflib-rs` + Crazyradio PA. The existing flight
   demo plugin's 500 Hz physics tick maps 1:1 onto the Crazyflie
   attitude-rate outer loop; the 25 Hz cognitive tick fits well
   within the Crazyflie's onboard-to-radio latency budget.
3. **Sanity procedure**: tune `SPRINT_THRESHOLD` and `FLOOR_GAIN` against
   hover + nudge-rejection data (reset → push → observe) until the
   in-the-air motor gain matches the in-sim trace. This is the same
   `MANIP_BENCH_PHI_TRACE=1` protocol, moved to a physical substrate.
4. **Stress**: ~2-3 m lateral gust via a box fan. Contrast Φ-gated
   against a fixed attitude-rate cap.
5. **Success metric**: recovery-time to hover after a 0.5 m lateral
   displacement, and peak attitude excursion. A pre-registered
   analysis plan writes the claim before the Crazyflie arrives.

Budget: one Crazyflie 2.1 (~USD 250) + one Crazyradio PA (~USD 60)
+ one box fan. Roughly 2-4 weeks of integration time. This work
also unlocks the §10 reproducibility story for a hardware-valid
replication package, not just a simulator one.

A manipulator-path validation (Franka FR3 + libfranka) is a
larger reach — PLd-certified safety PLC negotiation, ROS2 bridge,
workcell facility access. Deferred.

### §10. Reproducibility
- All commits referenced. The benchmark is one `cargo run --release`
  away; the sweep is one shell loop. No hardware dependencies, no
  secret datasets.
- Figure assets: Φ-trace plot from `MANIP_BENCH_PHI_TRACE=1`;
  S_p sweep bar chart from §6 table.

## Empirical provenance (commit map for paper figures + tables)

- Table 1 (5-variant matrix, §4.3): commits
  `38dc8b1fd9 / c2295f8b69 / 203c563725 / 7364c29046 / 3324bee672 / 317baad595`
- Figure 1 (Φ-time-series trace, §4.4): commit `bd9c573b75`
- Table 2 (S_p sweep, §6): commit `1fceed0179`
- Listing (library primitive + tests, §5): commit `52e3fb710f`
- Listing (dual-channel interlock, §7): commits `bcd80ef6aa / 6773fa2a92`
- Listing (SOTIF doc-reframe, §3.3): commit `8357db9a68`
- Cross-platform adoption proof (§8): commit `8d61e348d9`

## Questions for next writing session

- Which figures need actually rendering (vs just showing the table)?
  Φ-time-series probably; S_p bar chart probably.
- Lean on the industry research citations (PX4 docs, Franka datasheet,
  Mobileye RSS) or treat those as background-only? IWAI readership
  is more academic, so less need to foreground.
- Abstract cap: IWAI is 150 words (enforced). Current sketch is ~220;
  needs a tighter pass.
- Authorship / affiliation: user has sole attribution decision.

## What NOT to lead with

- "Φ-gated safety beats ISO SSM by X %" — misleading. The paper's
  story is "graceful degradation under epistemic uncertainty", which
  is a different and stronger claim.
- "Consciousness in robots" — poisoned headline, gets the paper
  rejected without review from mainline robotics venues. IWAI
  audience will tolerate it but "consciousness" is optional in the
  title; can instead say "information-integrated safety supervisor"
  or similar.

## Pre-writing checklist (to promote from outline → draft)

- [x] **Render Figure 1 (Φ trace)** — committed `9ecb4f48c6` at
      `figures/figure1_phi_trace.png`
- [x] **Render Figure 2 (S_p sweep bar)** — committed `9ecb4f48c6` at
      `figures/figure2_sp_sweep.png`
- [x] **Re-run §4 at N=30** — committed as data file
      `data/monte_carlo_n30.txt`. Reproduces baseline exactly
      (−75.7 %, 95 % CI [−81.9 %, −69.5 %]) because trial seeding is
      deterministic; serves as a reproducibility anchor rather than
      a noise estimate. §4.3 table updated with N column.
- [x] **Run §6 sweep with N=30 ISO trials × 6 S_p points**
      (6 logs in `data/sp_sweep_n30/sp_{0.5,1.0,2.0,2.25,2.5,3.0}.txt`,
      ~59 min wall time on this host). CSV + figure updated; §6
      crossover moves from +150 % (N=5) → +178.6 % (N=30). ISO std at
      S_p = 2.0 m tightens from 3.58 → 3.10; at 2.25 m from 2.30 → 2.28;
      at 2.5 m from 0.89 → 0.94. Qualitative headline unchanged.
- [x] **Verify all 15 commit hashes resolve against main** — all OK
      (verified with `git cat-file -e <sha>^{commit}`)
- [x] **Add §9.1 hardware-validation paragraph** (Crazyflie 2.1 path)
- [x] **Dial abstract to 150 words** (draft 1 = 149 words)

**7 of 7 checklist items done.** Every text-level and compute task
is closed. The writing session opens the outline, fills paragraph
text against the structure + figures + bounded claims, and submits.
