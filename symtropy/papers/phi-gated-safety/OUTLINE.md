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

## Abstract sketch (draft 0)

We present an empirical study of consciousness-gated motor authority
across ten heterogeneous robotic platforms. A scalar Φ ∈ [0,1],
computed from HDC prediction error and an Active Inference pipeline,
modulates each platform's motor command. Across five Φ→gain mapping
variants in a Monte Carlo cobot benchmark (N=5 trials × 100 s each, 7-DOF
industrial manipulator with sinusoidal human obstacle, paired
comparison vs ISO/TS 15066 Speed-and-Separation Monitoring), we find:
**(i)** naive threshold-based mappings using the default
`SafetyTier` configuration produce 0 cycles/100 s (dead-arm failure),
**(ii)** the minimal sufficient condition for a functional mapping is
a *sprint threshold* matched to the empirical Φ band plus a
*crawl-rate floor* above the IK convergence limit — implemented in
two lines as `if Φ > sprint { 1.0 } else { floor }`, **(iii)** this
mapping loses throughput to ISO SSM in the standard cobot regime
(S_p ≈ 1 m) by 85.7 % but **wins by +150 %** at S_p = 2.5 m and
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

    |     variant      | cyc/100s    | vs ISO   |
    |------------------|-------------|----------|
    | Default tiers    | 0.00 ± 0.00 | -100.0 % |
    | Continuous       | 0.60 ± 0.55 |  -91.4 % |
    | Clamped-linear   | 0.80 ± 0.45 |  -88.6 % |
    | Recalibrated     | 1.00 ± 0.00 |  -85.7 % |
    | SprintFloor      | 1.00 ± 0.00 |  -85.7 % |
    | ISO SSM          | 7.00 ± 0.00 | baseline |

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
- `sprint_floor_gain(phi, sprint_phi, floor)` library primitive
  (commit `52e3fb710f`), 4 regression tests lock the contract.

### §6. The S_p sweep (headline result)
- Same harness, sweep ISO's protective distance:

    | S_p   | ISO cyc     | Φ cyc        | Φ vs ISO        |
    |-------|-------------|--------------|-----------------|
    | 0.5 m | 7.00 ± 0.00 | 1.00 ± 0.00  |  -85.7 %        |
    | 1.0 m | 7.00 ± 0.00 | 1.00 ± 0.00  |  -85.7 %        |
    | 2.0 m | 2.60 ± 3.58 | 1.00 ± 0.00  |  -61.5 %        |
    | 2.25m | 1.60 ± 2.30 | 1.00 ± 0.00  |  -37.5 %        |
    | 2.5 m | 0.40 ± 0.89 | 1.00 ± 0.00  | **+150.0 %**    |
    | 3.0 m | 0.00 ± 0.00 | 1.00 ± 0.00  | catastrophic    |

- ISO is bimodal: full-throughput or dead-arm, depending on whether
  S_p fits within the human motion envelope. Φ is flat at ~1 cyc/100 s
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
- `sprint_floor_gain` primitive now wired into flight-demo (`8d61e348d9`)
  as proof of transfer. Per-platform calibration required: each
  platform's Φ band may drift because the observation-vector channels
  differ.
- Table: 10 platforms → which Φ-role → calibration status.

### §9. Discussion & limitations
- Φ is NOT a certified safety layer. SOTIF frame: it's a
  triggering-condition monitor.
- Benchmark is simulation-only. Hardware validation (Crazyflie for
  the flight path, Franka FR3 for the manipulator path) is the
  obvious follow-up.
- Paired-trial N=5 is thin; re-running N=30 tightens CIs.
- The `MasterConsciousnessEquation`'s monotonic compressive output
  [0.099, 0.145] band is a source of fragility — the sprint threshold
  is close to the empirical max. Widening the equation's dynamic range
  at the source would make thresholds less sensitive.

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

- [ ] Render Figure 1 (Φ trace): 40 s of TRACE lines → matplotlib
      line chart with SafetyTier thresholds overlaid.
- [ ] Render Figure 2 (S_p sweep bar): 6 groups × 3 bars each
      (ISO / Adaptive / Φ-SprintFloor).
- [ ] Re-run §4 with N=30 trials to tighten CIs (currently N=5,
      normal approx strained).
- [ ] Run §6 sweep with N=30 trials (currently N=5).
- [ ] Cross-reference all commit hashes against the main branch —
      ensure none were rebased.
- [ ] Add one paragraph on hardware-validation plan (Crazyflie 2.1,
      cflib-rs, radio stack) — strengthens §9.
- [ ] Dial abstract down to 150 words.
