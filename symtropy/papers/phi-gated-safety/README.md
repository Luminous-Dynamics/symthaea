# Φ-gated motor authority — paper artifacts

Supporting code, data, and figures for the IWAI / Active Inference Journal
paper on consciousness-gated motor authority across heterogeneous robot
platforms. The paper establishes a minimal 2-part sufficient condition
for a Φ→gain supervisor that beats binary ISO/TS 15066 SSM under
epistemic uncertainty about the human-motion envelope, and shows the
result replicates to a second platform (quadrotor).

## Contents

### Prose
- `OUTLINE.md` — 10-section paper outline with commit-map, bounded
  claims, abstract draft (149/150 words), pre-writing checklist (7/7
  closed), and §9.1 Crazyflie hardware-validation plan. Drafting the
  full prose from this outline is a pure-text session and is not yet
  done.

### Figures
- `figures/figure1_phi_trace.png` — 40 s Φ trace showing the narrow
  empirical band [0.099, 0.145] vs the default `SafetyTier` thresholds
  (Green > 0.6 etc.) that never fire. Motivates the recalibration
  argument.
- `figures/figure2_sp_sweep.png` — Headline S_p sweep. ISO SSM cycles
  collapse 7 → 0 across S_p ∈ {0.5, 1.0, 2.0, 2.25, 2.5, 3.0} m;
  Φ-SprintFloor holds ~1.3 cyc/100s. Crossover at S_p = 2.5 m at
  **+178.6 %** (N=30, 95 % CI wide at crossover).
- `figures/figure3_flight_benchmark.png` — Second-platform replication
  on the quadrotor. TierGate 0.180 ± 0.078 N (20.8 % red-frames);
  SprintFloor 0.275 ± 0.040 N (0 % red-frames). Paired advantage
  **+71.4 %**, 95 % CI [+54.1, +88.6] at N=30.
- `figures/render_figures.py` — Pure-matplotlib renderer for all three
  figures. Dependencies via `nix-shell` (see Reproducing below).

### Data
- `data/phi_trace_40s.csv` — 1000 samples from
  `MANIP_BENCH_PHI_TRACE=1 cargo run -p symtropy-manipulator-demo
  --example manipulator_benchmark --release`. Feeds Figure 1.
- `data/sp_sweep_results.csv` — Summary table used by Figure 2
  (N=30 for ISO/Adaptive, N=10 for Φ variants).
- `data/sp_sweep_n30/` — Six raw log files from the full N=30 sweep
  (`sp_{0.5,1.0,2.0,2.25,2.5,3.0}.txt`, ~10 min each on this host).
- `data/sp_sweep_n30.sh` — Reproducer script for the full sweep.
- `data/monte_carlo_n30.txt` — §4 Monte Carlo reproduction anchor
  (baseline Adaptive vs ISO SSM, -75.7 % loss in standard regime).
- `data/flight_benchmark_n30.csv` — N=30 paired-trial flight benchmark,
  feeds Figure 3.

## Reproducing

Every empirical claim in the paper has a one-line reproducer. From the
monorepo root:

```bash
# Figure 1 — Φ trace (1000 samples, ~3 s)
cd symthaea
MANIP_BENCH_PHI_TRACE=1 MANIP_BENCH_PHI_TRIALS=1 \
  cargo run -p symtropy-manipulator-demo --example manipulator_benchmark --release \
  > ../symtropy/papers/phi-gated-safety/data/phi_trace_40s.csv 2>&1

# Figure 2 — full S_p sweep (N=30, ~60 min wall-clock)
cd symtropy/papers/phi-gated-safety/data
./sp_sweep_n30.sh

# Figure 3 — flight benchmark (N=30, ~30 s)
cd symthaea
FB_TRIALS=30 FB_STEPS=500 \
  FB_CSV=../symtropy/papers/phi-gated-safety/data/flight_benchmark_n30.csv \
  cargo run -p symthaea-flight --example flight_benchmark --release

# Render all figures
cd symtropy/papers/phi-gated-safety/figures
nix-shell -p 'python3.withPackages (ps: [ps.matplotlib ps.numpy])' \
  --run "python3 render_figures.py"
```

## Commit map

The paper references specific commits for every empirical claim. Resolve
any of these against `main` with `git cat-file -e <sha>^{commit}`:

| Commit       | Contribution                                                 |
|--------------|--------------------------------------------------------------|
| `38dc8b1fd9` | Monte Carlo manipulator benchmark — baseline −75.7 %         |
| `c2295f8b69` | Φ-policy wiring                                              |
| `bd9c573b75` | `MANIP_BENCH_PHI_TRACE` trace capture                        |
| `203c563725` | Recalibration fix — threshold refit                          |
| `7364c29046` | Continuous Φ→gain variant                                    |
| `3324bee672` | Clamped-linear Φ→gain variant                                |
| `317baad595` | SprintFloor variant — minimal 2-part sufficiency closed      |
| `52e3fb710f` | Promote `sprint_floor_gain` to library primitive + 4 tests   |
| `8357db9a68` | SOTIF reframe in `RoboticAgent::tick` doc-comments           |
| `1fceed0179` | S_p sweep harness + initial N=5 headline                     |
| `ad43b0934c` | N=30 S_p sweep — final Figure 2 values                       |
| `8d61e348d9` | `sprint_floor_gain` adoption on flight-demo                  |
| `c2f2fb46c8` | `sprint_floor_gain` adoption on vehicle-demo                 |
| `36756e10b6` | Initial paper outline                                        |
| `9ecb4f48c6` | Figures 1 & 2 rendering + initial data                       |
| `9314215ecc` | Abstract draft 1 (149/150 words)                             |
| `48a8eafc4d` | §9.1 Crazyflie hardware-validation plan                      |
| `1a85fce8c8` | Humanoid `EmbodimentBridge` — closes 10-for-10 coverage      |
| `babb112e9e` | `flight_benchmark` harness — paired TierGate vs SprintFloor  |
| `c62d12c048` | Figure 3 — N=30 second-platform replication                  |

## The claim, in one sentence

Φ-gated motor authority implemented as `if Φ > sprint { 1.0 } else { floor }`
beats binary ISO/TS 15066 SSM under epistemic uncertainty about the
human-motion envelope, and the result transfers from a 7-DOF manipulator
to a 4-DOF quadrotor with no change to the supervisor code — by ~179 %
at the S_p = 2.5 m crossover (manipulator) and by ~71 % gating-shape
advantage (quadrotor).

## What this paper is NOT claiming

- That Φ is a certified safety layer. The SOTIF / ISO 21448 framing in
  `symtropy/crates/symtropy-robotics-bridge/src/agent.rs` positions it
  as a *triggering-condition monitor* — not a replacement for certified
  hardware envelopes. Single-point-of-failure hazards (cautery, blade
  engagement) still require a Φ-independent hardware interlock; the
  surgical demo's dual-channel pattern (commit `bcd80ef6aa`) is the
  reference implementation.
- That Symthaea's consciousness model is neurally realistic. The paper
  uses "Φ" to refer to the scalar output of `MasterConsciousnessEquation`
  — not to Tononi-sense integrated information. The narrative holds
  whether or not the underlying equation has the phenomenological
  properties its designers hope for; the supervisor story is about
  continuous vs binary gating, not consciousness claims.
- That sprint-floor is universally optimal. The 2-part sufficient
  condition is a *minimal* claim: this is the *least* mapping that
  works. Richer Φ→gain maps might perform equal or better on platforms
  whose empirical Φ distributions differ materially from the
  manipulator's [0.099, 0.145] band.

## License

Code: AGPL-3.0-or-later (matches the rest of the symtropy / symthaea
robotics stack). Paper prose: CC-BY-4.0 when drafted.
