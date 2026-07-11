# symthaea-orbital

On-orbit servicing arm platform (`EmbodimentBridge` impl), now with real
propagated orbital dynamics. ~1,100 LOC (arm) + ~1,000 LOC (orbit +
scenarios + trajectory planning), 69 tests (verified via
`cargo test -p symthaea-orbital --lib`).

Part of the Symthaea robotics platform roster. Identity decision (Phase 0 of
`symthaea/SPACE_AUTOMATION_PLAN_2026-07-06.md`, confirmed in Phase 1): this
crate stays **one crate, two coupled subsystems** — the servicing arm and the
orbital environment it rides on — rather than splitting into a separate
orbital-flight crate. A servicing arm's whole point is that it lives on a
spacecraft bus; the two are not separable concerns.

## What is modeled (real)

- **7-DOF arm dynamics** (`simulator.rs`, `types.rs`): per-joint torque → damped
  angular acceleration → integrated angle, with joint limits (±2.9 rad) and
  velocity zeroing at the limit.
- **Arm↔bus reaction coupling**: joint torques react onto the spacecraft bus's
  angular velocity; a reaction-wheel model (rate feedback) absorbs that
  reaction, momentum-capacity-limited with genuine saturation consequences
  (see momentum desaturation below — a real bug in this exact mechanism was
  found and fixed while building that scenario).
- **Forward kinematics** to a 3D end-effector position (4-link planar chain +
  a small out-of-plane term from the wrist joint).
- **Real two-body + drag orbital propagation** (Phase 1, 2026-07-07): bus
  position/velocity in a fixed inertial frame, propagated via symplectic
  (semi-implicit) Euler using the shared `orbital-mechanics` crate's gravity
  constant (`coordinates::wgs84::MU`) and real atmospheric drag
  (`atmosphere::drag_acceleration`, USSA-1976 exponential density model).
  Verified: a circular orbit stays circular (velocity ⊥ position), higher
  orbits move slower (Kepler), and a full ~90-minute LEO period roughly
  closes the loop (`simulator.rs` tests). **No J2 oblateness or third-body
  perturbations yet** — pure two-body + drag.
- **Impulsive thruster burns + delta-v budget**: `OrbitalCommand::translational_burn_mps`
  applies directly to velocity, clamped to whatever remains of
  `OrbitalConfig::delta_v_budget_m_s` (a 50 m/s placeholder order-of-magnitude,
  not a specific mission's figure). Cumulative usage tracked in
  `OrbitalState::delta_v_used_m_s`.
- **Real (but simplified) eclipse geometry**: `solar_exposure` uses an honest
  cylindrical-shadow test against a **fixed** reference sun direction — real
  geometry, no penumbra, and the sun direction doesn't actually move (see
  `simulator.rs::REFERENCE_DIRECTION` doc comment for why this is fine within
  one training episode but not over hours/days).
- **Real (but simplified) ground-visibility geometry**: `comm_window` uses a
  genuine horizon/elevation check (5° minimum elevation) against a single
  fixed ground point — real line-of-sight math, but Earth's rotation isn't
  modeled, so it's one fixed ground station at a fixed instant, not a real
  ground track.
- **HDC encoding** (`encoder.rs`): 33-channel state (7 original + 7 orbital,
  added Phase 1) → level-based hypervector encoding, bound per-channel,
  normalized. Deterministic given a `GenesisSeed`.
- **Controller** (`controller.rs`): `HdcLtcUnifiedNetwork` (closed-form CfC) →
  linear readout → `tanh`-bounded joint torques. **Does not yet drive
  `translational_burn_mps`** — the network still only outputs arm torques;
  burns stay zero unless set externally (e.g. by a test or future planner).
- **Safety tiers** (`embodiment.rs`): standard Green/Yellow/Orange/Red from
  `MotorSafetyLevel::from_phi`, plus a moral-gate hook — ahimsa violation or
  a blocked verdict forces Red, consent violation forces Orange. At Red the
  arm **Parks**: zero relative-motion torque AND zero all thruster burns,
  arm+bus drift passively rather than risking uncontrolled motion near other
  spacecraft or debris.
- **Stuck-thruster fault injection** (Phase 2, 2026-07-07):
  `OrbitalEmbodiment::inject_stuck_thruster([f32;3])` models a command-level
  fault — a burn that keeps trying to fire every step until
  `clear_stuck_thruster()`. Verified end-to-end: the fault genuinely spends
  delta-v at Green (`test_stuck_thruster_fault_applies_at_green`) and is
  genuinely zeroed by Red-tier Park (`test_stuck_thruster_fault_zeroed_at_red`)
  — this specifically closes a real gap where Red only zeroed
  `joint_torques`, not `translational_burn_mps`, added when burns were
  introduced in Phase 1. Explicitly does NOT model an actuator-level fault
  (a valve that ignores a zero-command) — that needs hardware redundancy,
  not a control-loop gate, and would defeat any software fix by construction.
- **Perturbation types** (`perturbations.rs`): `DebrisImpact`, `CommBlackout`,
  `SolarFlare` are defined as data but still have no consumer — the stuck-
  thruster fault above is a new, separate mechanism, not built on these.
- **Station-keeping under drag** (`scenarios.rs`, Phase 2, 2026-07-07):
  `run_station_keeping()` runs a scripted bang-bang baseline (burn prograde
  when altitude drops below a tolerance band, coast otherwise) against
  `SimpleOrbitalSimulator::with_config()` and reports `Success` / `Decayed`
  (hit a hard floor) / `PropellantExhausted` (needed to burn but budget was
  gone). Verified non-vacuous: a high-drag low-orbit config genuinely forces
  corrective burns, and a zero-budget version of the same genuinely fails.
  **Important numerical-fidelity finding**: at the 1.0s integration step used
  by the whole-orbit sanity tests, symplectic Euler's own per-step wobble is
  ~1.2km within 335s *with drag disabled* — i.e. that's integration noise,
  not physics. The scenario defaults to a 0.1s step and a 5km tolerance to
  stay clearly above that noise floor; don't tighten `tolerance_km` without
  re-checking the no-drag test's altitude spread first.
- **`SimpleOrbitalSimulator::with_config()`** (Phase 2): the simulator was
  previously always 400km/default-drag/50 m/s budget with no way to
  configure it from outside. Added `with_config(OrbitalConfig)` (`new()` is
  now just `with_config(OrbitalConfig::default())`) plus
  `OrbitalConfig::initial_altitude_km` and a `config()` accessor — needed for
  the station-keeping scenario, and generally useful for any future
  scenario/benchmark work.
- **Momentum desaturation** (`scenarios.rs`, Phase 2, 2026-07-07): fixing this
  found a real bug in the reaction-wheel model — the wheel's rate-feedback
  torque was applied to the bus IN FULL every step regardless of whether the
  wheel's stored momentum actually had room to absorb it (only the *stored*
  value `rwm` was clamped to capacity; the torque *applied to the bus* was
  not). That silently discarded excess momentum instead of reflecting that a
  saturated wheel physically loses authority — meaning saturation had
  **zero** behavioral consequence before this fix. Fixed by clamping the
  actually-absorbed torque from the delta between clamped and unclamped
  momentum, not just the stored value. Added
  `OrbitalCommand::desaturation_torque_nm` (RCS-thruster wheel unloading,
  magnitude-only — direction toward zero is inferred, budget-tracked via
  `OrbitalState::desaturation_used_nms` / `OrbitalConfig::desaturation_budget_nms`,
  never overshoots past zero) and `run_momentum_desaturation()`, a scripted
  baseline that fires desaturation once any axis crosses a threshold fraction
  of capacity, reporting `Success` / `PointingViolated` (saturated + drifted)
  / `DesaturationExhausted` (needed to dump but budget was gone). **Tuning
  note worth remembering**: the wheel's own proportional rate law has a
  nominal steady-state tracking-error floor (`v_ss ≈ disturbance_torque / 50`)
  — a pointing tolerance tighter than that floor fails "randomly" even with
  an undamaged, unsaturated wheel; the disturbance magnitude in the default
  scenario config was chosen specifically to keep that floor well under the
  tolerance, so a violation is actually caused by saturation, not normal
  control-law error.
- **Conjunction avoidance** (`scenarios.rs`, Phase 2, 2026-07-07): the first
  scenario to actually exercise the shared `orbital-mechanics` crate's real
  `ConjunctionAnalyzer`/`RiskLevel`/`CollisionProbability` risk-assessment
  code, not just its physics primitives. `run_conjunction_avoidance()` builds
  a fixed (non-maneuvering) secondary object's predicted TCA position offset
  cross-track from our own coast trajectory by a configured miss distance,
  then each step projects our current state forward to TCA via simple linear
  coast (the standard screening-stage simplification — full nonlinear
  propagation is for TCA *refinement*, not initial risk triage) and assesses
  risk against it. Once assessed risk reaches a trigger level it fires
  cross-track avoidance burns; reports `Success` / `CollisionRiskAtTca`
  (maneuvered the whole time, still unsafe by TCA) / `DeltaVExhausted`.
  **Tuning note**: the analyzer's no-covariance Pc fallback is
  `pc = exp(-x²/2) * (hbr_km)²` (x = miss_km); at the analyzer's own 20m
  default hard-body radius, max achievable Pc at miss=0 is only 4e-4 —
  `RiskLevel::Emergency` (Pc ≥ 1e-3) is mathematically unreachable via this
  fallback at that HBR regardless of how close the miss is. Used 100m HBR so
  Emergency is reachable at realistic sub-km misses; worth remembering before
  citing Emergency-risk behavior against the library's own defaults.
- **Rendezvous/docking** (`scenarios.rs`, Phase 2, 2026-07-07): the last of
  Phase 2's five failure-mode items. Required a genuine new library addition
  — the shared `orbital-mechanics` crate had no relative-motion physics at
  all, only absolute orbital state — so this added
  `orbital_mechanics::clohessy_wiltshire` (closed-form Clohessy-Wiltshire/
  Hill's equations state-transition-matrix propagation, standard textbook
  linearized relative dynamics for a chaser near a circular-orbit target;
  see that module's docs for the governing equations and references).
  `run_rendezvous_docking()` uses a proportional glideslope control law
  (desired closing velocity proportional to remaining along-track distance,
  lateral correction toward the corridor centerline) against a tapering
  approach corridor (full width at the start, narrowing linearly to zero at
  the target — a real docking approach must get straighter as it closes,
  not just stay in a fixed tube). Reports `Docked` (position AND velocity
  both within capture tolerance — a fast flyby through the capture point is
  a miss, not a dock), `AbortedCorridorViolation`, `DeltaVExhausted`, or
  `TimedOut`. Uses a separate, smaller proximity-ops delta-v budget
  (single-digit m/s) rather than `OrbitalConfig::delta_v_budget_m_s`, since
  this scenario is self-contained relative-motion and doesn't otherwise
  touch the absolute-state simulator. Unlike the numerically-approximate
  two-body+drag simulator, CW's closed-form solution is **exact for any
  step size** — no integration-noise tuning concern the way station-keeping
  had.
- **Lambert transfer planning** (`trajectory_planning.rs`, Phase 3,
  2026-07-07): `plan_min_delta_v_transfer()` grid-searches time-of-flight to
  find the minimum-delta-v Lambert transfer between two position vectors,
  cross-validated against the closed-form Hohmann transfer
  (`keplerian::hohmann_transfer`) for a coplanar LEO→GEO case — both the
  best TOF found (near the analytic Hohmann half-period) and its delta-v
  (within ~30% of the analytic value, expected given a 179° rather than
  exact 180° transfer angle — see below) agree with known physics.
  **Real documentation trap found and fixed while building this**:
  `orbital_mechanics::LambertSolution::delta_v_total` is documented as
  "Total ΔV required" but actually computes `|v1| + |v2|` — the raw speed
  magnitudes of the transfer orbit's own endpoints, correct only for
  departure/arrival from/to rest (deep space), never for an orbital
  rendezvous. Naively minimizing that field would have silently planned the
  wrong transfer. This crate's own `test_lambert_hohmann_agreement` already
  knew this and computed the real delta-v itself rather than trusting the
  field; confirmed zero consumers of `delta_v_total` exist anywhere else in
  the monorepo. Fixed the doc comment in `orbital-mechanics` with the
  correct formula and a warning (commit `55bf0ec324`); every search in this
  module computes real delta-v explicitly the same way.
  **Also confirmed empirically**: Lambert's problem doesn't reject a
  too-short time-of-flight the way one might expect — it happily returns a
  valid (if absurdly expensive) conic connecting LEO and GEO in 10 seconds.
  The real "this TOF is a bad idea" signal is prohibitive delta-v, not
  solver failure; this is exactly why the search (not just a single
  `solve_lambert()` call) is the actual planning primitive.
- **Earth-Jupiter gravity-assist demo** (`trajectory_planning.rs`, Phase 3
  item 2, 2026-07-07, commit `17e9f24617`): `plan_earth_jupiter_gravity_
  assist()` composes real VSOP87 planetary ephemeris (new
  `orbital_mechanics::planetary_ephemeris` module, wrapping the `astro`
  crate — the pre-existing `planetary-ephemeris`/`astro-coords` registry
  entries in `mycelix-workspace/Cargo.toml` turned out to be dead, don't
  resolve on crates.io at all) with a real Lambert solve and a real
  patched-conic Jupiter flyby (new `orbital_mechanics::gravity_assist`
  module) to demonstrate the honest "grav-craft": a trailing-side Jupiter
  encounter genuinely boosts heliocentric speed using gravity that already
  exists, at zero extra propellant cost beyond the initial departure burn.
  Grid-searches TOF, recomputing Jupiter's real position at each candidate
  arrival date (the interplanetary arrival point genuinely moves with TOF,
  unlike the fixed-endpoint LEO→GEO case above) and picking the candidate
  minimizing the departure burn. Uses the Sun's gravitational parameter for
  the heliocentric leg and Jupiter's for the flyby — both via the new
  `orbital_mechanics::solar_system::mu_km3_s2()` helper, added specifically
  because the crate's existing `coordinates::wgs84::MU` is Earth-specific
  and would have been physically wrong here. Single-flyby scope, not a
  multi-leg mission-design search; J2000.0 is an arbitrary real calendar
  date, not a claim about matching any historical mission's actual launch
  window. **Verified (commit `3d88517600`)**: the first version used an
  800-1200 day TOF window guessed from the idealized coplanar-Hohmann
  half-period, which found only a ~40 km/s "best" departure delta-v --
  caught by the test suite's own sanity bound the first time it actually
  ran. A wide empirical scan found the real minimum-energy window near
  TOF=3000 days at ~8.7-9.2 km/s, matching real Earth->Jupiter
  mission-class values (Galileo/Juno-era v_infinity ~9 km/s) -- the
  idealized formula assumes an optimal 180 deg relative phase at
  departure that an arbitrary real date generally doesn't have. Also
  corrected a doc overclaim: a leading-side flyby does NOT necessarily
  lose speed (as first written) -- at this real geometry both senses
  gain speed, trailing more than leading. All 72 crate tests pass.
- **Earth-Jupiter gravity-assist public example** (`examples/
  gravity_assist_demo.rs`, Phase 3 item 3, 2026-07-07, commits
  `7ec1ef910a`/`3d88517600`): runs `plan_earth_jupiter_gravity_assist()`
  directly (no LLM/consciousness loop) for both flyby senses and prints
  incoming/outgoing heliocentric speed. Deliberately scoped as a
  standalone example rather than a psych-bench cognitive-domain
  benchmark -- the function is a deterministic optimizer, not a
  reasoning task. Verified: builds and runs in debug mode
  (`cargo run --example gravity_assist_demo --features orbital`),
  prints physically sane numbers (departure delta-v 8.698 km/s).
  Registered in `Cargo.toml`'s `[[example]]` list (this crate has
  `autoexamples = false` -- every example needs an explicit entry).
- **Periapsis search + Earth-Jupiter-Saturn chain** (`trajectory_planning.rs`,
  2026-07-08, commit `17d32c7714`): two extensions.
  `search_periapsis_for_max_speed_gain()` grid-searches periapsis radius (real
  physical floor: Jupiter's actual radius) to maximize speed gain for a fixed
  departure date + TOF -- real finding: with no radiation/safety constraint,
  the optimum always lands at the search's own minimum periapsis (turn angle
  strictly increases as periapsis decreases), the same reason real missions
  add their own safety margin instead of pure speed-optimizing.
  `plan_earth_jupiter_saturn_chain()` is the "rediscover a Voyager-class
  route" two-leg extension: leg 1 fixed at a caller-supplied TOF, leg 2
  (Jupiter->Saturn) a separate real Lambert solve searched over TOF, then
  periapsis x leading/trailing searched to find the flyby closest to what
  leg 2 requires -- the "connection gap" honestly reports the delta-v a
  deep-space maneuver at Jupiter would still need to supply, since the
  modeled flyby (single-plane rotation) can't be steered to hit an arbitrary
  target. **Real bug caught, same class as the earlier TOF-window fix**: an
  initial 100k-1,000,000 km periapsis range for the chain landed its "best"
  exactly on the 1,000,000 km upper boundary; a wider scan found
  `connection_gap_kms` vs periapsis has a genuine INTERIOR minimum near
  periapsis=1,455,000 km (gap ~3.53 km/s), not a monotonic trend -- widened
  the range to 100k-3,000,000 km and added a boundary-check test assertion.
  10/10 `trajectory_planning` tests pass, 75/75 crate tests pass.
- **Two-flyby chain: Earth-Jupiter-Saturn-Uranus** (`trajectory_planning.rs`,
  2026-07-08, commit `6431cee7cc`): `plan_earth_jupiter_saturn_uranus_chain()`
  extends the two-leg chain by one more real gravity assist, matching the
  first three legs of Voyager 2's actual "grand tour" sequence (arbitrary
  real date, not a claim about matching Voyager 2's real 1977 launch). A
  genuine two-flyby search: each flyby (Jupiter, then Saturn) independently
  searches its own periapsis x side x next-leg TOF, using each planet's own
  real mu and physical radius as the floor. GREEDY composition, not jointly
  optimized -- Stage A picks the best Jupiter connection, then assumes a
  deep-space maneuver corrects onto that arc (the standard patched-conic
  technique real missions like Cassini/Galileo actually use) before Stage B
  independently optimizes the Saturn connection; a truly joint optimization
  across both flybys is real, explicitly out-of-scope future work. Verified:
  both flybys land at genuine interior optima (Jupiter periapsis=1,450,000km
  gap=3.532km/s, Saturn periapsis=2,100,000km gap=2.556km/s), not search
  boundaries. 77/77 crate tests pass.
- **Joint optimizer for the two-flyby chain** (`trajectory_planning.rs`,
  2026-07-08, commit `d0ca9cec29`): `plan_earth_jupiter_saturn_uranus_chain_
  jointly_optimized()` fixes the prior extension's greedy-composition
  limitation by jointly optimizing leg-2 TOF against the TOTAL connection gap
  (both flybys combined), not just Jupiter's own gap. Tractable without a 6D
  grid search: the two gaps are coupled ONLY through the shared leg-2 TOF
  (each flyby's own periapsis/side doesn't affect the other's gap under the
  DSM assumption), so for each candidate leg-2 TOF both flybys' searches stay
  independent and cheap. `test_joint_optimizer_never_worse_than_greedy`
  asserts the defining correctness property directly. **Real finding**: for
  this route, joint optimization picks leg-2 TOF=2650 days (not greedy's
  2600), reducing total connection gap from 6.088 to 5.465 km/s -- a genuine
  10.2% improvement, proving greedy really was leaving value on the table.
  78/78 crate tests pass.

## What is NOT modeled (be precise about this in any demo copy)

- **No J2/oblateness, no third-body (lunar/solar), no SGP4/TLE-based orbits.**
  Only two-body gravity + atmospheric drag. Real astrodynamics for all of
  these already exists in the shared `orbital-mechanics` crate
  (`mycelix-workspace/crates/orbital-mechanics` — Keplerian elements, Lambert
  transfers, TLE/CDM parsing, conjunction screening, SGP4 propagation, and
  now Clohessy-Wiltshire) but isn't wired into this simulator's per-step
  integration loop yet — see Phase 2/3 of the space-automation plan.
- **No real CDM ingestion.** All five Phase 2 failure-mode items are now
  done. The conjunction-avoidance scenario constructs its secondary object
  synthetically rather than parsing a real `cdm_parser::parse_cdm_kvn()`
  message — a smaller, real follow-up, since the parser itself already
  works and isn't urgent.
- **Rendezvous/docking's "final capture uses the existing 7-DOF arm"**
  (the plan's original framing) **isn't wired up.** The scenario ends at
  `Docked` (position+velocity within tolerance) as a relative-motion-only
  result; it doesn't hand off to `SimpleOrbitalSimulator`'s arm dynamics for
  an actual capture sequence. Reasonable next step, not done here.
- **Phase 3 (trajectory planning): items 1-3 done, plus four extensions.**
  Lambert transfer planning, the Earth-Jupiter gravity-assist demo, the
  public example, periapsis search, the Earth-Jupiter-Saturn chain, the
  Earth-Jupiter-Saturn-Uranus two-flyby chain, AND a joint optimizer for
  that chain (all above) are complete and verified. Psych-bench wiring was
  deliberately skipped as a scope mismatch (see above). The joint optimizer
  covers the two-flyby case's real degrees of freedom (both periapsides,
  both sides, both downstream TOFs) via the shared-leg-2-TOF coupling
  insight -- a genuinely joint optimum for THIS route. What remains
  unstarted is generalizing this to N>2 flybys (e.g. adding a further
  Uranus-onward leg), where the same coupling-through-shared-TOF trick
  would need to chain across more than one shared variable.
- **Eclipse and comm-window geometry are real but frame-simplified** — see
  above. Not valid over more than one short training episode without adding
  real solar ephemeris and Earth rotation.
- **The controller doesn't fly the bus.** Translational burns and
  desaturation commands are both commandable and physically real once
  applied, but nothing in this crate yet generates either autonomously — no
  station-keeping or desaturation policy, no learned or planned thruster
  behavior. Only the scripted scenario baselines in `scenarios.rs` drive
  them, for benchmark purposes.
- **`training.rs` does not train.** `OrbitalTrainer::run_episode()` runs a
  fixed-weight episode and reports metrics (mean spacecraft rate, mean
  effort, divergence). No gradient step, no weight update anywhere in this
  crate — `OrbitalController`'s weights are genesis-random and fixed for the
  process lifetime. This matches the cross-cutting finding in
  `SYMTHAEA_ROBOTICS_IMPROVEMENT_PLAN_2026-07-06.md`: cognition ≈ bias until
  an actual trainer runs, and this crate doesn't have one yet.
- **`fep_agent.rs` is a stub.** `ActiveInferenceOrbitalAgent` wraps
  `symthaea_fep::ActiveInferenceAgent` but `tick()` ignores it — `tau_factor`
  and `free_energy` are hand-computed from angular rate, not read from the
  wrapped agent's belief state.
- **`PerturbationSchedule` appears unused** by the simulator or trainer as of
  this writing — defined types with no consumer found. Verify before citing
  it as a working fault-injection mechanism.

## Roadmap

See `symthaea/SPACE_AUTOMATION_PLAN_2026-07-06.md` for the full phased plan
(propagated dynamics, failure-mode scenarios, trajectory planning, dispatch
wiring, hardware/qualification/institutional-adoption tracks). This crate
sits at tier 4-5 in the robotics roadmap priority (behind subterranean,
infrastructure) — treat this README and the plan as ready-to-go, not as
signaling active development.
