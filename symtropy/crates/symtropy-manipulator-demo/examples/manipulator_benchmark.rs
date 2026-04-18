// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Manipulator Quantitative Benchmark — Monte Carlo
//!
//! Paired comparison of Adaptive Safety Gradient vs ISO/TS 15066 SSM
//! across N trials. Each trial runs both arms on the **same** human
//! trajectory (a deterministic sinusoidal approach parameterized by
//! period / closest / farthest / phase), so the only variable is the
//! safety policy.
//!
//! For each trial we record:
//!   - cycles_adaptive  (pick-place cycles completed in 100 s sim)
//!   - cycles_iso       (same, under binary SSM policy)
//!   - advantage_pct    = (cycles_adaptive − cycles_iso) / cycles_iso · 100
//!
//! Across trials we compute sample mean, sample std-dev, and 95 % CI of
//! the throughput advantage. For N ≥ 30 we use the normal approximation
//! (z = 1.96); the script notes the assumption in the output.
//!
//! ```bash
//! cd symtropy/crates/symtropy-manipulator-demo
//! # default: 30 trials (~1–2 min)
//! cargo run --example manipulator_benchmark --release
//! # more trials for tighter CI
//! MANIP_BENCH_TRIALS=100 cargo run --example manipulator_benchmark --release
//! ```
//!
//! References this benchmark produces evidence for:
//!   - ISO/TS 15066 (now absorbed into ISO 10218-2:2025) SSM baseline
//!   - Research-identified path #3 from the Apr 18 industry comparison:
//!     "Monte Carlo study on manipulator demo vs the already-coded ISO
//!     baseline: produce the 20-40 % throughput claim with confidence
//!     intervals."

use std::time::Instant;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::encoder::ManipulatorHdcEncoder;
use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use symthaea_manipulator::types::NUM_JOINTS;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

// ── Scenario constants ──

/// 500 Hz physics tick.
const DT: f64 = 0.002;

/// 50 000 steps × 2 ms = 100 s of simulated time per arm per trial.
const TOTAL_STEPS: usize = 50_000;

/// Pick / place positions (meters, in the arm's base frame).
const PICK: [f64; 3] = [0.4, -0.3, 0.15];
const PLACE: [f64; 3] = [0.4, 0.3, 0.15];
const APPROACH_H: f64 = 0.30;

/// ISO/TS 15066 protective distance (S_p) — conservative for a 7-DOF arm.
const ISO_SP: f64 = 1.0;

/// Default number of Monte Carlo trials when `MANIP_BENCH_TRIALS` is unset.
const DEFAULT_TRIALS: usize = 30;

// ── Trial parameters ──

#[derive(Clone, Copy, Debug)]
struct TrialParams {
    /// Time for one full human approach-retreat cycle (seconds).
    human_approach_period: f64,
    /// Closest the human comes (m) — may enter the workspace.
    human_closest: f64,
    /// Farthest the human retreats (m).
    human_farthest: f64,
    /// Initial phase offset so different trials don't align with arm cycle.
    phase_offset: f64,
}

impl TrialParams {
    /// Deterministically derive a trial's parameters from its index, so the
    /// whole run is reproducible from the trial count alone.
    fn from_index(i: usize) -> Self {
        // Small splitmix variant: 4 pseudo-uniform floats per trial.
        let mut s = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1);
        let mut next = || -> f64 {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64).clamp(0.0, 1.0)
        };
        Self {
            // Period 4–12 s — captures everything from agitated to strolling.
            human_approach_period: 4.0 + next() * 8.0,
            // Closest 0.25–0.55 m (always deep enough to trip ISO's 1 m cone).
            human_closest: 0.25 + next() * 0.30,
            // Farthest 1.5–3.0 m (always out of the cone so both arms recover).
            human_farthest: 1.5 + next() * 1.5,
            phase_offset: next() * std::f64::consts::TAU,
        }
    }

    /// Sinusoidal human distance from the arm base at time `t`.
    fn human_dist(&self, t: f64) -> f64 {
        let amp = 0.5 * (self.human_farthest - self.human_closest);
        let mid = 0.5 * (self.human_farthest + self.human_closest);
        let phase = t / self.human_approach_period * std::f64::consts::TAU + self.phase_offset;
        mid - amp * phase.sin()
    }
}

// ── Arm-level simulation (paired) ──

/// Safety policy kind, so both variants can share the stepping loop.
#[derive(Clone, Copy)]
enum Policy {
    /// Adaptive gradient: gain is a piecewise-continuous function of
    /// human distance. This is a *deterministic stand-in* for the Φ-gated
    /// policy — the real plugin feeds `RoboticAgent.tick()` HDC PE and
    /// uses the returned motor_gain, but here we substitute a closed-form
    /// gradient keyed on proximity to keep the trial cost low and the
    /// experiment dominated by the policy difference, not the FEP tick.
    Adaptive,
    /// ISO/TS 15066 SSM: binary stop/go at protective distance S_p.
    IsoSsm,
}

fn policy_gain(policy: Policy, human_dist: f64) -> f64 {
    match policy {
        Policy::Adaptive => {
            if human_dist > 1.2 {
                1.0
            } else if human_dist > 0.8 {
                0.6 + 0.4 * (human_dist - 0.8) / 0.4
            } else if human_dist > 0.5 {
                0.3 + 0.3 * (human_dist - 0.5) / 0.3
            } else {
                0.1
            }
        }
        Policy::IsoSsm => {
            if human_dist > ISO_SP {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Run one arm for TOTAL_STEPS and return the number of completed cycles.
fn run_trial_arm(policy: Policy, params: &TrialParams) -> u32 {
    let kinematics = ManipulatorKinematics::default_7dof();
    let mut sim = SimpleManipulatorSimulator::new();
    let mut cycles = 0u32;
    let mut phase = 0;
    let mut target = [PICK[0], PICK[1], APPROACH_H];

    for step in 0..TOTAL_STEPS {
        let t = step as f64 * DT;
        let human_dist = params.human_dist(t);
        let gain = policy_gain(policy, human_dist);

        let state = sim.state();
        if gain > 0.0 {
            if let Some(q_target) = kinematics.ik_dls(&target, &state.joint_angles, 0.1, 30, 0.01) {
                let mut cmd = symthaea_manipulator::ManipulatorCommand::zero();
                for i in 0..NUM_JOINTS {
                    let err = q_target[i] - state.joint_angles[i];
                    let vel = state.joint_velocities[i];
                    cmd.joint_torques[i] =
                        (gain as f32 * (8.0 * err - 2.0 * vel) as f32).clamp(-1.0, 1.0);
                }
                sim.step(&cmd, DT);
            }
        }

        let ee = sim.state().end_effector_position;
        let dist = ((ee[0] - target[0]).powi(2)
            + (ee[1] - target[1]).powi(2)
            + (ee[2] - target[2]).powi(2))
        .sqrt();
        if dist < 0.02 {
            phase = (phase + 1) % 4;
            target = match phase {
                0 => [PICK[0], PICK[1], APPROACH_H],
                1 => [PLACE[0], PLACE[1], APPROACH_H],
                2 => [PLACE[0], PLACE[1], APPROACH_H],
                _ => [PICK[0], PICK[1], APPROACH_H],
            };
            if phase == 0 {
                cycles += 1;
            }
        }
    }

    cycles
}

/// Run the Φ-gated policy for one trial — this is the arm that exercises
/// the **actual `RoboticAgent.tick()` path**, not a proximity-keyed
/// stand-in.
///
/// The gain is updated at a cognitive rate (`COG_HZ = 25`, so every 20th
/// physics step at 500 Hz) from `RoboticAgent.tick(observation,
/// danger_level)`. The observation vector is
/// `[pe, danger, human_norm, effort_norm]` where `pe` is the cosine
/// dissimilarity of the current state's HDC encoding vs the previous
/// cognitive-tick encoding. In between cognitive ticks the last gain is
/// held, matching the real platform's multi-rate control architecture
/// (500 Hz motor / 25 Hz cognitive).
///
/// This is more expensive per trial than [`run_trial_arm`] — about
/// 10× — so the default Φ sim time is shorter (40 s) than
/// Adaptive/ISO's 100 s. Override via `MANIP_BENCH_PHI_STEPS=50000` for
/// a fair 100 s comparison.
///
/// Set `MANIP_BENCH_PHI_TRACE=1` to dump one CSV-ready line per cognitive
/// tick of trial 0 containing `t, human_dist, pe, danger, phi, gain`.
/// Use this to distinguish "Φ pinned at Red" from "Φ oscillating but
/// below activation threshold" when the Φ policy produces zero cycles.
/// Threshold set for mapping Φ → motor_gain. Defaults track the
/// out-of-the-box `symtropy-consciousness-physics::SafetyTier::from_phi`
/// thresholds (Green > 0.6, Yellow > 0.3, Orange > 0.1). The
/// "recalibrated" set is fit to the empirical Φ distribution observed
/// in this benchmark ([0.099, 0.145]) so the arm can actually reach
/// the Green tier in practice.
#[derive(Clone, Copy)]
pub enum ThresholdSet {
    Default,
    Recalibrated,
}

fn gain_from_phi(phi: f64, set: ThresholdSet) -> f64 {
    match set {
        ThresholdSet::Default => {
            // Matches SafetyTier::from_phi + motor_gain exactly.
            if phi > 0.6 {
                1.0
            } else if phi > 0.3 {
                0.6
            } else if phi > 0.1 {
                0.3
            } else {
                0.0
            }
        }
        ThresholdSet::Recalibrated => {
            // Fit to empirical Φ range [0.099, 0.145] observed in the
            // 40 s trace. Top of the band → Green; bottom → Red.
            if phi > 0.135 {
                1.0
            } else if phi > 0.120 {
                0.6
            } else if phi > 0.105 {
                0.3
            } else {
                0.0
            }
        }
    }
}

fn run_trial_phi(params: &TrialParams, trace: bool, thresholds: ThresholdSet) -> u32 {
    const COG_INTERVAL: usize = 20; // 500 Hz / 25 Hz = 20 physics steps per cognitive tick.
    let total_steps = phi_steps();
    let kinematics = ManipulatorKinematics::default_7dof();
    let mut sim = SimpleManipulatorSimulator::new();
    let genesis = GenesisSeed::from_phrase("manipulator-benchmark-phi");
    let mut encoder = ManipulatorHdcEncoder::new(&genesis, 32);
    let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Manipulator, "bench-phi");

    let mut cycles = 0u32;
    let mut phase = 0;
    let mut target = [PICK[0], PICK[1], APPROACH_H];
    let mut last_gain = 1.0_f64;
    let mut last_perception: Option<ContinuousHV> = None;

    if trace {
        println!("# Φ-trace trial_0: t,human_dist,pe,danger,phi,gain");
    }

    for step in 0..total_steps {
        let t = step as f64 * DT;
        let human_dist = params.human_dist(t);

        // Cognitive tick at 25 Hz: update gain from the RoboticAgent.
        if step % COG_INTERVAL == 0 {
            let state = sim.state();
            let hv = encoder.encode(state);
            let pe = match &last_perception {
                Some(prev) => (1.0 - hv.similarity(prev).max(0.0) as f64).clamp(0.0, 1.0),
                None => 0.0,
            };
            // Danger: closer human → higher danger. Linear in (2 m − dist), clamped.
            let danger = ((2.0 - human_dist) / 2.0).clamp(0.0, 1.0);
            let human_norm = (1.0 / (1.0 + human_dist)).clamp(0.0, 1.0);
            let effort_norm = (state
                .joint_velocities
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                / 5.0)
                .clamp(0.0, 1.0);
            let obs = [pe, danger, human_norm, effort_norm];
            // Always run the tick — it updates internal state AND returns
            // the default-threshold gain (which we may ignore in
            // Recalibrated mode).
            let default_gain = agent.tick(&obs, danger);
            last_gain = match thresholds {
                ThresholdSet::Default => default_gain,
                ThresholdSet::Recalibrated => gain_from_phi(agent.phi(), thresholds),
            };
            last_perception = Some(hv);

            if trace {
                // Emit: t,human_dist,pe,danger,phi,gain — one row per cognitive tick.
                // `phi` is the raw consciousness_level from the last compute,
                // `gain` is what the SafetyTier::from_phi→motor_gain pipeline returned.
                println!(
                    "TRACE,{:.3},{:.3},{:.4},{:.4},{:.4},{:.4}",
                    t,
                    human_dist,
                    pe,
                    danger,
                    agent.phi(),
                    last_gain,
                );
            }
        }

        let state = sim.state();
        let gain = last_gain;
        if gain > 0.0 {
            if let Some(q_target) = kinematics.ik_dls(&target, &state.joint_angles, 0.1, 30, 0.01) {
                let mut cmd = symthaea_manipulator::ManipulatorCommand::zero();
                for i in 0..NUM_JOINTS {
                    let err = q_target[i] - state.joint_angles[i];
                    let vel = state.joint_velocities[i];
                    cmd.joint_torques[i] =
                        (gain as f32 * (8.0 * err - 2.0 * vel) as f32).clamp(-1.0, 1.0);
                }
                sim.step(&cmd, DT);
            }
        }

        let ee = sim.state().end_effector_position;
        let dist = ((ee[0] - target[0]).powi(2)
            + (ee[1] - target[1]).powi(2)
            + (ee[2] - target[2]).powi(2))
        .sqrt();
        if dist < 0.02 {
            phase = (phase + 1) % 4;
            target = match phase {
                0 => [PICK[0], PICK[1], APPROACH_H],
                1 => [PLACE[0], PLACE[1], APPROACH_H],
                2 => [PLACE[0], PLACE[1], APPROACH_H],
                _ => [PICK[0], PICK[1], APPROACH_H],
            };
            if phase == 0 {
                cycles += 1;
            }
        }
    }

    cycles
}

// Φ-gated trials use fewer steps because each cognitive tick is ~10× the
// cost of a pure physics step. Keeps the full benchmark well under 30 min
// total wall time even with the Φ policy enabled. Configurable via
// `MANIP_BENCH_PHI_STEPS` — raise to 50_000 for a fair 100 s comparison.
const DEFAULT_PHI_STEPS: usize = 20_000; // 40 s of sim at 500 Hz.
const DEFAULT_PHI_TRIALS: usize = 10;

fn phi_steps() -> usize {
    std::env::var("MANIP_BENCH_PHI_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PHI_STEPS)
}

// ── Monte Carlo harness ──

#[derive(Clone, Copy)]
struct TrialResult {
    adaptive_cycles: u32,
    iso_cycles: u32,
    /// ((adaptive - iso) / iso) * 100, or NaN if iso_cycles == 0.
    advantage_pct: f64,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|x| (x - m).powi(2)).sum();
    (ss / (xs.len() - 1) as f64).sqrt()
}

fn main() {
    let trials: usize = std::env::var("MANIP_BENCH_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TRIALS);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Manipulator Monte Carlo Benchmark                           ║");
    println!("║  Adaptive Safety Gradient vs ISO/TS 15066 SSM                ║");
    println!(
        "║  {:>3} paired trials × 100 s simulated per arm                 ║",
        trials
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();
    let mut results: Vec<TrialResult> = Vec::with_capacity(trials);

    for i in 0..trials {
        let params = TrialParams::from_index(i);
        let adaptive_cycles = run_trial_arm(Policy::Adaptive, &params);
        let iso_cycles = run_trial_arm(Policy::IsoSsm, &params);
        let advantage_pct = if iso_cycles > 0 {
            (adaptive_cycles as f64 - iso_cycles as f64) / iso_cycles as f64 * 100.0
        } else {
            f64::NAN
        };
        results.push(TrialResult {
            adaptive_cycles,
            iso_cycles,
            advantage_pct,
        });
        println!(
            "  trial {:>3}: period={:>5.2}s closest={:.2}m farthest={:.2}m — adaptive={:>2} iso={:>2} adv={:>+6.1}%",
            i + 1,
            params.human_approach_period,
            params.human_closest,
            params.human_farthest,
            adaptive_cycles,
            iso_cycles,
            advantage_pct,
        );
    }

    let elapsed = start.elapsed();

    // Aggregate
    let adv_vals: Vec<f64> = results
        .iter()
        .map(|r| r.advantage_pct)
        .filter(|v| v.is_finite())
        .collect();
    let adaptive_vals: Vec<f64> = results.iter().map(|r| r.adaptive_cycles as f64).collect();
    let iso_vals: Vec<f64> = results.iter().map(|r| r.iso_cycles as f64).collect();

    let adv_mean = mean(&adv_vals);
    let adv_std = std_dev(&adv_vals);
    let adv_se = if adv_vals.len() >= 2 {
        adv_std / (adv_vals.len() as f64).sqrt()
    } else {
        0.0
    };
    // 95 % CI: normal approximation for N ≥ 30, reported either way.
    let z = 1.96;
    let adv_ci_lo = adv_mean - z * adv_se;
    let adv_ci_hi = adv_mean + z * adv_se;

    println!();
    println!(
        "━━━ Summary ({} trials, {:.1}s wall time) ━━━",
        trials,
        elapsed.as_secs_f64()
    );
    println!(
        "  Adaptive cycles:   mean = {:6.2}   std = {:5.2}",
        mean(&adaptive_vals),
        std_dev(&adaptive_vals)
    );
    println!(
        "  ISO/SSM cycles:    mean = {:6.2}   std = {:5.2}",
        mean(&iso_vals),
        std_dev(&iso_vals)
    );
    println!();
    println!("  THROUGHPUT ADVANTAGE (Adaptive over ISO/TS 15066 SSM):",);
    println!("    mean  = {:+6.1} %", adv_mean);
    println!(
        "    sd    = {:6.2} %   (paired sample, N = {})",
        adv_std,
        adv_vals.len()
    );
    println!(
        "    95 % CI ≈ [{:+6.1} %, {:+6.1} %]  (normal approx, z=1.96)",
        adv_ci_lo, adv_ci_hi
    );
    if trials < 30 {
        println!("    (caveat: normal approximation is loose for N < 30 — re-run with",);
        println!("     MANIP_BENCH_TRIALS=50 or higher for a tighter claim)");
    }

    // CSV — one row per trial, for import into R / pandas
    println!();
    println!("━━━ CSV (per-trial) ━━━");
    println!(
        "trial,approach_period_s,closest_m,farthest_m,adaptive_cycles,iso_cycles,advantage_pct"
    );
    for (i, r) in results.iter().enumerate() {
        let p = TrialParams::from_index(i);
        println!(
            "{},{:.4},{:.4},{:.4},{},{},{:.4}",
            i + 1,
            p.human_approach_period,
            p.human_closest,
            p.human_farthest,
            r.adaptive_cycles,
            r.iso_cycles,
            r.advantage_pct,
        );
    }

    // ── Φ-gated policy sweep (opt-in) ──────────────────────────────────
    //
    // Opt-in because each Φ trial is ~10× the cost of a proximity-keyed
    // trial (cognitive tick at 25 Hz runs the HDC encoder + FEP inference).
    // Enabled by setting `MANIP_BENCH_PHI=1`. Scale-matched to
    // `DEFAULT_PHI_TRIALS` × `DEFAULT_PHI_STEPS` so the section adds ~a few minutes
    // of wall time, not an hour.
    let run_phi = std::env::var("MANIP_BENCH_PHI")
        .ok()
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    if !run_phi {
        println!();
        println!("(Φ-gated sweep skipped — set MANIP_BENCH_PHI=1 to include the actual");
        println!(" RoboticAgent.tick()/HDC-PE policy path in the comparison.)");
        return;
    }

    let phi_trials: usize = std::env::var("MANIP_BENCH_PHI_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PHI_TRIALS);

    let phi_n_steps = phi_steps();
    let recal_enabled = std::env::var("MANIP_BENCH_PHI_RECAL")
        .ok()
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let thresholds = if recal_enabled {
        ThresholdSet::Recalibrated
    } else {
        ThresholdSet::Default
    };
    let threshold_label = if recal_enabled {
        "recalibrated [0.105/0.120/0.135]"
    } else {
        "default SafetyTier [0.1/0.3/0.6]"
    };
    println!();
    println!(
        "━━━ Φ-gated sweep ({} trials × {} steps = {} s sim each, thresholds = {}) ━━━",
        phi_trials,
        phi_n_steps,
        (phi_n_steps as f64 * DT) as usize,
        threshold_label,
    );
    let phi_start = Instant::now();
    let mut phi_cycles_vec: Vec<f64> = Vec::with_capacity(phi_trials);
    // Φ runs over a (possibly) shorter sim; rate-normalize to "cycles per
    // 100 s" for apples-to-apples comparison vs Adaptive/ISO at 100 s.
    let scale = TOTAL_STEPS as f64 / phi_n_steps as f64;
    // Only trace trial 0 so the output stays readable.
    let trace_enabled = std::env::var("MANIP_BENCH_PHI_TRACE")
        .ok()
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    for i in 0..phi_trials {
        let params = TrialParams::from_index(i);
        let phi_raw = run_trial_phi(&params, trace_enabled && i == 0, thresholds);
        let phi_scaled = phi_raw as f64 * scale;
        phi_cycles_vec.push(phi_scaled);
        println!(
            "  trial {:>3}: period={:>5.2}s closest={:.2}m farthest={:.2}m — phi_raw={:>2} → {:.2} cycles/100s",
            i + 1,
            params.human_approach_period,
            params.human_closest,
            params.human_farthest,
            phi_raw,
            phi_scaled,
        );
    }
    let phi_elapsed = phi_start.elapsed();

    let phi_mean = mean(&phi_cycles_vec);
    let phi_std = std_dev(&phi_cycles_vec);
    let phi_se = if phi_cycles_vec.len() >= 2 {
        phi_std / (phi_cycles_vec.len() as f64).sqrt()
    } else {
        0.0
    };
    println!();
    println!(
        "  Φ-gated cycles (rate-normalized to 100 s):  mean = {:6.2}  std = {:5.2}",
        phi_mean, phi_std,
    );
    let iso_cycles_100 = mean(&iso_vals); // ISO already at 100 s scale
    if iso_cycles_100 > 0.0 {
        let phi_vs_iso = (phi_mean - iso_cycles_100) / iso_cycles_100 * 100.0;
        let se_vs_iso = 1.96 * phi_se / iso_cycles_100 * 100.0;
        println!(
            "  Φ vs ISO:  mean = {:+6.1} %   95 % CI ≈ [{:+6.1} %, {:+6.1} %]",
            phi_vs_iso,
            phi_vs_iso - se_vs_iso,
            phi_vs_iso + se_vs_iso,
        );
    }
    println!(
        "  (Φ sweep wall time: {:.1}s, {} trials × {:.0}s sim)",
        phi_elapsed.as_secs_f64(),
        phi_trials,
        phi_n_steps as f64 * DT,
    );
}
