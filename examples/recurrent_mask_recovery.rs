// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Step 1.5: actuator bandwidth of the recurrent-dimension masking lever.
//!
//! Step 1 (`docs/RECURRENT_MASK_CHARACTERIZATION_2026-07-30.md`) found that the
//! lesion **compounds and does not recover** under sustained masking, and that
//! expansion is not the inverse of contraction — restored dimensions re-enter at
//! exactly zero and must be repopulated by the dynamics.
//!
//! That raises a question that decides whether a controller on this lever is
//! possible at all, independent of how good its control signal is:
//!
//! > After expanding, how many cycles until the restored dimensions carry a
//! > normal share of the state again?
//!
//! If that recovery time is long relative to the timescale on which a controller
//! wants to adapt (prediction error moves cycle-to-cycle), the mechanism is
//! disqualified as an adaptive actuator — not a signal problem, a bandwidth
//! problem.
//!
//! # Preregistered decision rule
//!
//! Fixed before running, and not to be adjusted after seeing the result:
//!
//! - median half-recovery **> 20 cycles** → DISQUALIFIED as an adaptive actuator.
//!   Skip the Step 2 controller ladder; record the dynamic-dimensionality
//!   hypothesis as cheaply falsified on this lever.
//! - median half-recovery **< 5 cycles** → the compounding is a transient.
//!   Step 2 proceeds, with contraction duration as an explicit controlled
//!   variable.
//! - **between 5 and 20** → Step 2 proceeds but restricted to duty-cycle-matched
//!   arms only, because otherwise an adaptive controller can win purely by
//!   spending time unmasked rather than by having an informative signal.
//!
//! # Why measure an energy *share* rather than a norm
//!
//! Step 1 disclosed that the unmasked control is not stationary — its state norm
//! grew ~5 orders of magnitude over 200 cycles. An absolute norm therefore cannot
//! serve as a recovery target. The share of total energy carried by the
//! previously-masked dimensions is baseline-invariant, and Step 1 established its
//! healthy value: energy is essentially **uniform** across dimensions, so the
//! trailing half should carry ~0.5 of total energy in a healthy state.
//!
//! Section F characterizes the norm growth directly, since it is an open question
//! in its own right and contaminates any absolute-magnitude measurement here.
//!
//! Run: `cargo run --release --example recurrent_mask_recovery`

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Contraction durations to test. Recovery plausibly depends on how deep the
/// compounding got, so this is measured at a short and a long contraction.
const CONTRACTIONS: [usize; 2] = [40, 160];
/// Cycles to observe after expanding.
const RECOVERY_WINDOW: usize = 120;
/// Cycles for the norm-trajectory characterization (section F).
const NORM_TRAJECTORY_CYCLES: usize = 600;
/// Independent repeats, so "median half-recovery" in the decision rule is a
/// median over something rather than a single observation.
const REPEATS: usize = 3;

/// The mask fraction used during contraction. 0.5 keeps the previously-masked
/// set equal to the surviving set, which makes the uniform-energy target exactly
/// 0.5 and the arithmetic transparent.
const CONTRACT_FRAC: f32 = 0.5;

/// Healthy share of total energy in the trailing half, per Step 1's finding that
/// energy is uniform across dimensions.
const UNIFORM_TARGET: f64 = 0.5;

// Preregistered thresholds.
const DISQUALIFY_ABOVE_CYCLES: usize = 20;
const TRANSIENT_BELOW_CYCLES: usize = 5;

const PROBE: &str = "recurrent mask recovery probe";

fn base_config() -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::with_cfc();
    c.enable_validation_overlay = false;
    c
}

fn masked_config(frac: f32) -> CognitiveLoopConfig {
    let mut c = base_config();
    c.enable_recurrent_dim_masking = true;
    c.effective_dim_fraction_override = Some(frac);
    c
}

/// Share of total squared-L2 energy carried by the trailing `1 - frac` of dims.
fn trailing_share(state: &[f32], frac: f32) -> f64 {
    let total: f64 = state.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let start = ((frac * state.len() as f32) as usize).min(state.len());
    let tail: f64 = state[start..]
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum();
    tail / total
}

fn median_usize(mut v: Vec<usize>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable();
    v[v.len() / 2] as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// F. Is the baseline stationary?
// ═══════════════════════════════════════════════════════════════════════════

fn section_f_norm_trajectory() {
    println!("\n═══ F. UNMASKED STATE-NORM TRAJECTORY ═══");
    println!(
        "  Step 1 disclosed ~5 orders of magnitude of growth over 200 cycles and did not\n  \
         resolve whether that is convergence from initialization or unbounded growth.\n  \
         Running {NORM_TRAJECTORY_CYCLES} cycles to see whether it plateaus."
    );
    println!("{:>8} {:>16} {:>16}", "cycle", "state_l2", "trailing_share");

    let mut svc = CognitiveLoopService::new(base_config()).expect("service");
    for i in 0..NORM_TRAJECTORY_CYCLES {
        svc.cycle(PROBE);
        if i % 50 == 0 || i == NORM_TRAJECTORY_CYCLES - 1 {
            if let Some(s) = svc.cfc_state_snapshot() {
                let l2: f64 = s
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                println!(
                    "{:>8} {:>16.6} {:>16.4}",
                    i,
                    l2,
                    trailing_share(&s, CONTRACT_FRAC)
                );
            }
        }
    }
    println!(
        "\n  A plateau means the growth was convergence and absolute magnitudes are usable\n  \
         after warmup. Continued growth means every absolute-magnitude comparison in this\n  \
         area is against a moving baseline and must use shares instead."
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// G. Recovery after expansion
// ═══════════════════════════════════════════════════════════════════════════

struct Recovery {
    contraction: usize,
    repeat: usize,
    share_at_expansion: f64,
    half_recovery: Option<usize>,
    full_recovery: Option<usize>,
    final_share: f64,
}

fn measure_recovery(contraction: usize, repeat: usize) -> Recovery {
    let mut svc = CognitiveLoopService::new(masked_config(CONTRACT_FRAC)).expect("service");

    // Phase 1 — contract and hold, letting the compounding develop.
    for _ in 0..contraction {
        svc.cycle(PROBE);
    }

    // Phase 2 — expand. Restored dimensions are exactly zero at this instant.
    svc.set_effective_dim_fraction_override(Some(1.0));
    let share_at_expansion = svc
        .cfc_state_snapshot()
        .map(|s| trailing_share(&s, CONTRACT_FRAC))
        .unwrap_or(f64::NAN);

    let half_target = UNIFORM_TARGET * 0.5;
    let full_target = UNIFORM_TARGET * 0.9;
    let mut half_recovery = None;
    let mut full_recovery = None;
    let mut final_share = f64::NAN;

    for c in 0..RECOVERY_WINDOW {
        svc.cycle(PROBE);
        if let Some(s) = svc.cfc_state_snapshot() {
            let share = trailing_share(&s, CONTRACT_FRAC);
            final_share = share;
            if half_recovery.is_none() && share >= half_target {
                half_recovery = Some(c + 1);
            }
            if full_recovery.is_none() && share >= full_target {
                full_recovery = Some(c + 1);
                break;
            }
        }
    }

    Recovery {
        contraction,
        repeat,
        share_at_expansion,
        half_recovery,
        full_recovery,
        final_share,
    }
}

fn section_g_recovery() -> Vec<Recovery> {
    println!("\n═══ G. RECOVERY AFTER EXPANSION ═══");
    println!(
        "  Contract at frac={CONTRACT_FRAC} for D cycles, expand to 1.0, then count cycles\n  \
         until the previously-masked trailing half carries a normal share of energy.\n  \
         Uniform target = {UNIFORM_TARGET} (Step 1: energy is uniform across dimensions).\n  \
         half = {:.2}, full = {:.2}. Window = {RECOVERY_WINDOW} cycles.",
        UNIFORM_TARGET * 0.5,
        UNIFORM_TARGET * 0.9
    );
    println!(
        "\n{:>8} {:>8} {:>18} {:>14} {:>14} {:>14}",
        "contract", "repeat", "share_at_expand", "half_recov", "full_recov", "final_share"
    );

    let mut out = Vec::new();
    for &d in CONTRACTIONS.iter() {
        for r in 0..REPEATS {
            let rec = measure_recovery(d, r);
            println!(
                "{:>8} {:>8} {:>18.6} {:>14} {:>14} {:>14.4}",
                rec.contraction,
                rec.repeat,
                rec.share_at_expansion,
                rec.half_recovery
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| ">window".into()),
                rec.full_recovery
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| ">window".into()),
                rec.final_share
            );
            out.push(rec);
        }
    }
    out
}

fn main() {
    println!("Recurrent-dimension masking — Step 1.5 actuator bandwidth");
    println!(
        "PREREGISTERED (fixed before running): median half-recovery > {DISQUALIFY_ABOVE_CYCLES} \
         cycles => DISQUALIFIED,\n  < {TRANSIENT_BELOW_CYCLES} => transient (Step 2 proceeds \
         with duration controlled),\n  in between => Step 2 restricted to duty-cycle-matched arms."
    );

    section_f_norm_trajectory();
    let recs = section_g_recovery();

    println!("\n═══ VERDICT ═══");
    // Unrecovered runs are censored observations. Counting them as
    // RECOVERY_WINDOW understates the true value, so the verdict is stated
    // conservatively: a censored median can only push the answer toward
    // DISQUALIFIED, never away from it.
    let censored = recs.iter().filter(|r| r.half_recovery.is_none()).count();
    let halves: Vec<usize> = recs
        .iter()
        .map(|r| r.half_recovery.unwrap_or(RECOVERY_WINDOW))
        .collect();
    let med = median_usize(halves);

    println!("  observations           : {}", recs.len());
    println!("  never half-recovered   : {censored} (censored at {RECOVERY_WINDOW})");
    println!("  median half-recovery   : {med:.1} cycles");

    let verdict = if med > DISQUALIFY_ABOVE_CYCLES as f64 {
        "DISQUALIFIED as an adaptive actuator — skip the Step 2 ladder"
    } else if med < TRANSIENT_BELOW_CYCLES as f64 {
        "TRANSIENT — Step 2 proceeds, with contraction duration controlled"
    } else {
        "MARGINAL — Step 2 restricted to duty-cycle-matched arms only"
    };
    println!("  verdict                : {verdict}");
    if censored > 0 {
        println!(
            "  note                   : censored observations make the median a LOWER bound;\n\
             \x20                          the true value can only be worse."
        );
    }
    println!("\nDone.");
}
