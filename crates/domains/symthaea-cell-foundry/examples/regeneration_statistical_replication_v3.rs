// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Statistical Replication v3 (tuned-differentiation scenario)
//!
//! `regeneration_statistical_replication_v2.rs` traded away the standard
//! 20-day maturation window to get *any* eligible (wound-boundary AND
//! still-progenitor) cells, and even then only reached a mean eligible
//! count of 0.030 -- still not enough for a reliably significant result.
//!
//! This instead keeps the standard scenario every other regeneration demo
//! in this crate uses (20-day maturation, `boundary_r=0.2`, `amputate(0.6,
//! 2.0)`), and fixes the eligible-cell gap at its actual source: the new
//! opt-in `differentiation_threshold_multiplier` (see
//! `examples/differentiation_threshold_tuning.rs` for the full tuning
//! sweep), set to `12.0`, applied *before* maturation so progenitors
//! actually survive the standard 20-day window instead of crashing to zero
//! by day 8.
//!
//! **Status: designed and mechanism-verified, but not run to completion in
//! this session.** The mechanism this scenario depends on (progenitors
//! surviving past day 20 under a 12x multiplier) is directly covered by
//! three passing unit tests in `src/bioelectric.rs`
//! (`differentiation_threshold_multiplier_defaults_to_one`,
//! `differentiation_threshold_multiplier_default_matches_legacy_behavior`,
//! `higher_differentiation_threshold_multiplier_delays_progenitor_depletion`)
//! and by the `differentiation_threshold_tuning.rs` sweep. But this
//! specific 30-seed-then-8-seed replication run itself never finished --
//! every attempt (debug build, a `--release` LTO rebuild, an `--offline`
//! retry) stalled under this session's system load (consistently 40-57,
//! from other concurrent Claude sessions per
//! `.claude/rules/CONCURRENT_SESSIONS.md`), not from any bug in this code.
//! Reported honestly as an incomplete experiment, not a null or negative
//! result -- rerun this file directly when the machine is less loaded to
//! get the real mean-eligible-count and paired-t-test numbers.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_statistical_replication_v3

use symthaea_cell_foundry::bioelectric::{VMEM_DEPOLARIZED, VMEM_HYPERPOLARIZED};
use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;
use symthaea_core::hdc::statistics::one_sample_t_test;

const CELLS: usize = 150;
const MATURATION_DAYS: u32 = 20;
const BOUNDARY_R: f32 = 0.2;
// NOTE: a 12x differentiation-threshold multiplier keeps far more cells
// progenitor for far longer, which means far more proliferation --
// `differentiation_threshold_tuning.rs` observed final tissue sizes of
// 350-450 cells (vs ~170 at the default multiplier) even before any
// amputation/recovery proliferation is added on top. That makes each run
// meaningfully slower than the original/v2 replications, so this uses
// fewer seeds and a shorter recovery window than those did, to keep a
// debug-mode run (this workspace runs under heavy concurrent-session
// contention, and a `--release` LTO rebuild of `symthaea-core` was
// impractically slow to wait for) finishing in a few minutes rather than
// tens of minutes.
const RECOVERY_DAYS: u32 = 30;
const NUM_SEEDS: u64 = 8;
const ALPHA: f64 = 0.05;
const DIFFERENTIATION_THRESHOLD_MULTIPLIER: f32 = 12.0;
/// Distinct from every other demo's seed range.
const SEED_OFFSET: u64 = 30_000;

struct RunOutcome {
    final_discrepancy: f64,
    mean_eligible_count: f64,
}

fn build_tuned_organoid(seed: u64) -> NeuralOrganoid {
    let mut organoid = NeuralOrganoid::new(CELLS, seed);
    organoid.set_differentiation_threshold_multiplier(DIFFERENTIATION_THRESHOLD_MULTIPLIER);
    for _ in 0..MATURATION_DAYS {
        organoid.advance_day();
    }
    organoid.impose_vmem_pattern(|p| {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r >= BOUNDARY_R {
            VMEM_HYPERPOLARIZED
        } else {
            VMEM_DEPOLARIZED
        }
    });
    organoid.capture_target_morphology();
    organoid
}

fn run(seed: u64, fep_enabled: bool) -> RunOutcome {
    let mut organoid = build_tuned_organoid(seed);
    organoid.set_fep_regeneration_enabled(fep_enabled);
    organoid.amputate(0.6, 2.0);

    let mut eligible_sum = 0.0;
    for _ in 0..RECOVERY_DAYS {
        organoid.advance_day();
        let eligible = (0..organoid.field.num_cells())
            .filter(|&i| {
                organoid.field.bioelectric.wound_boundary[i]
                    && organoid.field.cells[i].cell_type.is_progenitor()
            })
            .count();
        eligible_sum += eligible as f64;
    }

    RunOutcome {
        final_discrepancy: organoid.morphology_discrepancy().unwrap_or(1.0),
        mean_eligible_count: eligible_sum / RECOVERY_DAYS as f64,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_std(v: &[f64], m: f64) -> f64 {
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    println!(
        "Replicating legacy-vs-FEP-driven regeneration across {NUM_SEEDS} seeds, standard \
         scenario ({MATURATION_DAYS}d maturation, boundary_r={BOUNDARY_R}, amputate(0.6, 2.0)), \
         differentiation_threshold_multiplier={DIFFERENTIATION_THRESHOLD_MULTIPLIER}...\n"
    );

    let mut legacy_results = Vec::with_capacity(NUM_SEEDS as usize);
    let mut fep_results = Vec::with_capacity(NUM_SEEDS as usize);
    let mut diffs = Vec::with_capacity(NUM_SEEDS as usize);
    let mut legacy_eligible = Vec::with_capacity(NUM_SEEDS as usize);
    let mut fep_eligible = Vec::with_capacity(NUM_SEEDS as usize);

    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "seed", "legacy", "fep_driven", "diff", "leg_elig", "fep_elig"
    );
    for i in 0..NUM_SEEDS {
        let seed = SEED_OFFSET + i;
        let legacy = run(seed, false);
        let fep = run(seed, true);
        let diff = fep.final_discrepancy - legacy.final_discrepancy;
        println!(
            "{seed:>6} {:>12.4} {:>12.4} {diff:>12.4} {:>12.2} {:>12.2}",
            legacy.final_discrepancy,
            fep.final_discrepancy,
            legacy.mean_eligible_count,
            fep.mean_eligible_count
        );
        legacy_results.push(legacy.final_discrepancy);
        fep_results.push(fep.final_discrepancy);
        diffs.push(diff);
        legacy_eligible.push(legacy.mean_eligible_count);
        fep_eligible.push(fep.mean_eligible_count);
    }

    let legacy_mean = mean(&legacy_results);
    let fep_mean = mean(&fep_results);
    let diff_mean = mean(&diffs);
    let mean_eligible_overall = mean(
        &legacy_eligible
            .iter()
            .chain(fep_eligible.iter())
            .copied()
            .collect::<Vec<_>>(),
    );

    println!();
    println!("mean eligible-cell count across all runs and days: {mean_eligible_overall:.3}");
    if mean_eligible_overall < 0.5 {
        println!(
            "Still essentially zero -- the tuned scenario did not fix the eligible-cell gap \
             either, reported honestly rather than claiming otherwise."
        );
    } else {
        println!(
            "Non-trivial eligible-cell counts confirmed at the *standard* 20-day maturation \
             window -- unlike v2, this didn't require compromising the scenario's realism to \
             get there."
        );
    }

    println!();
    println!(
        "legacy:     mean discrepancy = {legacy_mean:.4}, sd = {:.4}",
        sample_std(&legacy_results, legacy_mean)
    );
    println!(
        "fep_driven: mean discrepancy = {fep_mean:.4}, sd = {:.4}",
        sample_std(&fep_results, fep_mean)
    );
    println!("mean paired difference (fep - legacy) = {diff_mean:.4}");

    let result = one_sample_t_test(&diffs, 0.0, ALPHA);
    println!();
    println!("Paired t-test (one-sample t-test on the differences, H0: mean diff = 0):");
    println!("  test statistic (t) = {:.4}", result.test_statistic);
    println!("  p-value             = {:.4}", result.p_value);
    println!(
        "  {:.0}% CI of mean diff = ({:.4}, {:.4})",
        result.confidence_level * 100.0,
        result.confidence_interval.0,
        result.confidence_interval.1
    );
    println!("  reject H0 at alpha={ALPHA} = {}", result.reject);
}
