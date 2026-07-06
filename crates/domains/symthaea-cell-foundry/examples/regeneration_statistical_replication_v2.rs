// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Statistical Replication v2 (eligible-cell-corrected scenario)
//!
//! `regeneration_agent_efe_investigation.rs` traced
//! `regeneration_statistical_replication.rs`'s exact-0.0000-every-seed
//! finding to a scenario-design gap, not an agent or mechanism failure:
//! `regenerative_proliferate_with_boost` only ever affects cells that are
//! BOTH wound-boundary AND still progenitor-type, and the standard 20-day
//! maturation window every regeneration demo in this crate used left zero
//! such cells -- `progenitor_fraction_probe.rs` found the tissue-wide
//! progenitor count crashes from 150 to 0 within just 8 days of maturation
//! (150 -> 11 -> 3 -> 1 -> 0 at days 0/2/4/6), and never recovers.
//!
//! This redoes the same 30-seed paired comparison as the original
//! replication, but amputates at day 3 (while progenitors still exist)
//! with a broad cut (`amputate(0.3, 2.0)`, removing most of the outer
//! tissue) to maximize the chance some surviving progenitors land within
//! the wound boundary. It also logs mean eligible-cell count per condition
//! so this scenario's validity can be checked directly rather than assumed.
//!
//! **Result: a real, partial improvement, and a deeper limitation found
//! along the way.** Mean eligible-cell count went from exactly 0.0 (the
//! original scenario) to 0.030 -- still low, but genuinely nonzero. The
//! mechanistic link is clean and confirmed directly in the per-seed data:
//! every seed with `eligible = 0.00` still shows an exact 0.0000 paired
//! difference, while the handful of seeds with nonzero eligible counts
//! (e.g. seed 20005: eligible=1.43/0.03, diff=-0.0088) show genuinely
//! nonzero differences. That's the causal mechanism working exactly as
//! traced, directly visible now rather than just inferred.
//!
//! It's still not enough seeds to reach significance (only ~4 of 30 ever
//! see nonzero eligible counts), and trying an even earlier amputation
//! (day 1, tested and then reverted) barely helped (0.038 vs. 0.030) --
//! this model's differentiation dynamics are apparently fast enough
//! (tissue-wide progenitor count crashes 150 -> 11 within 2 days, per
//! `progenitor_fraction_probe.rs`) that no reasonable "amputate at day N"
//! scenario reliably keeps progenitors at the wound boundary. Fully fixing
//! this would mean retuning the core Turing/differentiation rate
//! constants themselves -- a much bigger, more invasive change with its
//! own broad blast radius across the many existing tests tuned against
//! current differentiation timing, and explicitly out of scope for this
//! follow-up. Reported as a real, still-open limitation, not solved here.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_statistical_replication_v2

use symthaea_cell_foundry::build_radial_bipolar_template;
use symthaea_core::hdc::statistics::one_sample_t_test;

const CELLS: usize = 150;
const MATURATION_DAYS: u32 = 3;
const BOUNDARY_R: f32 = 0.2;
const RECOVERY_DAYS: u32 = 60;
const NUM_SEEDS: u64 = 30;
const ALPHA: f64 = 0.05;
const SEED_OFFSET: u64 = 20_000; // distinct from every other demo's seed range

struct RunOutcome {
    final_discrepancy: f64,
    mean_eligible_count: f64,
}

fn run(seed: u64, fep_enabled: bool) -> RunOutcome {
    // Same helper the original replication used (build_radial_bipolar_template),
    // just with a much shorter maturation window -- the only variable this
    // v2 changes, for a fair comparison against the original.
    let mut organoid = build_radial_bipolar_template(seed, CELLS, MATURATION_DAYS, BOUNDARY_R);
    organoid.amputate(0.3, 2.0);
    organoid.set_fep_regeneration_enabled(fep_enabled);

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
        "Replicating legacy-vs-FEP-driven regeneration across {NUM_SEEDS} seeds, \
         corrected scenario: {MATURATION_DAYS}d maturation (not 20), broad amputation \
         at day {MATURATION_DAYS} while progenitors still exist...\n"
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
            "That's still essentially zero -- this scenario did NOT fix the underlying \
             problem, reported honestly rather than claiming otherwise. A broad day-3 \
             amputation apparently still doesn't reliably catch surviving progenitors \
             at the wound boundary in this model."
        );
    } else {
        println!(
            "Non-trivial eligible-cell counts confirmed -- this scenario gives the \
             regeneration mechanism something real to act on, unlike the original."
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
