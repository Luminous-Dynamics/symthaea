// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Statistical Replication
//!
//! `regeneration_agent_demo.rs` claimed "the FEP-driven agent converges to
//! the same policy as the legacy flat rate" from a single seed. That's an
//! honest report of what happened, but it's also exactly the kind of
//! comparative claim a single run can't actually support -- one seed can't
//! distinguish "genuinely no effect" from "this particular random tissue
//! happened to land there." This replicates that same single-episode
//! comparison across 30 independent seeds and runs a real paired
//! significance test on the result, instead of eyeballing one number.
//!
//! **Why a paired t-test, not Wilcoxon signed-rank.** A paired design (same
//! seed, same starting tissue, only the intervention differs) calls for a
//! paired test. Research before writing this found no Wilcoxon signed-rank
//! implementation anywhere in this ~3.6M-line monorepo, and `statrs` (the
//! obvious crates.io candidate) provides distributions but no hypothesis
//! tests at all -- adding it would still mean hand-writing the ranking/
//! tie-correction/W-statistic logic from scratch, while also pulling in a
//! second, semver-incompatible copy of `nalgebra`. What *is* already real,
//! tested, and reachable (no new dependency) is
//! `symthaea_core::hdc::statistics::one_sample_t_test`, and a paired t-test
//! is exactly a one-sample t-test on the paired differences against
//! `mu0 = 0.0` -- so that's what this uses.
//!
//! **A real limitation, stated plainly.** That function's own doc comment
//! says its p-value uses "the normal approximation... accurate for n > 30."
//! This experiment uses exactly 30 seeds to sit right at that documented
//! threshold rather than comfortably inside it -- treat the p-value as
//! indicative, not exact, and look at the effect size and confidence
//! interval alongside it, not the p-value in isolation.
//!
//! **A real methodological bug found and fixed while building this, and
//! what happened after fixing it.** The first run of this experiment (with
//! `symthaea_fep::ActiveInferenceAgent`'s stock constructor) produced a
//! paired difference of *exactly* 0.0000 for all 30 seeds -- not just "not
//! significant," bit-for-bit identical. Investigating why found a real bug:
//! `ActiveInferenceAgent::new()` always seeds its internal action-selection
//! RNG from the same hardcoded golden-ratio constant, with no way to vary
//! it per instance. Every organoid's regeneration agent was making decisions
//! from the exact same internal random stream regardless of the tissue's
//! own seed -- a genuine confound, not just a footnote. Fixed upstream in
//! `symthaea-fep` with a small, purely additive `ActiveInferenceAgent::
//! set_rng_seed()` method (verified against `symthaea-fep`'s own 163 tests
//! and this crate's 295 -- all still green; nothing existing calls the new
//! method, so nothing existing could regress), then threaded a real
//! per-organoid seed through `RegenerationAgent::new` and
//! `NeuralOrganoid::creation_seed`.
//!
//! Re-running after the fix: **the result did not change.** Still exactly
//! 0.0000 across all 30 seeds, now with genuinely independent per-organoid
//! agent randomness. That rules out shared RNG state as the explanation.
//!
//! **Update, from a follow-up investigation
//! (`regeneration_agent_efe_investigation.rs`): the real explanation is
//! simpler and more mechanistic than "near-deterministic argmax," which
//! that follow-up directly disproved** (the actual action-probability
//! distribution stays close to uniform throughout, and the chosen action
//! genuinely varies day to day -- this agent is not behaving
//! deterministically at all). What actually explains the exact-0.0000
//! result: `regenerative_proliferate_with_boost` only affects cells that
//! are BOTH wound-boundary AND still progenitor-type, and in this scenario
//! (a 20-day-matured organoid, then amputated) that eligible count is 0 on
//! every single day -- essentially no wound-adjacent cells are still
//! progenitors by then. No matter which multiplier gets chosen, by the
//! agent or the legacy flat rate, there's nothing left to apply it to. This
//! is a real gap in this scenario's design, not a property of the agent or
//! the mechanism itself -- see that file for the full investigation.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_statistical_replication

use symthaea_cell_foundry::build_radial_bipolar_template;
use symthaea_core::hdc::statistics::one_sample_t_test;

const CELLS: usize = 150;
const MATURATION_DAYS: u32 = 20;
const BOUNDARY_R: f32 = 0.2;
const RECOVERY_DAYS: u32 = 60;
const NUM_SEEDS: u64 = 30;
const ALPHA: f64 = 0.05;
/// Offset so this experiment's seeds don't collide with any other demo's.
const SEED_OFFSET: u64 = 10_000;

fn final_discrepancy(seed: u64, fep_enabled: bool) -> f64 {
    let mut organoid = build_radial_bipolar_template(seed, CELLS, MATURATION_DAYS, BOUNDARY_R);
    organoid.set_fep_regeneration_enabled(fep_enabled);
    organoid.amputate(0.6, 2.0);
    for _ in 0..RECOVERY_DAYS {
        organoid.advance_day();
    }
    organoid.morphology_discrepancy().unwrap_or(1.0)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_std(v: &[f64], m: f64) -> f64 {
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    println!(
        "Replicating the legacy-vs-FEP-driven regeneration comparison across \
         {NUM_SEEDS} independent seeds ({CELLS} cells, {MATURATION_DAYS}d maturation, \
         {RECOVERY_DAYS}d recovery per condition)...\n"
    );

    let mut legacy_results = Vec::with_capacity(NUM_SEEDS as usize);
    let mut fep_results = Vec::with_capacity(NUM_SEEDS as usize);
    let mut diffs = Vec::with_capacity(NUM_SEEDS as usize);

    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "seed", "legacy", "fep_driven", "diff(fep-leg)"
    );
    for i in 0..NUM_SEEDS {
        let seed = SEED_OFFSET + i;
        let legacy = final_discrepancy(seed, false);
        let fep = final_discrepancy(seed, true);
        let diff = fep - legacy;
        println!("{seed:>6} {legacy:>12.4} {fep:>12.4} {diff:>12.4}");
        legacy_results.push(legacy);
        fep_results.push(fep);
        diffs.push(diff);
    }

    let legacy_mean = mean(&legacy_results);
    let fep_mean = mean(&fep_results);
    let diff_mean = mean(&diffs);

    println!();
    println!(
        "legacy:     mean = {legacy_mean:.4}, sd = {:.4}",
        sample_std(&legacy_results, legacy_mean)
    );
    println!(
        "fep_driven: mean = {fep_mean:.4}, sd = {:.4}",
        sample_std(&fep_results, fep_mean)
    );
    println!("mean paired difference (fep - legacy) = {diff_mean:.4}");

    let result = one_sample_t_test(&diffs, 0.0, ALPHA);
    println!();
    println!("Paired t-test (via one-sample t-test on the differences, H0: mean diff = 0):");
    println!("  test statistic (t)      = {:.4}", result.test_statistic);
    println!("  p-value                 = {:.4}", result.p_value);
    println!(
        "  {:.0}% CI of mean diff    = ({:.4}, {:.4})",
        result.confidence_level * 100.0,
        result.confidence_interval.0,
        result.confidence_interval.1
    );
    println!("  reject H0 at alpha={ALPHA}     = {}", result.reject);

    println!();
    if result.reject {
        println!(
            "Statistically distinguishable from zero at this seed count: the FEP-driven \
             agent's effect on final discrepancy is not just single-seed noise, whichever \
             direction it points."
        );
    } else {
        println!(
            "Not statistically distinguishable from zero at this seed count and alpha level. \
             This is now a real, replicated null result across {NUM_SEEDS} independent seeds \
             -- not \"we didn't look hard enough,\" but \"across {NUM_SEEDS} independent \
             tissues, this specific FEP configuration does not detectably change final \
             regeneration discrepancy versus the legacy flat rate.\""
        );
    }
    if diffs.iter().all(|d| d.abs() < 1e-12) {
        println!(
            "\nNote: every paired difference above is exactly zero, even with the RNG-seeding \
             bug (see module docs) already fixed and genuinely independent per-organoid agent \
             randomness in place. That points past \"shared randomness\" to something more \
             structural: this untrained agent's action selection is apparently a near-\
             deterministic argmax for this observation shape, not meaningfully stochastic \
             exploration -- worth investigating directly (e.g. logging the actual expected-\
             free-energy values per action) if this agent is developed further."
        );
    }
}
