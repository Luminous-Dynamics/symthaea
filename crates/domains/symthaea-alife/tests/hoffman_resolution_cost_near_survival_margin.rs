// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Attempted positive control for `hoffman_fitness_beats_truth.rs`, per
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md` Phase 1 follow-up.
//!
//! The goal was a task where truth-tracking (fine `perceptual_grain`) should *win*: raise
//! `forage_activity_cost` so the forage-vs-rest breakeven sits near `Environment::default()`'s
//! oscillation midpoint (0.5), reasoning that correctly resolving which side of that midpoint the
//! true resource reading is on should carry a real survival payoff under a monotonic environment.
//!
//! A diagnostic sweep (`activity_cost` from 0.05 to 0.40, single organisms, 8 seeds each) found
//! something different and more interesting than the hypothesized "fine wins near the threshold":
//! a **sharp cliff**, not a gradual zone -- at `activity_cost=0.05` both perceptual strategies
//! survive comfortably; by `activity_cost=0.08` both die unconditionally regardless of
//! resolution. In the narrow band between (`0.065`-`0.075`), fine-grained organisms died far
//! *more* often than coarse-grained ones (at `0.07`: fine 7/8 deaths vs coarse 0/8; at `0.075`:
//! fine 8/8 vs coarse 7/8) -- i.e. resolution's own real Landauer tax, on its own, was enough to
//! push already-marginal organisms over the death threshold *before* any decision-quality benefit
//! from finer resolution had a chance to matter. This is a stronger, more dramatic demonstration
//! of the same direction as the original invasion experiment, not the hoped-for reversal.
//!
//! Consistent with the very first research pass's prediction: under a *monotonic* payoff (more
//! true resource -> more energy, unconditionally), coarse bucketing already tracks "high vs low
//! enough to forage" about as well as fine bucketing does for a binary threshold decision -- so
//! raising the overall cost bar sharpens fine's built-in tax into lethality rather than creating
//! room for a resolution-driven advantage. A genuine positive control (a task where truth-
//! tracking pays off) likely needs a *non-monotonic* payoff (interior optimum -- e.g. too much
//! resource is spoilage/danger, not just more gain), matching this plan's originally-scoped
//! Phase 2, not a monotonic-environment cost tweak. Not attempted here; left as the concrete next
//! step if this line of research continues.
//!
//! Kept as a real, asserted finding rather than deleted: this is genuine evidence about the
//! resolution cost's magnitude relative to the survival margin, worth having as a regression
//! test even though it isn't the positive control it set out to be.

use symthaea_alife::{Environment, Organism, OrganismConfig};

fn run_single(grain: Option<f64>, activity_cost: f64, seed: u64, ticks: u64) -> (f64, bool) {
    let cfg = OrganismConfig {
        forage_efficiency: 0.6,
        forage_activity_cost: activity_cost,
        perceptual_grain: grain,
        ..OrganismConfig::default()
    };
    let mut organism = Organism::new(cfg, seed);
    let env = Environment::default();
    let mut sum_energy = 0.0;
    let mut count = 0u64;
    let mut died = false;
    for t in 0..ticks {
        let tick = organism.tick(env.resource_at(t), None);
        if t >= ticks / 4 {
            sum_energy += tick.energy;
            count += 1;
        }
        if tick.is_dead {
            died = true;
            break;
        }
    }
    (sum_energy / count.max(1) as f64, died)
}

const SEEDS: u64 = 8;
const TICKS: u64 = 3000;

fn death_counts(grain: Option<f64>, activity_cost: f64, seed_offset: u64) -> u32 {
    (0..SEEDS)
        .filter(|&s| run_single(grain, activity_cost, seed_offset + s, TICKS).1)
        .count() as u32
}

#[test]
fn mild_cost_both_survive_coarse_still_slightly_ahead() {
    // Extends hoffman_fitness_beats_truth.rs's finding to a config where survival isn't at
    // stake at all: at a mild activity cost (0.05, everyone comfortably survives), coarse still
    // nets more energy than fine on average -- the resolution tax is a real, if small, drag with
    // no offsetting benefit here, exactly as the invasion experiment's mechanism predicts.
    let (fine_energy, fine_died) = run_single(Some(0.02), 0.05, 1000, TICKS);
    let (coarse_energy, coarse_died) = run_single(Some(0.4), 0.05, 2000, TICKS);
    assert!(
        !fine_died && !coarse_died,
        "both should comfortably survive at this mild cost"
    );
    assert!(
        coarse_energy > fine_energy,
        "coarse should net more energy than fine at a mild activity cost with no decision \
         benefit to fine resolution: fine={fine_energy}, coarse={coarse_energy}"
    );
}

#[test]
fn near_critical_cost_resolution_tax_alone_can_be_lethal() {
    // The actual, unexpected finding: right at the survival cliff (activity_cost=0.07), fine's
    // own real energy cost -- not any decision-quality difference -- is enough to kill it far
    // more often than coarse. A single seed pair could be noise; assert the aggregate across 8
    // seeds per strategy, matching this crate's own multi-seed discipline.
    let fine_deaths = death_counts(Some(0.02), 0.07, 1000);
    let coarse_deaths = death_counts(Some(0.4), 0.07, 2000);
    assert!(
        fine_deaths >= 5,
        "expected fine-grained organisms to die in most seeds near the survival cliff, got \
         {fine_deaths}/{SEEDS}"
    );
    assert_eq!(
        coarse_deaths, 0,
        "expected coarse-grained organisms to survive every seed at the same cost, got \
         {coarse_deaths}/{SEEDS} deaths"
    );
}
