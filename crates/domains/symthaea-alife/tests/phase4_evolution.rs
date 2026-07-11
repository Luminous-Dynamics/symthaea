// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 4 ground-truth test, per `ALIFE_PLAN_2026-07-08.md` §4c.
//!
//! The falsifiable claim: a population where offspring genuinely inherit their genome from the
//! parent that actually reproduced (`InheritanceMode::FromParent` -- real selection, since only
//! organisms that stayed alive long enough to hit `reproduction_energy_threshold` get to pass
//! their genome on) should end up fitter, after many generations, than an otherwise-identical
//! population where offspring instead inherit from a uniformly-random *current* population
//! member (`InheritanceMode::RandomPeer` -- the same selection *pressure* from death/reproduction
//! thresholds applies, but a successful reproduction event no longer preferentially propagates
//! *that individual's* genome). If evolution-by-selection isn't doing anything beyond population
//! drift, these two conditions should be statistically indistinguishable.
//!
//! **Two complementary checks, not one**, after a first run's raw numbers showed why relying on
//! just the outcome metric would be less honest than it looks:
//! - `mean_forage_efficiency` (the *mechanism*): selection should evolve the population's average
//!   `forage_efficiency` upward from the marginal `OrganismConfig::default()` starting value
//!   (Phase 1/3 already established 0.15 is a knife-edge value real fitness pressure should
//!   improve on) more than the shuffled control does. Traced values: selection was higher in
//!   **5 of 5 seeds**, by a real, substantial margin (0.28-0.40 vs. 0.20-0.27) -- a robust,
//!   consistent signal.
//! - `total_births` (the *outcome*): selection should accumulate more total births in aggregate.
//!   True on average across the 5 seeds, but noisier than the mechanism check -- **2 of 5
//!   individual seeds** actually had the shuffled control produce more births than selection,
//!   even though selection's efficiency was higher in every single one of those same seeds.
//!   `total_births` is a downstream, stochastic aggregate (population-size trajectory, exact
//!   timing of birth events) that doesn't track the underlying genome-quality trend perfectly
//!   tick-to-tick, even when that trend is real. Averaging over seeds is how Phase 0-3's tests
//!   have always handled this kind of per-seed noise, so the mean-based assertion stays, but the
//!   mechanism-level check is asserted too rather than treated as an interesting side note --
//!   it's the more reliable evidence that selection is doing something real.

use symthaea_alife::{InheritanceMode, OrganismConfig, Population, PopulationConfig};

fn base_population_config(inheritance: InheritanceMode) -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(), // deliberately the knife-edge default, not the
        // "sustainable" 0.6 efficiency other phases' tests use -- real room for evolution to
        // matter, since the starting population is genuinely marginal.
        mutation_rate: 0.1,
        mutation_std: 0.05,
        inheritance,
    }
}

fn mean_forage_efficiency(pop: &Population) -> f64 {
    pop.organisms
        .iter()
        .map(|o| o.cfg.forage_efficiency)
        .sum::<f64>()
        / pop.organisms.len().max(1) as f64
}

#[test]
fn selection_outperforms_shuffled_inheritance_control() {
    const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
    const TICKS: u64 = 4000;
    const INITIAL_COUNT: usize = 4;
    const PLANT_RESOURCE_TOTAL: f64 = 3.0;

    let mut selection_total_births_sum = 0u64;
    let mut shuffled_total_births_sum = 0u64;
    let mut selection_efficiency_sum = 0.0;
    let mut shuffled_efficiency_sum = 0.0;
    let mut efficiency_higher_under_selection_count = 0usize;

    for &seed in SEEDS {
        let mut selection = Population::new(
            base_population_config(InheritanceMode::FromParent),
            INITIAL_COUNT,
            seed,
        );
        let mut shuffled = Population::new(
            base_population_config(InheritanceMode::RandomPeer),
            INITIAL_COUNT,
            seed,
        );

        for _ in 0..TICKS {
            selection.step(|n| PLANT_RESOURCE_TOTAL / (n.max(1) as f64));
            shuffled.step(|n| PLANT_RESOURCE_TOTAL / (n.max(1) as f64));
        }

        let selection_efficiency = mean_forage_efficiency(&selection);
        let shuffled_efficiency = mean_forage_efficiency(&shuffled);
        if selection_efficiency > shuffled_efficiency {
            efficiency_higher_under_selection_count += 1;
        }

        selection_total_births_sum += selection.total_births;
        shuffled_total_births_sum += shuffled.total_births;
        selection_efficiency_sum += selection_efficiency;
        shuffled_efficiency_sum += shuffled_efficiency;
    }

    let mean_selection_births = selection_total_births_sum as f64 / SEEDS.len() as f64;
    let mean_shuffled_births = shuffled_total_births_sum as f64 / SEEDS.len() as f64;
    let mean_selection_efficiency = selection_efficiency_sum / SEEDS.len() as f64;
    let mean_shuffled_efficiency = shuffled_efficiency_sum / SEEDS.len() as f64;

    // The mechanism: real selection should evolve higher forage_efficiency, consistently, not
    // just on average -- a per-seed check, stronger than an averaged one.
    assert_eq!(
        efficiency_higher_under_selection_count,
        SEEDS.len(),
        "real selection should evolve higher mean forage_efficiency than the shuffled control \
         in every seed, not just on average: only {efficiency_higher_under_selection_count}/{} \
         seeds showed this (mean_selection_efficiency={mean_selection_efficiency:.4}, \
         mean_shuffled_efficiency={mean_shuffled_efficiency:.4})",
        SEEDS.len()
    );

    // The outcome: real selection should accumulate more total births in aggregate (averaged
    // over seeds -- this specific metric is noisier per-seed than the mechanism check above).
    assert!(
        mean_selection_births > mean_shuffled_births,
        "real selection (FromParent) should accumulate more total births than the shuffled-\
         inheritance control (RandomPeer) after {TICKS} ticks, averaged over {} seeds: \
         mean_selection_births={mean_selection_births:.1}, \
         mean_shuffled_births={mean_shuffled_births:.1}",
        SEEDS.len()
    );
}
