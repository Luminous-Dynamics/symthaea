// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 5a ground-truth test, per `ALIFE_PLAN_2026-07-08.md` §5a.
//!
//! The falsifiable claim: coupling a [`Population`] to real ice-albedo climate physics
//! (`EarthForcedEnvironment`, `src/earth_forcing.rs`) instead of a synthetic sine wave should
//! produce genuinely different ecological outcomes when the real physics is pushed into a
//! genuinely different regime — not merely swap the source of an otherwise-decorative number.
//!
//! Two scenarios, same population config, same seed, same tick count: one under
//! `EarthForcedEnvironment::earth_like`'s real Earth-like solar constant (habitable branch of
//! the ice-albedo bistability), one with `solar_constant` dimmed to 600 W/m² — the same fixture
//! `symthaea-earth-system::ice_albedo`'s own `snowball_state_is_stable_at_low_insolation` test
//! uses for its real frozen-branch equilibrium. If the coupling is real, the dimmed-sun
//! population should experience a genuine resource collapse (the real integrator driving
//! temperature to the frozen branch, per `earth_forcing`'s own
//! `dimmed_sun_pushes_the_real_model_toward_the_frozen_branch` unit test) and decline, while the
//! habitable-scenario population persists under the same organism config and initial count.
//!
//! First draft used a fixed (not divided by population size) resource share -- Phase 1a's own
//! dev notes already document exactly this trap (unbounded growth, millions of organisms within
//! a few hundred ticks); traced here too before this test file was ever committed. Fixed by
//! scaling the `[0, 1]` habitability proxy against the same shared-pool total
//! (`PLANT_RESOURCE_TOTAL = 3.0`) Phase 2/4's tests already calibrate for this
//! `forage_efficiency`, divided per-capita. With that fix, a real, non-borderline effect: traced
//! across 5 seeds, the habitable scenario settles to a stable 21-25 organisms with zero
//! extinctions over the full 4000-tick window; the snowball scenario collapses to extinction in
//! every seed, consistently between tick 377-416 (well inside the window, not a last-tick
//! coincidence), with temperature settling around 207 K -- deep in the frozen branch, ~56 K
//! below the model's own 263 K ice line.

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};

fn population_config() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig {
            forage_efficiency: 0.6, // Phase 1's "sustainable" value -- real room to observe
            // growth/decline rather than a knife-edge default muddying which scenario is doing
            // the work.
            ..OrganismConfig::default()
        },
        ..Default::default()
    }
}

#[test]
fn dimming_the_sun_past_the_snowball_threshold_collapses_the_population() {
    const TICKS: u64 = 4000;
    const INITIAL_COUNT: usize = 6;
    const SEASONAL_PERIOD_TICKS: f64 = 200.0;
    const SEED: u64 = 11;
    // Same shared-pool total Phase 2/4's tests calibrate against for this forage_efficiency
    // (see `phase2_coalitions.rs`/`phase4_evolution.rs`) -- the real habitability proxy
    // ([0, 1]) scales this shared total, then divides per-capita, so the population is
    // genuinely density-dependent instead of the unbounded-growth "fixed generous resource"
    // pattern Phase 1a's dev notes already document as a real trap (compounds to millions of
    // organisms within a few hundred ticks).
    const PLANT_RESOURCE_TOTAL: f64 = 3.0;

    let mut habitable_env = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS);
    let mut habitable_pop = Population::new(population_config(), INITIAL_COUNT, SEED);
    for _ in 0..TICKS {
        habitable_pop.step(|n| habitable_env.step() * PLANT_RESOURCE_TOTAL / (n.max(1) as f64));
    }

    let mut snowball_env = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS);
    snowball_env.model.solar_constant = 600.0; // matches ice_albedo.rs's own snowball fixture
    let mut snowball_pop = Population::new(population_config(), INITIAL_COUNT, SEED);
    for _ in 0..TICKS {
        snowball_pop.step(|n| snowball_env.step() * PLANT_RESOURCE_TOTAL / (n.max(1) as f64));
    }

    assert!(
        !habitable_pop.organisms.is_empty(),
        "a population under real Earth-like solar forcing should persist over {TICKS} ticks"
    );
    assert!(
        snowball_pop.organisms.is_empty(),
        "a population under a dimmed, sub-snowball-threshold sun should collapse to extinction \
         within {TICKS} ticks (real climate physics genuinely driving the outcome, not a \
         decorative number): final population={}, final temperature={:.1}K (ice line={:.1}K)",
        snowball_pop.organisms.len(),
        snowball_env.temperature,
        snowball_env.model.t_ice
    );
}
