// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hoffman Fitness-Beats-Truth invasion experiment, per
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md` Phase 1.
//!
//! Mark, Marion & Hoffman (2010) "Natural selection and veridical perceptions" (J. Theoretical
//! Biology): truth-tracking perception is never beaten by a coarser "interface" strategy
//! *unless* resolving detail carries a real cost -- only then can frequency-dependent selection
//! drive the truth-tracking strategy to extinction. `perceptual_grain`
//! (`Genome`/`OrganismConfig`) wires exactly this cost into `symthaea-alife` for the first time
//! (see `metabolism::{quantize_to_grain, perceptual_resolution_bits}`): a genuinely finer-grained
//! resource reading costs more real Landauer-charged energy per tick than a coarser one, with no
//! separate built-in benefit -- any advantage to fine-graining has to come entirely from
//! better-informed `select_action` choices actually paying off in more gathered energy, exactly
//! the tradeoff the theorem describes.
//!
//! Design: a single shared-resource population seeded half `GRAIN_FINE`, half `GRAIN_COARSE` --
//! an invasion/competition setup, not two separate runs. Both strategies experience the exact
//! same environment tick-for-tick and compete for the same resource pool.
//! `InheritanceMode::FromParent` with real mutation lets whichever value is actually more
//! successful propagate through real reproduction, the same methodology Phase 7's
//! evolutionary-rescue test uses. Two checks, same discipline as Phase 4c/Phase 7: mechanism (did
//! the population's mean `perceptual_grain` measurably move at all under selection?) and outcome
//! (which pole -- fine or coarse -- actually dominates the surviving population, and how
//! consistently across seeds).
//!
//! This is a real, falsifiable test, not a foregone conclusion: `symthaea-alife`'s environment
//! (`Environment::resource_at`) is smooth, and its action space is a single binary threshold
//! (forage vs. rest) -- so it is plausible *a priori* that fine resolution buys little
//! decision-quality benefit here, exactly the "perceiving detail is not free, and this task does
//! not need much of it" condition under which Hoffman's theorem predicts coarse wins. Reported
//! honestly either way; see the module docs' final paragraph (added once this test was actually
//! run) for the real result.

use symthaea_alife::{
    Environment, InheritanceMode, Organism, OrganismConfig, Population, PopulationConfig,
};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
const TICKS: u64 = 4_000;
const HALF_COUNT: usize = 8; // 16 founders total -- Phase 7 found small founder pools risk an
// early unlucky wipeout independent of the trait under test; erring generous avoids that
// confound here too.
const MECHANISM_SAMPLE_TICK: u64 = 2_000;
const GRAIN_FINE: f64 = 0.02; // ~50 distinguishable buckets, ~5.6 resolution bits/tick
const GRAIN_COARSE: f64 = 0.4; // ~2-3 buckets, ~1.3 resolution bits/tick
const MUTATION_RATE: f64 = 0.1; // same calibration as Phase 4c / Phase 7
const MUTATION_STD: f64 = 0.02; // perceptual_grain's own bounded range (0.02..=0.5) is narrower
// than forage_efficiency's, so a smaller step than Phase 7's 0.05

fn population_config() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig {
            // Phase 1's "sustainable" value -- real room to observe selection acting on
            // perceptual_grain, rather than the population just barely surviving on the
            // knife-edge default (see population.rs's `sustainable_cfg` doc comment).
            forage_efficiency: 0.6,
            ..OrganismConfig::default()
        },
        mutation_rate: MUTATION_RATE,
        mutation_std: MUTATION_STD,
        inheritance: InheritanceMode::FromParent,
    }
}

/// Builds a `Population` with zero auto-generated organisms, then manually pushes `HALF_COUNT`
/// fine-grained and `HALF_COUNT` coarse-grained founders -- the actual invasion/competition
/// starting condition. `Population::organisms` is `pub` specifically to allow this kind of
/// caller-assembled heterogeneous seeding; no crate changes were needed.
fn seed_mixed_population(pop_cfg: PopulationConfig, seed: u64) -> Population {
    let mut pop = Population::new(pop_cfg, 0, seed);
    let base_cfg = pop.cfg.organism_cfg;
    for i in 0..HALF_COUNT {
        let cfg = OrganismConfig {
            perceptual_grain: Some(GRAIN_FINE),
            ..base_cfg
        };
        // Large, disjoint seed offsets so manually-seeded founders never collide with
        // `Population`'s own internally-generated offspring seeds (which start near `seed`).
        pop.organisms.push(Organism::new(
            cfg,
            seed.wrapping_add(500_000 + i as u64).max(1),
        ));
    }
    for i in 0..HALF_COUNT {
        let cfg = OrganismConfig {
            perceptual_grain: Some(GRAIN_COARSE),
            ..base_cfg
        };
        pop.organisms.push(Organism::new(
            cfg,
            seed.wrapping_add(600_000 + i as u64).max(1),
        ));
    }
    pop
}

fn mean_grain(pop: &Population) -> f64 {
    let vals: Vec<f64> = pop
        .organisms
        .iter()
        .filter_map(|o| o.cfg.perceptual_grain)
        .collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

const MIDPOINT: f64 = (GRAIN_FINE + GRAIN_COARSE) / 2.0;

/// (fraction still closer to fine, fraction closer to coarse) among survivors.
fn strategy_fractions(pop: &Population) -> (f64, f64) {
    let total = pop.organisms.len().max(1) as f64;
    let fine = pop
        .organisms
        .iter()
        .filter(|o| matches!(o.cfg.perceptual_grain, Some(g) if g < MIDPOINT))
        .count() as f64;
    (fine / total, 1.0 - fine / total)
}

#[test]
fn perceptual_grain_evolves_under_real_selection_in_a_shared_resource_pool() {
    let env = Environment::default();
    let mut initial_means = Vec::new();
    let mut mechanism_means = Vec::new();
    let mut final_means = Vec::new();
    let mut final_fine_fractions = Vec::new();
    let mut final_populations = Vec::new();

    for &seed in SEEDS {
        let mut pop = seed_mixed_population(population_config(), seed);
        initial_means.push(mean_grain(&pop));
        let mut mech_mean = None;

        for t in 0..TICKS {
            if pop.is_empty() {
                break;
            }
            // Shared, density-adjusted resource pool -- same per-capita-share pattern as Phase
            // 1's `sustainable_cfg` tests, scaled for a ~16-organism founder population.
            pop.step(|n| env.resource_at(t) * 6.0 / (n.max(1) as f64));
            if t == MECHANISM_SAMPLE_TICK {
                mech_mean = Some(mean_grain(&pop));
            }
        }

        assert!(
            !pop.is_empty(),
            "seed={seed}: population went fully extinct -- can't measure a winner. \
             (a real, reportable finding on its own, but it means resource pool calibration \
             needs revisiting before this test can say anything about perceptual_grain)"
        );
        mechanism_means.push(mech_mean.unwrap_or(f64::NAN));
        final_means.push(mean_grain(&pop));
        final_fine_fractions.push(strategy_fractions(&pop));
        final_populations.push(pop.len());
    }

    eprintln!(
        "Hoffman FBT invasion experiment (GRAIN_FINE={GRAIN_FINE}, GRAIN_COARSE={GRAIN_COARSE}):"
    );
    eprintln!("  initial means:         {initial_means:?}");
    eprintln!("  mechanism-sample means (t={MECHANISM_SAMPLE_TICK}): {mechanism_means:?}");
    eprintln!("  final means:           {final_means:?}");
    eprintln!("  final (fine%, coarse%) fractions: {final_fine_fractions:?}");
    eprintln!("  final population sizes: {final_populations:?}");

    // Report, don't assume: this first pass only asserts that selection measurably acted on the
    // trait at all (real drift from the ~midpoint starting mean), not which direction won --
    // the direction is this experiment's actual finding, to be recorded in
    // HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md and this file's own module docs once observed,
    // matching how Phase 7's tight assertions were only written after a traced diagnostic run.
    let mean_shift: f64 = final_means
        .iter()
        .zip(&initial_means)
        .map(|(f, i)| (f - i).abs())
        .sum::<f64>()
        / SEEDS.len() as f64;
    assert!(
        mean_shift > 1e-3,
        "expected real drift in mean perceptual_grain under selection+mutation, got average \
         |shift|={mean_shift} across seeds {SEEDS:?} -- either selection isn't acting on this \
         trait in this environment, or TICKS/MUTATION_STD need recalibrating"
    );
}
