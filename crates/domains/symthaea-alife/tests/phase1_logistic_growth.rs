// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 1a ground-truth test, per `ALIFE_PLAN_2026-07-08.md` §1a.
//!
//! The claim: a population of independent free-energy-minimizing organisms, sharing a finite
//! resource pool (per-capita share = `plant_resource_total / population_count`), should
//! self-limit into a logistic-growth trajectory `dN/dt = rN(1 - N/K)` -- an emergent property of
//! individual birth/death decisions under resource competition, not a scripted curve.
//!
//! Both `K` and `r` are derived from single-organism calibration experiments, run *before* and
//! *independently of* the population simulation being evaluated -- neither is fit to the
//! multi-organism trajectory.
//!
//! - `r` (intrinsic growth rate): ticks for one organism, started at the post-reproduction energy
//!   baseline, to reach the reproduction threshold under an abundant resource signal.
//!   `r ~= ln(2) / T` (doubling-time-to-rate).
//! - `K` (carrying capacity): a first draft used a closed-form mean-field guess
//!   (`plant_resource_total / (metabolic_cost + forage_activity_cost)`), assuming organisms
//!   forage on essentially every tick near equilibrium. A traced diagnostic run falsified that
//!   assumption -- the actual (untrained/lightly-trained) policy forages only ~50% of the time,
//!   putting a naive formula's predicted breakeven share off by nearly an order of magnitude
//!   from the real one. Replaced with an empirical bisection: find the resource share at which a
//!   single isolated organism's long-run energy neither grows nor shrinks, then `K =
//!   plant_resource_total / breakeven_share`. This is the same "measure it, don't assume it"
//!   discipline Phase 0 needed for its own ground-truth threshold.

use symthaea_alife::{Organism, OrganismConfig, Population, PopulationConfig};

const PLANT_RESOURCE_TOTAL: f64 = 3.0;

fn population_config() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8, // == OrganismConfig::default().set_point
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig {
            // See module docs: default 0.15 leaves populations on a knife-edge given the real
            // (~50%, not ~100%) equilibrium forage rate. Bumped for robust, legible growth.
            forage_efficiency: 0.6,
            ..OrganismConfig::default()
        },
        ..Default::default()
    }
}

/// Independent calibration: ticks for one organism to go from the post-reproduction baseline to
/// the reproduction threshold under an abundant (not shared/divided) resource signal.
fn calibrate_doubling_ticks(cfg: &PopulationConfig, seed: u64) -> u64 {
    let mut organism = Organism::new(cfg.organism_cfg, seed);
    organism.energy = cfg.reproduction_energy_cost;
    const ABUNDANT_RESOURCE: f64 = 1.0;
    const MAX_TICKS: u64 = 5000;
    for t in 1..=MAX_TICKS {
        let tick = organism.tick(ABUNDANT_RESOURCE, None);
        if tick.energy >= cfg.reproduction_energy_threshold {
            return t;
        }
    }
    panic!(
        "organism never reached the reproduction threshold in {MAX_TICKS} ticks under an \
         abundant resource signal -- calibration itself is broken, not just the population sim"
    );
}

/// Net energy drift (final - initial) for one isolated organism run at a fixed resource share
/// for `ticks` steps, starting from its own set-point. No reproduction/death, no other
/// organisms -- a clean single-agent measurement, independent of the population sim.
fn net_drift(cfg: &PopulationConfig, resource_share: f64, seed: u64, ticks: u64) -> f64 {
    let mut organism = Organism::new(cfg.organism_cfg, seed);
    let start = organism.energy;
    for _ in 0..ticks {
        organism.tick(resource_share, None);
    }
    organism.energy - start
}

/// Bisect for the resource share at which net drift is ~0, averaging over a few seeds per step
/// to damp the single-seed noise a knife-edge equilibrium is naturally sensitive to.
fn calibrate_breakeven_share(cfg: &PopulationConfig) -> f64 {
    const CALIBRATION_SEEDS: &[u64] = &[9001, 9002, 9003];
    let mean_drift_at = |share: f64| -> f64 {
        CALIBRATION_SEEDS
            .iter()
            .map(|&s| net_drift(cfg, share, s, 1000))
            .sum::<f64>()
            / CALIBRATION_SEEDS.len() as f64
    };

    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    for _ in 0..14 {
        let mid = (lo + hi) / 2.0;
        if mean_drift_at(mid) > 0.0 {
            hi = mid; // more than enough -- true breakeven is lower
        } else {
            lo = mid;
        }
    }
    (lo + hi) / 2.0
}

#[test]
fn population_self_limits_toward_the_predicted_carrying_capacity() {
    let cfg = population_config();

    let breakeven_share = calibrate_breakeven_share(&cfg);
    let predicted_k = PLANT_RESOURCE_TOTAL / breakeven_share;

    let calibration_seeds = [101u64, 202, 303];
    let mean_doubling_ticks: f64 = calibration_seeds
        .iter()
        .map(|&s| calibrate_doubling_ticks(&cfg, s) as f64)
        .sum::<f64>()
        / calibration_seeds.len() as f64;
    let predicted_r = std::f64::consts::LN_2 / mean_doubling_ticks;

    const N0: usize = 2;
    let mut population = Population::new(cfg, N0, 7);

    let total_ticks = ((10.0 / predicted_r) as u64).clamp(2000, 20_000);
    let mut trajectory = Vec::with_capacity(total_ticks as usize);
    for _ in 0..total_ticks {
        let summary = population.step(|n| PLANT_RESOURCE_TOTAL / (n.max(1) as f64));
        trajectory.push(summary.population);
    }

    let late_window = &trajectory[trajectory.len() * 9 / 10..];
    let late_mean: f64 = mean_of(late_window);

    // Compare against the known starting population, not an "early window" mean -- doubling
    // time under abundant early-tick conditions turned out fast enough (a couple of ticks) that
    // even the first 10% of a 2000-tick run was already past saturation, making an early-window
    // baseline silently measure post-growth noise instead of actual growth.
    assert!(
        late_mean > N0 as f64 * 1.5,
        "population should have grown substantially from its N0={N0} starting point: \
         late_mean={late_mean:.2}"
    );

    // Wide but meaningful tolerance: bounds real remaining approximation (the breakeven share
    // was calibrated in isolation; multi-organism density effects on permeability/beliefs could
    // still shift the true equilibrium) without accepting "grew without bound" or "collapsed" as
    // a pass.
    let ratio = late_mean / predicted_k;
    assert!(
        (0.4..2.5).contains(&ratio),
        "late-run population should be within a wide but real band of the predicted carrying \
         capacity: predicted_k={predicted_k:.2} (plant_resource_total={PLANT_RESOURCE_TOTAL} / \
         calibrated breakeven_share={breakeven_share:.4}), predicted_r={predicted_r:.5} (mean \
         doubling time={mean_doubling_ticks:.1} ticks), late_mean={late_mean:.2}, ratio={ratio:.2}"
    );
}

fn mean_of(xs: &[usize]) -> f64 {
    xs.iter().sum::<usize>() as f64 / xs.len() as f64
}
