// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests `OrganismConfig::resource_prior` (see its doc comment in `organism.rs`), and the
//! convergent finding it produces for `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`.
//!
//! The mechanism works as intended: lowering `resource_prior` away from the anchoring default
//! (0.5) genuinely changes belief dynamics -- under maximally garbage perception
//! (`perceptual_grain=Some(2.0)`), belief now visibly drops (traced: 0.48 -> 0.29 -> 0.21 -> 0.42
//! -> 0.20 over 500 ticks with `resource_prior=0.0`, vs. staying pinned at ~0.47-0.52 with the
//! default 0.5 -- see `tests/hoffman_directed_join_no_effect.rs`). This is a real behavior change,
//! not a no-op.
//!
//! **But it still doesn't flip any outcome in `Environment::default()`'s range.** Belief still
//! stays above the ~0.075-0.1 decision crossover (`resource_preference`'s doc comment) almost all
//! of the time, because the true environment (oscillating ~0.2-0.8) never gets close enough to
//! that threshold for a lowered prior, individual resolution, or a Directed Join peer signal to
//! matter -- confirmed by re-running both the fine-vs-coarse comparison and the peer-signal
//! experiment across `resource_prior` values 0.5, 0.3, 0.1, 0.0: **identical results in every
//! case** (fine/coarse gap and peer-signal effect both unchanged to 4 decimal places regardless
//! of prior).
//!
//! **This is the third independent mechanism tested and ruled out today**, all converging on the
//! same root cause: `Environment::default()`'s range simply never approaches the decision
//! threshold closely enough for perception quality -- individual resolution
//! (`hoffman_fitness_beats_truth.rs`), goal-preference calibration (now fixed,
//! `resource_preference`), prior-anchoring (this file), or directed social information
//! (`hoffman_directed_join_no_effect.rs`) -- to matter. The actual, sole remaining bottleneck for
//! a genuine positive control is an environment straddling that threshold, which three prior
//! calibration attempts already failed to build for unrelated reasons (extinction x2,
//! unbounded-growth hang) and were stopped rather than forced.

use symthaea_alife::{Environment, Organism, OrganismConfig};

#[test]
fn lowering_resource_prior_changes_belief_but_not_the_forage_rest_outcome() {
    const SEEDS: u64 = 8;
    const TICKS: u64 = 3000;

    fn mean_energy(grain: f64, prior: f64, seed_offset: u64) -> f64 {
        let mut sum = 0.0;
        for s in 0..SEEDS {
            let cfg = OrganismConfig {
                forage_efficiency: 0.6,
                perceptual_grain: Some(grain),
                resource_prior: prior,
                ..OrganismConfig::default()
            };
            let mut organism = Organism::new(cfg, seed_offset + s);
            let env = Environment::default();
            let mut e = 0.0;
            let mut count = 0u64;
            for t in 0..TICKS {
                let tick = organism.tick(env.resource_at(t), None);
                if t >= TICKS / 4 {
                    e += tick.energy;
                    count += 1;
                }
            }
            sum += e / count.max(1) as f64;
        }
        sum / SEEDS as f64
    }

    let fine_at_default_prior = mean_energy(0.02, 0.5, 1000);
    let coarse_at_default_prior = mean_energy(0.4, 0.5, 2000);
    let fine_at_zero_prior = mean_energy(0.02, 0.0, 1000);
    let coarse_at_zero_prior = mean_energy(0.4, 0.0, 2000);

    // The mechanism should reproduce exactly (this test's own regression check on the numbers
    // reported in the module docs): recalibrating the prior changes nothing measurable about
    // this specific decision task, in this specific environment.
    assert!(
        (fine_at_default_prior - fine_at_zero_prior).abs() < 1e-3,
        "fine-grained mean energy should be unaffected by resource_prior in this environment: \
         default_prior={fine_at_default_prior:.4}, zero_prior={fine_at_zero_prior:.4}"
    );
    assert!(
        (coarse_at_default_prior - coarse_at_zero_prior).abs() < 1e-3,
        "coarse-grained mean energy should be unaffected by resource_prior in this environment: \
         default_prior={coarse_at_default_prior:.4}, zero_prior={coarse_at_zero_prior:.4}"
    );
}

#[test]
fn resource_prior_genuinely_changes_belief_dynamics_under_garbage_perception() {
    // Confirms the mechanism itself is real (not a no-op that happens to not matter here):
    // under maximally coarse perception, a lowered prior should measurably pull belief's mean
    // absolute deviation from 0.5 upward, over a real trajectory.
    fn mean_abs_deviation_from_half(prior: f64) -> f64 {
        let cfg = OrganismConfig {
            forage_efficiency: 0.6,
            perceptual_grain: Some(2.0), // collapses every true value to one constant bucket
            resource_prior: prior,
            ..OrganismConfig::default()
        };
        let mut organism = Organism::new(cfg, 42);
        let env = Environment::default();
        let mut sum_dev = 0.0;
        let mut count = 0u64;
        for t in 0..500u64 {
            organism.tick(env.resource_at(t), None);
            if t >= 100 {
                sum_dev += (organism.agent.belief.mean[0] - 0.5).abs();
                count += 1;
            }
        }
        sum_dev / count.max(1) as f64
    }

    let deviation_default = mean_abs_deviation_from_half(0.5);
    let deviation_zero = mean_abs_deviation_from_half(0.0);
    assert!(
        deviation_zero > deviation_default,
        "a lowered prior should pull belief measurably further from 0.5 under garbage \
         perception than the anchoring default: default={deviation_default:.4}, \
         zero_prior={deviation_zero:.4}"
    );
}
