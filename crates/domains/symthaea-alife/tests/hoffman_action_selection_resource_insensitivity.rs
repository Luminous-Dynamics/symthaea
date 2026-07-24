// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root-cause finding behind `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md` Phase 2's negative
//! result: an invasion sweep over 16 `(spoilage_sigma, forage_activity_cost)` combinations
//! (interior-optimum, non-monotonic payoffs, per `organism.rs`'s `spoilage_sigma`) never once
//! showed fine-grained perception netting more energy than coarse-grained -- every combination
//! favored coarse, exactly like Phase 1's monotonic environment. This test isolates why.
//!
//! At a *constant* true `resource_level` (belief given hundreds of ticks to converge, no
//! quantization involved at all -- this test doesn't touch `perceptual_grain`), `select_action`'s
//! own real, non-forced choice of Forage vs. Rest over the final 100 ticks is **identical**
//! across resource levels spanning the full oscillation range (0.05 through 0.95) and, per seed,
//! bit-for-bit identical to each other. That means the forage/rest decision does not depend on
//! the resource observation at all in this regime -- it's some other factor (RNG stream and/or
//! the energy/set_point dimension) driving the choice, not the resource reading.
//!
//! This is the deeper reason neither Phase 1 (monotonic) nor Phase 2 (non-monotonic) payoff
//! shapes could ever reward finer resolution: if the decision doesn't use the resolved
//! information regardless of shape, any real cost charged for resolving it -- `perceptual_grain`'s
//! Landauer tax -- is pure waste no matter how the true payoff is shaped. Investigating *why*
//! `select_action` is resource-insensitive here would mean touching `symthaea-fep`'s
//! `ActiveInferenceAgent` internals, a different crate and a materially larger scope than this
//! plan's Phase 1/2 -- explicitly not attempted, left as an open question for a separately-scoped
//! investigation if wanted.

use symthaea_alife::{Action, Organism, OrganismConfig};

const TICKS: u64 = 500;
const WINDOW: u64 = 100;
const RESOURCE_LEVELS: &[f64] = &[0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95];
const SEEDS: &[u64] = &[100, 200, 300, 400, 500];

fn forage_fraction_at_constant_resource(resource_level: f64, seed: u64) -> f64 {
    let cfg = OrganismConfig {
        forage_efficiency: 0.6,
        ..OrganismConfig::default()
    };
    let mut organism = Organism::new(cfg, seed);
    let mut forage_count = 0u64;
    let mut sampled = 0u64;
    for t in 0..TICKS {
        let tick = organism.tick(resource_level, None);
        if t >= TICKS.saturating_sub(WINDOW) {
            if tick.action == Action::Forage.index() {
                forage_count += 1;
            }
            sampled += 1;
        }
    }
    forage_count as f64 / sampled.max(1) as f64
}

#[test]
fn forage_decision_is_insensitive_to_constant_resource_level() {
    // Per-seed forage-fraction at the lowest resource level is the reference; every other
    // resource level must match it exactly, for every seed -- not just "close," identical,
    // matching what was actually observed (byte-identical per-seed vectors across all 7 levels).
    let reference: Vec<f64> = SEEDS
        .iter()
        .map(|&seed| forage_fraction_at_constant_resource(RESOURCE_LEVELS[0], seed))
        .collect();

    for &r in &RESOURCE_LEVELS[1..] {
        for (i, &seed) in SEEDS.iter().enumerate() {
            let frac = forage_fraction_at_constant_resource(r, seed);
            assert_eq!(
                frac, reference[i],
                "seed={seed}: forage-fraction at resource_level={r} ({frac}) differs from the \
                 reference at resource_level={} ({}) -- if this now fails, the decision policy \
                 has become resource-sensitive and this plan's Phase 2 negative result should be \
                 re-examined",
                RESOURCE_LEVELS[0], reference[i]
            );
        }
    }
}
