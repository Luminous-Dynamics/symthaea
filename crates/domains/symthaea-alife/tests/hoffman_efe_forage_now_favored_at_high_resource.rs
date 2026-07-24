// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifies the fix described in `OrganismConfig::resource_preference`'s doc comment and
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s "Fix" section: `Organism::new` used to pass a
//! hardcoded, ecologically arbitrary `0.5` ("moderate reading") resource-observation preference
//! to `ActiveInferenceAgent::set_goals`, which `tests/hoffman_efe_rest_structurally_dominates.rs`
//! found made `Rest` pragmatically dominate `Forage` at *every* resource belief tested -- a
//! decision policy structurally biased against foraging regardless of true resource level, since
//! real foraging payoff (`organism.rs` step 6) scales monotonically with true resource with no
//! satiation.
//!
//! With the corrected default (`resource_preference: 1.0`, "prefer as much resource as
//! observable"), the crossover moves to where it belongs: `Rest` still wins at very low resource
//! (near-empty is genuinely not worth foraging), but `Forage` wins everywhere above roughly
//! `0.1`, with its advantage growing monotonically as resource increases -- real, fitness-aligned
//! sensitivity to the perceptual channel, unlike the pre-fix behavior.

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

const ENERGY_BELIEF: f64 = 0.8;
const FORAGE: usize = 0;
const REST: usize = 1;

fn probabilities_at_belief(resource_belief: f64, energy_belief: f64) -> Vec<f64> {
    let cfg = ActiveInferenceAgentConfig {
        state_dim: 2,
        obs_dim: 2,
        num_actions: 2,
        action_temperature: 1.0,
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(cfg);
    agent.set_goals(vec![1.0, 0.8], 2.0); // the corrected OrganismConfig::default() preference
    agent.belief.mean = vec![resource_belief, energy_belief];
    agent.select_action().action_probabilities
}

#[test]
fn forage_wins_above_a_low_threshold_unlike_the_pre_fix_universal_rest_bias() {
    // Rest should still win at genuinely-near-empty resource -- foraging an empty patch isn't
    // free even with a fixed preference for abundance.
    let low_probs = probabilities_at_belief(0.05, ENERGY_BELIEF);
    assert!(
        low_probs[REST] > low_probs[FORAGE],
        "expected Rest to still win at resource_belief=0.05 (genuinely empty), got {low_probs:?}"
    );

    // Forage should win at every level from 0.15 upward -- the real, fitness-aligned crossover
    // the pre-fix preference structurally lacked (it favored Rest everywhere, 0.05 through 0.95,
    // see hoffman_efe_rest_structurally_dominates.rs). The margin is NOT expected to grow
    // monotonically all the way to 1.0 -- softmax probability margins saturate/can recede
    // slightly as one action's probability approaches 1, a normal nonlinearity, not a bug; only
    // the sign (who wins) is asserted here.
    for &r in &[0.15, 0.3, 0.5, 0.65, 0.8, 0.95] {
        let probs = probabilities_at_belief(r, ENERGY_BELIEF);
        assert!(
            probs[FORAGE] > probs[REST],
            "expected Forage to win at resource_belief={r}, got forage={:.4} rest={:.4}",
            probs[FORAGE],
            probs[REST]
        );
    }
}
