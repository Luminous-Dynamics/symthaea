// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! A single tissue-level active-inference agent driving regeneration
//! proliferation rate, replacing the ad hoc flat-rate rule in
//! [`crate::bioelectric`]'s `advance_regeneration` with genuine
//! free-energy-minimizing action selection.
//!
//! Uses `symthaea_fep`'s real `ActiveInferenceAgent`/`GenerativeModel` --
//! dense small-vector-based (not HDC), cheap enough for one decision per
//! simulated day. This is deliberately a single tissue-level agent, not a
//! per-cell or multi-level hierarchy: `symthaea_fep::HierarchicalFepManager`
//! turns out to be a 67-line, entirely bespoke 2-level example (two
//! hardcoded agent fields, a literal binary goal-propagation branch), not a
//! reusable abstraction to generalize, and per-cell agent instantiation at
//! this crate's scale (hundreds to `MAX_CELLS` cells) is real, unvalidated
//! extra risk this phase deliberately avoids. See module docs in
//! `crate::bioelectric` for how this plugs into `advance_regeneration`.

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

/// Discrete action -> proliferation-boost multiplier, applied on top of
/// `REGENERATION_PROLIFERATION_BOOST` in `crate::bioelectric`. Index 2
/// (1.0x) reproduces the legacy flat rate exactly.
const ACTION_BOOST_MULTIPLIERS: [f64; 4] = [0.0, 0.5, 1.0, 2.0];

/// Wraps a real `ActiveInferenceAgent` observing tissue-level regeneration
/// state and selecting a proliferation-boost multiplier each simulated day.
#[derive(Debug, Clone)]
pub(crate) struct RegenerationAgent {
    agent: ActiveInferenceAgent,
}

impl RegenerationAgent {
    pub(crate) fn new() -> Self {
        let config = ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 4,
            num_actions: ACTION_BOOST_MULTIPLIERS.len(),
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(config);
        // Prefer low discrepancy, low defected fraction, low wound-boundary
        // fraction, and fast healing (low days-since-wound fraction) -- all
        // four observation channels' preferred value is 0.0.
        agent.set_goals(vec![0.0, 0.0, 0.0, 0.0], 2.0);
        Self { agent }
    }

    /// One decision step: observe tissue state, select a proliferation-
    /// boost multiplier via expected-free-energy minimization, and let the
    /// agent learn from the outcome (model + TD learning are both enabled
    /// via `ActiveInferenceAgentConfig`'s defaults).
    pub(crate) fn decide(
        &mut self,
        discrepancy: f64,
        days_since_wound_frac: f64,
        defected_fraction: f64,
        wound_boundary_fraction: f64,
    ) -> f64 {
        let obs = Observation::new(
            vec![
                discrepancy,
                days_since_wound_frac,
                defected_fraction,
                wound_boundary_fraction,
            ],
            1.0,
            "regeneration_state",
        );
        self.agent.perceive(&obs);
        let result = self.agent.select_action();
        self.agent.act(result.action);
        self.agent.learn_from_outcome(result.action, &obs);
        ACTION_BOOST_MULTIPLIERS[result.action]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_returns_a_valid_multiplier() {
        let mut agent = RegenerationAgent::new();
        for _ in 0..10 {
            let m = agent.decide(0.3, 0.1, 0.0, 0.2);
            assert!(
                ACTION_BOOST_MULTIPLIERS.contains(&m),
                "decide() should always return one of the configured multipliers, got {m}"
            );
        }
    }

    #[test]
    fn decide_is_sensitive_to_observations_not_constant() {
        // Regression guard: a single agent instance, swept across a wide
        // range of discrepancy observations, should choose more than one
        // action -- if this ever collapses to a single constant multiplier
        // regardless of input, something in the perceive/select_action
        // wiring is broken (verified once, manually, against this exact
        // sweep: real variety across steps).
        let mut agent = RegenerationAgent::new();
        let mut seen = std::collections::HashSet::new();
        for i in 0..30 {
            let discrepancy = 0.1 + (i as f64 * 0.05) % 0.8;
            let m = agent.decide(discrepancy, 0.0, 0.0, 0.3);
            seen.insert(m.to_bits());
        }
        assert!(
            seen.len() > 1,
            "expected decide() to choose more than one action across a wide \
             observation sweep, got only: {seen:?}"
        );
    }
}
