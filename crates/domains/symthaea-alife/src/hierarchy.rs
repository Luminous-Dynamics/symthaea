// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! N-level generalization of `symthaea_fep::hierarchical::HierarchicalFepManager`, per
//! `ALIFE_PLAN_2026-07-08.md` Phase 2 §2b.
//!
//! `HierarchicalFepManager` hardcodes exactly two levels ("cortex" and "motor"). A
//! coalition-of-coalitions (cells→organs→bodies→societies, the actual hierarchy Friston's
//! Markov-blanket-of-life framework describes) needs an arbitrary number of nested scales.
//! `HierarchicalStack` generalizes the same top-down-goal / bottom-up-action pattern to
//! `Vec<ActiveInferenceAgent>` of any depth -- level 0 is the highest scale, the last level is
//! the one that actually acts on the world.

use symthaea_fep::{ActiveInferenceAgent, Observation};

pub struct HierarchicalStack {
    /// Level 0 = highest scale (e.g. "coalition"), last = lowest (e.g. "individual organism").
    pub levels: Vec<ActiveInferenceAgent>,
}

impl HierarchicalStack {
    pub fn new(levels: Vec<ActiveInferenceAgent>) -> Self {
        Self { levels }
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// One step: every level perceives its own observation, then goals propagate top-down
    /// through each adjacent pair (same heuristic `HierarchicalFepManager::step` uses: action 0
    /// injects a low-activity prior, anything else a high-performance prior), and the bottom
    /// level's real action is returned.
    ///
    /// `observations.len()` must equal `self.depth()` -- one observation per level.
    pub fn step(&mut self, observations: &[Observation]) -> usize {
        assert_eq!(
            observations.len(),
            self.levels.len(),
            "one observation required per hierarchy level"
        );

        for (level, obs) in self.levels.iter_mut().zip(observations.iter()) {
            level.perceive(obs);
        }

        for i in 0..self.levels.len().saturating_sub(1) {
            let action = self.levels[i].select_action().action;
            let state_dim = self.levels[i + 1].config.state_dim;
            let goal_value = if action == 0 { 0.1 } else { 0.8 };
            let goal_mean = vec![goal_value; state_dim];
            let precision = vec![2.0; state_dim];
            self.levels[i + 1].inject_priors(goal_mean, precision);
        }

        self.levels
            .last_mut()
            .expect("HierarchicalStack must have at least one level")
            .select_action()
            .action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fep::ActiveInferenceAgentConfig;

    fn agent(state_dim: usize) -> ActiveInferenceAgent {
        ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim,
            obs_dim: state_dim,
            num_actions: 2,
            ..Default::default()
        })
    }

    #[test]
    fn propagates_priors_through_three_levels() {
        let mut stack = HierarchicalStack::new(vec![agent(2), agent(2), agent(2)]);
        assert_eq!(stack.depth(), 3);

        let obs = Observation::new(vec![0.5, 0.5], 1.0, "test");
        let observations = vec![obs.clone(), obs.clone(), obs];

        // Priors on levels 1 and 2 start at the generic default (0.5); after a step, top-down
        // goal propagation should have actually moved them, proving the cascade is real and not
        // a no-op stub.
        let before_level1_prior = stack.levels[1].model.prior_mean.clone();
        let before_level2_prior = stack.levels[2].model.prior_mean.clone();

        let action = stack.step(&observations);
        assert!(action < 2);

        assert_ne!(
            stack.levels[1].model.prior_mean, before_level1_prior,
            "level 1's prior should have been injected by level 0's goal propagation"
        );
        assert_ne!(
            stack.levels[2].model.prior_mean, before_level2_prior,
            "level 2's prior should have been injected by level 1's goal propagation"
        );
    }

    #[test]
    fn single_level_stack_just_acts() {
        let mut stack = HierarchicalStack::new(vec![agent(2)]);
        let obs = Observation::new(vec![0.5, 0.5], 1.0, "test");
        let action = stack.step(&[obs]);
        assert!(action < 2);
    }
}
