// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical Active Inference — Scale-free cognitive coordination.

use crate::{ActiveInferenceAgent, Observation};

/// A hierarchical coordination of nested Active Inference agents.
///
/// Implements top-down goal propagation (empirical priors) and
/// bottom-up prediction error propagation.
pub struct HierarchicalFepManager {
    /// The higher-level "Cortex" agent (e.g. Town scale).
    pub cortex: ActiveInferenceAgent,
    /// The lower-level "Motor" agent (e.g. Limb scale).
    pub motor: ActiveInferenceAgent,
}

impl HierarchicalFepManager {
    /// Create a new hierarchical manager with two nested scales.
    pub fn new(cortex: ActiveInferenceAgent, motor: ActiveInferenceAgent) -> Self {
        Self { cortex, motor }
    }

    /// Execute one hierarchical step.
    ///
    /// 1. Cortex perceives global state and selects a "Goal Action".
    /// 2. Goal Action is translated into a prior for the Motor agent.
    /// 3. Motor agent perceives local state + Cortex prior to select real action.
    pub fn step(&mut self, global_obs: &Observation, local_obs: &Observation) -> usize {
        // 1. Higher Level Perception (The Town feels the world)
        let _cortex_perception = self.cortex.perceive(global_obs); // Reserved for downstream motor prior modulation
        let cortex_action = self.cortex.select_action();

        // 2. Goal Propagation (The Town tells the Limb what to "believe")
        // In a real implementation, we would use a mapping function.
        // Here, we translate cortex action into a motor prior mean.
        let mut goal_prior_mean = vec![0.5; self.motor.config.state_dim];
        if cortex_action.action == 0 {
            // e.g. "Conserve" action
            for val in goal_prior_mean.iter_mut() {
                *val = 0.1;
            } // Low activity prior
        } else {
            for val in goal_prior_mean.iter_mut() {
                *val = 0.8;
            } // High performance prior
        }

        // 3. Inject Top-Down Prior into the lower level
        let precision = vec![2.0; self.motor.config.state_dim]; // High precision goal
        self.motor.inject_priors(goal_prior_mean, precision);

        // 4. Lower Level Perception & Action (The Limb acts to fulfill the Town's prior)
        self.motor.perceive(local_obs);
        let motor_result = self.motor.select_action();

        tracing::info!(
            "🧠 Hierarchical Sync: Cortex (Act={}) -> Motor (Act={})",
            cortex_action.action,
            motor_result.action
        );

        motor_result.action
    }
}
