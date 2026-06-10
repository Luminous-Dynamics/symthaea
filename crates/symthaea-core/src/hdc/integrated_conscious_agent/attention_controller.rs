// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Enhanced Self-Directed Attention Control

use super::super::attention_dynamics::AttentionMode;
use super::agent::IntegratedConsciousAgent;
use super::types::AttentionControlStatus;

use std::collections::VecDeque;

/// Enhanced attention control that enables metacognitive direction of attention
pub struct SelfDirectedAttentionController {
    /// Prediction error history (for curiosity-driven attention)
    prediction_errors: VecDeque<f64>,
    /// Habituation tracker - how long attention has been on each target
    habituation: std::collections::HashMap<String, HabituationState>,
    /// Current attention strategy
    strategy: AttentionStrategy,
    /// Exploration vs exploitation balance (0 = pure exploit, 1 = pure explore)
    exploration_rate: f64,
    /// Fatigue accumulator
    fatigue: f64,
    /// Recovery timer
    recovery_countdown: usize,
}

/// Habituation state for a target
#[derive(Clone, Debug)]
pub struct HabituationState {
    /// How many steps focused on this target
    exposure: usize,
    /// Current habituation level (0 = fresh, 1 = fully habituated)
    level: f64,
    /// Time since last exposure
    time_since: usize,
}

/// Attention strategy selection
#[derive(Clone, Debug, PartialEq)]
pub enum AttentionStrategy {
    /// Follow current goals
    GoalDirected,
    /// Attend to surprising/novel stimuli
    NoveltyDriven,
    /// Explore the environment
    Exploratory,
    /// Rest and recover
    Recovery,
    /// Balanced between goal and novelty
    Balanced,
}

impl SelfDirectedAttentionController {
    pub fn new() -> Self {
        Self {
            prediction_errors: VecDeque::with_capacity(50),
            habituation: std::collections::HashMap::new(),
            strategy: AttentionStrategy::Balanced,
            exploration_rate: 0.2,
            fatigue: 0.0,
            recovery_countdown: 0,
        }
    }

    /// Update controller with new prediction error
    pub fn update(&mut self, prediction_error: f64, focused_target: Option<&str>) {
        // Track prediction error
        self.prediction_errors.push_back(prediction_error);
        if self.prediction_errors.len() > 50 {
            self.prediction_errors.pop_front();
        }

        // Update habituation
        for (_, state) in self.habituation.iter_mut() {
            state.time_since += 1;
            // Recover from habituation when not attending
            state.level = (state.level - 0.02).max(0.0);
        }

        // Update habituation for current target
        if let Some(target) = focused_target {
            let state = self
                .habituation
                .entry(target.to_string())
                .or_insert(HabituationState {
                    exposure: 0,
                    level: 0.0,
                    time_since: 0,
                });
            state.exposure += 1;
            state.time_since = 0;
            // Habituation increases with sustained attention
            state.level = (state.level + 0.05).min(1.0);
        }

        // Update fatigue
        if self.recovery_countdown > 0 {
            self.recovery_countdown -= 1;
            self.fatigue = (self.fatigue - 0.1).max(0.0);
        } else {
            self.fatigue = (self.fatigue + 0.02).min(1.0);
        }

        // Select strategy
        self.update_strategy();
    }

    /// Select appropriate attention strategy
    fn update_strategy(&mut self) {
        // Check if recovery needed
        if self.fatigue > 0.8 {
            self.strategy = AttentionStrategy::Recovery;
            self.recovery_countdown = 5;
            return;
        }

        // Compute average prediction error
        let avg_error = if self.prediction_errors.is_empty() {
            0.5
        } else {
            self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
        };

        // High prediction error: switch to novelty-driven
        if avg_error > 0.7 {
            self.strategy = AttentionStrategy::NoveltyDriven;
            return;
        }

        // Low prediction error for long time: explore
        if avg_error < 0.2 && self.prediction_errors.len() > 20 {
            let recent_errors: Vec<_> = self.prediction_errors.iter().rev().take(10).collect();
            let all_low = recent_errors.iter().all(|&&e| e < 0.3);
            if all_low {
                self.strategy = AttentionStrategy::Exploratory;
                return;
            }
        }

        // Default to balanced
        self.strategy = AttentionStrategy::Balanced;
    }

    /// Get attention weight adjustment for a target
    pub fn get_weight_adjustment(&self, target_name: &str, base_priority: f64) -> f64 {
        let habituation_penalty = self
            .habituation
            .get(target_name)
            .map(|h| h.level)
            .unwrap_or(0.0);

        let strategy_modifier = match self.strategy {
            AttentionStrategy::GoalDirected => 1.0,
            AttentionStrategy::NoveltyDriven => 0.5, // Reduce goal-directed weight
            AttentionStrategy::Exploratory => 0.3,
            AttentionStrategy::Recovery => 0.1,
            AttentionStrategy::Balanced => 0.8,
        };

        // Reduce priority based on habituation
        base_priority * strategy_modifier * (1.0 - habituation_penalty * 0.5)
    }

    /// Should we attend to this novel stimulus?
    pub fn should_attend_novel(&self, novelty: f64) -> bool {
        match self.strategy {
            AttentionStrategy::NoveltyDriven => novelty > 0.3,
            AttentionStrategy::Exploratory => novelty > 0.2,
            AttentionStrategy::Balanced => novelty > 0.6,
            _ => novelty > 0.8, // Very high novelty always captures attention
        }
    }

    /// Get current exploration rate
    pub fn exploration_rate(&self) -> f64 {
        match self.strategy {
            AttentionStrategy::Exploratory => 0.8,
            AttentionStrategy::NoveltyDriven => 0.5,
            AttentionStrategy::Balanced => self.exploration_rate,
            AttentionStrategy::Recovery => 0.1,
            AttentionStrategy::GoalDirected => 0.1,
        }
    }

    /// Get current strategy
    pub fn strategy(&self) -> &AttentionStrategy {
        &self.strategy
    }

    /// Get current fatigue level
    pub fn fatigue(&self) -> f64 {
        self.fatigue
    }

    /// Force a specific strategy (for metacognitive override)
    pub fn set_strategy(&mut self, strategy: AttentionStrategy) {
        let is_recovery = strategy == AttentionStrategy::Recovery;
        self.strategy = strategy;
        if is_recovery {
            self.recovery_countdown = 10;
        }
    }
}

impl Default for SelfDirectedAttentionController {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional IntegratedConsciousAgent methods for attention control
// ═══════════════════════════════════════════════════════════════════════════

impl IntegratedConsciousAgent {
    /// Get detailed attention control status
    pub fn attention_control_status(&self) -> AttentionControlStatus {
        // Note: Self-directed attention controller would need to be added to struct
        // This provides a summary based on current attention state
        let intro = self.introspect();

        AttentionControlStatus {
            current_mode: intro.attention_mode,
            num_goals: intro.num_active_goals,
            is_goal_directed: intro.num_active_goals > 0,
            stream_support: intro.stream_coherence > 0.5,
            phi_support: intro.believed_phi > 0.4,
        }
    }

    /// Adjust goal priorities based on recent success
    pub fn adapt_goal_priorities(&mut self, success_signals: &[(String, f64)]) {
        for (goal_name, success) in success_signals {
            for goal in &mut self.goals {
                if goal.name == *goal_name {
                    // Increase priority for successful goals, decrease for unsuccessful
                    let adjustment = (success - 0.5) * 0.1;
                    goal.priority = (goal.priority + adjustment).clamp(0.1, 1.0);
                }
            }
        }
    }

    /// Set exploration mode for curiosity-driven attention
    pub fn set_exploration_mode(&mut self, explore: bool) {
        if explore {
            // Reduce goal priorities temporarily
            for goal in &mut self.goals {
                goal.priority *= 0.5;
            }
        } else {
            // Restore goal priorities
            for goal in &mut self.goals {
                goal.priority = (goal.priority * 2.0).min(1.0);
            }
        }
    }

    /// Metacognitive attention override
    pub fn metacognitive_attention_override(&mut self, force_mode: AttentionMode) {
        // This allows the self-model to directly control attention
        // Useful when the agent "decides" to focus or rest
        match force_mode {
            AttentionMode::Spotlight => {
                // Force high focus on highest priority goal
                if let Some(goal) = self
                    .goals
                    .iter()
                    .filter(|g| g.active)
                    .max_by(|a, b| a.priority.total_cmp(&b.priority))
                {
                    self.attention.add_target(goal.target.clone(), 1.0);
                }
            }
            AttentionMode::Diffuse => {
                // Clear all specific targets for broad awareness
                // (Would need AttentionDynamics.clear_targets())
            }
            _ => {}
        }
    }
}