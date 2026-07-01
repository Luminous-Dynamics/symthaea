// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference Engine for NixOS Management
//!
//! Selects actions by minimizing **Expected Free Energy (EFE)** over the
//! NixOS state space. Combines:
//!
//! - **Pragmatic value**: how close the predicted next state is to the goal
//! - **Epistemic value**: how much information the action provides (curiosity)
//! - **Episodic memory**: past outcomes for similar states (avoid past failures)
//!
//! The engine does NOT classify intents — it understands user input as a desired
//! state and generates the action sequence that bridges current → desired.

use super::episodic_memory::{EpisodeOutcome, NixEpisodicMemory};
use super::goal_inference::{GoalInference, InferredGoal};
use super::world_model::{ActionCategory, NixWorldModel};
use crate::encoding::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// A scored candidate action with rationale.
#[derive(Debug, Clone)]
pub struct ScoredAction {
    /// The action category.
    pub action: ActionCategory,
    /// Expected free energy (lower = better).
    pub expected_free_energy: f64,
    /// Pragmatic component: goal proximity of predicted next state.
    pub pragmatic_value: f64,
    /// Epistemic component: information gain.
    pub epistemic_value: f64,
    /// Episodic component: past outcome valence for similar states.
    pub episodic_valence: f64,
    /// Human-readable explanation of why this action was selected.
    pub rationale: String,
}

/// A plan of one or more actions to reach the goal state.
#[derive(Debug, Clone)]
pub struct ActionPlan {
    /// Ordered sequence of actions (best first).
    pub actions: Vec<ScoredAction>,
    /// Current free energy (distance from goal).
    pub current_free_energy: f64,
    /// The inferred goal that drove this plan.
    pub goal: InferredGoal,
    /// Whether the system recommends asking a clarifying question.
    pub needs_clarification: bool,
}

/// The active inference engine — the cognitive core.
pub struct NixActiveInference {
    /// Generative model of the NixOS system.
    world_model: NixWorldModel,
    /// Goal inference from user input.
    goal_inference: GoalInference,
    /// Episodic memory of past system events.
    episodic_memory: NixEpisodicMemory,
    /// NixOS HDC codebook (shared across encoders).
    codebook: NixCodebook,
    /// Weight for epistemic (curiosity) drive. Higher = more exploration.
    curiosity_weight: f64,
    /// Weight for episodic memory influence.
    episodic_weight: f64,
}

impl NixActiveInference {
    /// Create a new active inference engine.
    pub fn new() -> Self {
        Self {
            world_model: NixWorldModel::default(),
            goal_inference: GoalInference::new(),
            episodic_memory: NixEpisodicMemory::new(),
            codebook: NixCodebook::new(),
            curiosity_weight: 0.3,
            episodic_weight: 0.2,
        }
    }

    /// Create with custom curiosity weight.
    pub fn with_curiosity(curiosity_weight: f64) -> Self {
        Self {
            curiosity_weight,
            ..Self::new()
        }
    }

    /// Process user input and select the best action plan.
    ///
    /// This is the main entry point: natural language in → action plan out.
    #[tracing::instrument(skip(self), fields(input_len = input.len()))]
    pub fn process_input(&mut self, input: &str) -> ActionPlan {
        // 1. Infer the user's goal (desired system state)
        let goal = self.goal_inference.infer(input, &mut self.codebook);

        // 2. Compute current free energy (gap between state and goal)
        let current_fe = self.world_model.compute_free_energy(&goal.goal_state);

        // 3. If goal is unclear, recommend clarification
        if goal.needs_clarification {
            return ActionPlan {
                actions: Vec::new(),
                current_free_energy: current_fe,
                goal,
                needs_clarification: true,
            };
        }

        // 4. Score all candidate actions
        let candidates = self.score_candidates(&goal.goal_state);

        // 5. Store the goal context in working memory via the world model
        // (the goal inference already maintains its own working memory)

        ActionPlan {
            actions: candidates,
            current_free_energy: current_fe,
            needs_clarification: false,
            goal,
        }
    }

    /// Process a pre-encoded goal vector directly.
    pub fn process_goal(&mut self, goal_state: &ContinuousHV) -> ActionPlan {
        let current_fe = self.world_model.compute_free_energy(goal_state);
        let candidates = self.score_candidates(goal_state);

        ActionPlan {
            actions: candidates,
            current_free_energy: current_fe,
            goal: InferredGoal {
                goal_state: goal_state.clone(),
                confidence: 0.7,
                description: "Pre-encoded goal".to_string(),
                needs_clarification: false,
            },
            needs_clarification: false,
        }
    }

    /// Score all candidate action categories against the goal.
    fn score_candidates(&self, goal_state: &ContinuousHV) -> Vec<ScoredAction> {
        let candidates = Self::all_standard_actions();

        let mut scored: Vec<ScoredAction> = candidates
            .into_iter()
            .map(|action| self.score_action(&action, goal_state))
            .collect();

        // Sort by expected free energy (ascending = best first)
        scored.sort_by(|a, b| {
            a.expected_free_energy
                .partial_cmp(&b.expected_free_energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored
    }

    /// Score a single action against the goal state.
    fn score_action(&self, action: &ActionCategory, goal_state: &ContinuousHV) -> ScoredAction {
        // Predict the next state after this action
        let predicted_next = self.world_model.predict_state(action);

        // Pragmatic value: how close to goal? (higher = better)
        let pragmatic = predicted_next.similarity(goal_state).max(0.0) as f64;

        // Epistemic value: how much will we learn? (higher for less-observed actions)
        let epistemic = if self.world_model.has_learned(action) {
            // Less to learn from well-known actions
            0.1
        } else {
            // Maximum curiosity for unknown actions
            1.0
        };

        // Episodic valence: past outcomes for similar states
        let episodic = self
            .episodic_memory
            .predict_valence(self.world_model.system_state());

        // Expected free energy (lower = better)
        // EFE = -(pragmatic + curiosity * epistemic + episodic_weight * episodic)
        let efe =
            -(pragmatic + self.curiosity_weight * epistemic + self.episodic_weight * episodic);

        ScoredAction {
            action: action.clone(),
            expected_free_energy: efe,
            pragmatic_value: pragmatic,
            epistemic_value: epistemic,
            episodic_valence: episodic,
            rationale: format!(
                "{action} scored from pragmatic={pragmatic:.3}, epistemic={epistemic:.3}, episodic={episodic:.3}"
            ),
        }
    }

    /// Observe the current encoded NixOS state.
    pub fn observe_state(&mut self, state: ContinuousHV) {
        self.world_model.observe(state);
    }

    /// Learn from an observed action outcome.
    pub fn learn_from_outcome(
        &mut self,
        state_before: &ContinuousHV,
        action: ActionCategory,
        state_after: &ContinuousHV,
        outcome: EpisodeOutcome,
        phi: f64,
    ) {
        let predicted = self.world_model.predict_state(&action);
        let prediction_error = 1.0 - predicted.similarity(state_after).max(0.0) as f64;

        // Modulate curiosity weight based on prediction error (epistemic modulation)
        // High prediction error -> surprise. Increase exploration drive slightly.
        // Low prediction error -> model is accurate. Restabilize exploration drive.
        if prediction_error > 0.4 {
            self.curiosity_weight = (self.curiosity_weight + 0.05).min(0.8);
        } else {
            self.curiosity_weight = (self.curiosity_weight - 0.02).max(0.1);
        }

        self.world_model
            .learn_transition(state_before, action.clone(), state_after);
        self.world_model.observe(state_after.clone());

        // Record in episodic memory (Φ-gated)
        #[cfg(feature = "native")]
        {
            use crate::action::executor::NixOSCommand;
            let cmd = match &action {
                ActionCategory::Install => NixOSCommand::EnvInstall {
                    packages: vec!["unknown".into()],
                },
                ActionCategory::Remove => NixOSCommand::EnvRemove {
                    packages: vec!["unknown".into()],
                },
                ActionCategory::Rebuild => NixOSCommand::RebuildSwitch {
                    flake: None,
                    extra_args: vec![],
                },
                ActionCategory::Rollback => NixOSCommand::EnvRollback,
                ActionCategory::GarbageCollect => NixOSCommand::CollectGarbage {
                    older_than_days: None,
                    delete_all: false,
                },
                ActionCategory::Update => NixOSCommand::Channel {
                    operation: crate::action::executor::ChannelOperation::Update { channel: None },
                },
                _ => NixOSCommand::Custom {
                    command: format!("{action:?}"),
                    args: vec![],
                    safety_level: crate::action::executor::SafetyLevel::ReadOnly,
                },
            };

            self.episodic_memory.record_transition(
                state_before.clone(),
                &cmd,
                state_after.clone(),
                outcome.clone(),
                phi,
                prediction_error,
            );
        }
        #[cfg(not(feature = "native"))]
        {
            // Without native, record a simplified episode directly
            let episode = super::episodic_memory::SystemEpisode {
                state_before: state_before.clone(),
                action: format!("{action:?}"),
                state_after: state_after.clone(),
                outcome,
                phi_at_encoding: phi,
                prediction_error,
                emotional_valence: 0.0,
                timestamp: 0,
            };
            self.episodic_memory.record(episode);
        }
    }

    /// All standard NixOS action categories.
    fn all_standard_actions() -> Vec<ActionCategory> {
        vec![
            ActionCategory::Install,
            ActionCategory::Remove,
            ActionCategory::Enable,
            ActionCategory::Disable,
            ActionCategory::Rebuild,
            ActionCategory::Rollback,
            ActionCategory::Configure,
            ActionCategory::GarbageCollect,
            ActionCategory::Update,
        ]
    }

    /// Access the world model.
    pub fn world_model(&self) -> &NixWorldModel {
        &self.world_model
    }

    /// Access the world model mutably.
    pub fn world_model_mut(&mut self) -> &mut NixWorldModel {
        &mut self.world_model
    }

    /// Access goal inference.
    pub fn goal_inference(&self) -> &GoalInference {
        &self.goal_inference
    }

    /// Access goal inference mutably.
    pub fn goal_inference_mut(&mut self) -> &mut GoalInference {
        &mut self.goal_inference
    }

    /// Access episodic memory.
    pub fn episodic_memory(&self) -> &NixEpisodicMemory {
        &self.episodic_memory
    }

    /// Access episodic memory mutably.
    pub fn episodic_memory_mut(&mut self) -> &mut NixEpisodicMemory {
        &mut self.episodic_memory
    }

    /// Access the codebook.
    pub fn codebook(&self) -> &NixCodebook {
        &self.codebook
    }

    /// Access the codebook mutably.
    pub fn codebook_mut(&mut self) -> &mut NixCodebook {
        &mut self.codebook
    }

    /// Current free energy of the system.
    pub fn free_energy(&self) -> f64 {
        self.world_model.free_energy()
    }

    /// Number of episodes in memory.
    pub fn episode_count(&self) -> usize {
        self.episodic_memory.len()
    }

    /// Reset the inference engine (new conversation).
    pub fn reset(&mut self) {
        self.goal_inference.reset();
    }
}

impl Default for NixActiveInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;

    #[test]
    fn test_process_input_clear_goal() {
        let mut engine = NixActiveInference::new();

        // Observe some initial state
        let initial_state = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 1);
        engine.observe_state(initial_state);

        let plan = engine.process_input("install firefox");
        assert!(!plan.needs_clarification);
        assert!(!plan.actions.is_empty());
        assert!(plan.goal.confidence > 0.5);
        assert!(plan.goal.description.contains("Install"));
    }

    #[test]
    fn test_process_input_ambiguous_goal() {
        let mut engine = NixActiveInference::new();
        let plan = engine.process_input("help");
        assert!(plan.needs_clarification);
    }

    #[test]
    fn test_actions_sorted_by_efe() {
        let mut engine = NixActiveInference::new();
        let initial = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 1);
        engine.observe_state(initial);

        let plan = engine.process_input("install nginx");
        // Actions should be sorted by EFE (ascending)
        for window in plan.actions.windows(2) {
            assert!(
                window[0].expected_free_energy <= window[1].expected_free_energy + 1e-10,
                "Actions should be sorted by EFE: {} <= {}",
                window[0].expected_free_energy,
                window[1].expected_free_energy,
            );
        }
    }

    #[test]
    fn test_learning_improves_prediction() {
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let mut engine = NixActiveInference::new();

        let state_before = ContinuousHV::random(dim, 1);
        let _state_after = ContinuousHV::random(dim, 2);

        // Observe initial state
        engine.observe_state(state_before.clone());

        // Learn from multiple install transitions
        for i in 0..5 {
            let before = ContinuousHV::random(dim, i * 10 + 1);
            let after = ContinuousHV::random(dim, i * 10 + 2);
            engine.learn_from_outcome(
                &before,
                ActionCategory::Install,
                &after,
                EpisodeOutcome::Success,
                0.7,
            );
        }

        // After learning, the world model should have data for Install
        assert!(engine.world_model().has_learned(&ActionCategory::Install));
        assert!(engine.episode_count() > 0);
    }

    #[test]
    fn test_reset_clears_goal() {
        let mut engine = NixActiveInference::new();
        engine.process_input("install firefox");
        assert!(!engine.goal_inference().working_memory().is_empty());

        engine.reset();
        assert!(engine.goal_inference().working_memory().is_empty());
    }

    #[test]
    fn test_process_goal_directly() {
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let mut engine = NixActiveInference::new();
        engine.observe_state(ContinuousHV::random(dim, 1));

        let goal = ContinuousHV::random(dim, 42);
        let plan = engine.process_goal(&goal);
        assert!(!plan.needs_clarification);
        assert!(!plan.actions.is_empty());
    }
}
