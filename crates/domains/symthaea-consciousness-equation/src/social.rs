// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;
// web-time: drop-in Instant for wasm32 (std::time::Instant panics on wasm32-unknown-unknown)
use web_time::Instant;

/// Social Embedding: Soc = other_modeling_accuracy × self_other_distinction
///
/// This factor measures the system's social cognition:
/// - Other modeling accuracy: Theory of Mind - predicting others' mental states
/// - Self-other distinction: Maintaining clear boundary between self and others
#[derive(Debug, Clone)]
pub struct SocialEmbedding {
    /// Theory of Mind accuracy (predicting others)
    other_modeling_accuracy: f64,

    /// Self-other distinction clarity
    self_other_distinction: f64,

    /// Models of other agents
    agent_models: Vec<AgentModel>,

    /// Self-model state
    self_model: SelfModel,

    /// Prediction history for ToM
    tom_predictions: VecDeque<ToMPrediction>,

    /// Smoothing factor
    smoothing: f64,

    /// Maximum models to maintain
    max_agents: usize,
}

/// Model of another agent's mental state
#[derive(Debug, Clone)]
pub struct AgentModel {
    /// Agent identifier
    pub id: String,
    /// Inferred beliefs
    pub beliefs: Vec<String>,
    /// Inferred goals
    pub goals: Vec<String>,
    /// Inferred emotional state
    pub emotional_state: f64, // -1 to 1
    /// Confidence in this model
    pub confidence: f64,
    /// Last update time
    pub last_update: Instant,
}

/// Self-model for self-other distinction
#[derive(Debug, Clone)]
pub struct SelfModel {
    /// Own goals
    pub goals: Vec<String>,
    /// Own beliefs
    pub beliefs: Vec<String>,
    /// Own emotional state
    pub emotional_state: f64,
    /// Self-coherence score
    pub coherence: f64,
}

impl Default for SelfModel {
    fn default() -> Self {
        Self {
            goals: Vec::new(),
            beliefs: Vec::new(),
            emotional_state: 0.0,
            coherence: 0.5,
        }
    }
}

/// A Theory of Mind prediction and its outcome
#[derive(Debug, Clone)]
struct ToMPrediction {
    /// Agent being predicted
    agent_id: String,
    /// What we predicted
    predicted_state: f64,
    /// What actually happened
    actual_state: Option<f64>,
    /// Timestamp
    #[allow(dead_code)] // Stored for potential temporal analysis
    timestamp: Instant,
}

impl Default for SocialEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl SocialEmbedding {
    pub fn new() -> Self {
        Self {
            other_modeling_accuracy: 0.5,
            self_other_distinction: 0.7,
            agent_models: Vec::new(),
            self_model: SelfModel::default(),
            tom_predictions: VecDeque::with_capacity(100),
            smoothing: 0.1,
            max_agents: 10,
        }
    }

    /// Update or create model of another agent
    pub fn update_agent_model(
        &mut self,
        id: &str,
        beliefs: Vec<String>,
        goals: Vec<String>,
        emotional_state: f64,
        confidence: f64,
    ) {
        // Check if agent exists
        if let Some(agent) = self.agent_models.iter_mut().find(|a| a.id == id) {
            agent.beliefs = beliefs;
            agent.goals = goals;
            agent.emotional_state = emotional_state;
            agent.confidence = confidence;
            agent.last_update = Instant::now();
        } else {
            // Create new model
            if self.agent_models.len() >= self.max_agents {
                // Remove oldest
                self.agent_models
                    .sort_by(|a, b| b.last_update.cmp(&a.last_update));
                self.agent_models.pop();
            }

            self.agent_models.push(AgentModel {
                id: id.to_string(),
                beliefs,
                goals,
                emotional_state,
                confidence,
                last_update: Instant::now(),
            });
        }

        // Update self-other distinction
        self.update_self_other_distinction();
    }

    /// Record a Theory of Mind prediction
    pub fn record_tom_prediction(&mut self, agent_id: &str, predicted_state: f64) {
        if self.tom_predictions.len() >= 100 {
            self.tom_predictions.pop_front();
        }

        self.tom_predictions.push_back(ToMPrediction {
            agent_id: agent_id.to_string(),
            predicted_state,
            actual_state: None,
            timestamp: Instant::now(),
        });
    }

    /// Provide feedback on a ToM prediction
    pub fn provide_tom_feedback(&mut self, agent_id: &str, actual_state: f64) {
        // Find most recent prediction for this agent
        for pred in self.tom_predictions.iter_mut().rev() {
            if pred.agent_id == agent_id && pred.actual_state.is_none() {
                pred.actual_state = Some(actual_state);

                // Update accuracy
                let error = (pred.predicted_state - actual_state).abs();
                let accuracy = 1.0 - error.min(1.0);

                self.other_modeling_accuracy = self.other_modeling_accuracy
                    * (1.0 - self.smoothing)
                    + accuracy * self.smoothing;

                break;
            }
        }
    }

    /// Update self-model
    pub fn update_self_model(
        &mut self,
        goals: Vec<String>,
        beliefs: Vec<String>,
        emotional_state: f64,
    ) {
        self.self_model.goals = goals;
        self.self_model.beliefs = beliefs;
        self.self_model.emotional_state = emotional_state;

        // Compute self-coherence
        self.self_model.coherence = self.compute_self_coherence();

        // Update self-other distinction
        self.update_self_other_distinction();
    }

    /// Compute self-coherence
    fn compute_self_coherence(&self) -> f64 {
        // Heuristic: more goals and beliefs = more coherent self-model
        let goal_factor = (self.self_model.goals.len() as f64 / 5.0).min(1.0);
        let belief_factor = (self.self_model.beliefs.len() as f64 / 10.0).min(1.0);

        (goal_factor + belief_factor) / 2.0
    }

    /// Update self-other distinction
    fn update_self_other_distinction(&mut self) {
        if self.agent_models.is_empty() {
            self.self_other_distinction = 0.7; // Default when no others
            return;
        }

        // Measure distinctiveness of self from other models
        let mut total_distinction = 0.0;
        let mut count = 0;

        for agent in &self.agent_models {
            // Check goal overlap
            let self_goals: std::collections::HashSet<_> = self.self_model.goals.iter().collect();
            let other_goals: std::collections::HashSet<_> = agent.goals.iter().collect();
            let goal_overlap = self_goals.intersection(&other_goals).count() as f64
                / self_goals.len().max(1) as f64;

            // Check belief overlap
            let self_beliefs: std::collections::HashSet<_> =
                self.self_model.beliefs.iter().collect();
            let other_beliefs: std::collections::HashSet<_> = agent.beliefs.iter().collect();
            let belief_overlap = self_beliefs.intersection(&other_beliefs).count() as f64
                / self_beliefs.len().max(1) as f64;

            // Distinction is inverse of overlap
            let distinction = 1.0 - (goal_overlap * 0.5 + belief_overlap * 0.5);
            total_distinction += distinction;
            count += 1;
        }

        if count > 0 {
            self.self_other_distinction = total_distinction / count as f64;
        }
    }

    /// Compute social embedding factor Soc
    pub fn compute(&self) -> f64 {
        self.other_modeling_accuracy * self.self_other_distinction
    }

    /// Get other modeling accuracy
    pub fn other_modeling_accuracy(&self) -> f64 {
        self.other_modeling_accuracy
    }

    /// Get self-other distinction
    pub fn self_other_distinction(&self) -> f64 {
        self.self_other_distinction
    }

    /// Get number of agent models
    pub fn agent_count(&self) -> usize {
        self.agent_models.len()
    }
}
