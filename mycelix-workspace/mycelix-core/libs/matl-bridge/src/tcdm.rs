// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust-Confidence-Decay Model (TCDM)
//!
//! TCDM tracks how trust evolves over time with:
//! - **Trust**: Current trust level based on history
//! - **Confidence**: How certain we are about the trust estimate
//! - **Decay**: Trust naturally decays without activity

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use mycelix_core_types::trust::TCDMState;
use crate::error::{MatlError, MatlResult};

/// Default decay rate (per day)
pub const DEFAULT_DECAY_RATE: f32 = 0.99;

/// Minimum confidence before trust becomes unreliable
pub const MIN_CONFIDENCE: f32 = 0.1;

/// Interaction types that affect trust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Successfully participated in FL round
    FLParticipation,
    /// Successfully validated others' contributions
    Validation,
    /// Contributed high-quality gradient
    HighQualityContribution,
    /// Contributed low-quality gradient
    LowQualityContribution,
    /// Failed to participate when expected
    MissedRound,
    /// Detected Byzantine behavior
    ByzantineBehavior,
    /// Successfully challenged invalid contribution
    ValidChallenge,
    /// Made invalid challenge
    InvalidChallenge,
}

impl InteractionType {
    /// Get the trust impact of this interaction type
    pub fn trust_impact(&self) -> f32 {
        match self {
            Self::FLParticipation => 0.1,
            Self::Validation => 0.15,
            Self::HighQualityContribution => 0.2,
            Self::LowQualityContribution => -0.1,
            Self::MissedRound => -0.15,
            Self::ByzantineBehavior => -0.5,
            Self::ValidChallenge => 0.25,
            Self::InvalidChallenge => -0.2,
        }
    }

    /// Get the confidence impact of this interaction type
    pub fn confidence_impact(&self) -> f32 {
        match self {
            Self::FLParticipation => 0.05,
            Self::Validation => 0.05,
            Self::HighQualityContribution => 0.08,
            Self::LowQualityContribution => 0.05,
            Self::MissedRound => 0.03,
            Self::ByzantineBehavior => 0.1,
            Self::ValidChallenge => 0.1,
            Self::InvalidChallenge => 0.05,
        }
    }

    /// Is this a positive interaction?
    pub fn is_positive(&self) -> bool {
        self.trust_impact() > 0.0
    }
}

/// An interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Agent involved
    pub agent_id: String,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Round number (if applicable)
    pub round: Option<u64>,
    /// Additional context
    pub context: String,
    /// Timestamp
    pub timestamp: i64,
}

impl Interaction {
    /// Create a new interaction
    pub fn new(agent_id: String, interaction_type: InteractionType) -> Self {
        Self {
            agent_id,
            interaction_type,
            round: None,
            context: String::new(),
            timestamp: current_timestamp(),
        }
    }

    /// Set round number
    pub fn with_round(mut self, round: u64) -> Self {
        self.round = Some(round);
        self
    }

    /// Set context
    pub fn with_context(mut self, context: String) -> Self {
        self.context = context;
        self
    }
}

/// TCDM Tracker for managing trust state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCDMTracker {
    /// Trust states by agent ID
    states: HashMap<String, TCDMState>,
    /// Decay rate per day
    decay_rate: f32,
    /// Interaction history (last N interactions)
    history: Vec<Interaction>,
    /// Maximum history size
    max_history: usize,
}

impl TCDMTracker {
    /// Create a new TCDM tracker
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            decay_rate: DEFAULT_DECAY_RATE,
            history: Vec::new(),
            max_history: 10000,
        }
    }

    /// Create with custom decay rate
    pub fn with_decay_rate(decay_rate: f32) -> Self {
        Self {
            decay_rate,
            ..Self::new()
        }
    }

    /// Get or create trust state for an agent
    pub fn get_or_create(&mut self, agent_id: &str) -> &mut TCDMState {
        if !self.states.contains_key(agent_id) {
            self.states.insert(agent_id.to_string(), TCDMState::initial());
        }
        self.states.get_mut(agent_id)
            .expect("agent_id entry just inserted above")
    }

    /// Get trust state for an agent (read-only)
    pub fn get(&self, agent_id: &str) -> Option<&TCDMState> {
        self.states.get(agent_id)
    }

    /// Record an interaction
    pub fn record_interaction(&mut self, interaction: Interaction) -> MatlResult<f32> {
        let agent_id = interaction.agent_id.clone();
        let current_time = interaction.timestamp;
        let impact = interaction.interaction_type.trust_impact();
        let confidence_impact = interaction.interaction_type.confidence_impact();

        // Store in history first
        self.history.push(interaction);
        let max_history = self.max_history;
        while self.history.len() > max_history {
            self.history.remove(0);
        }

        // Get or create state
        let state = self.get_or_create(&agent_id);

        // Apply time decay first
        state.apply_decay(current_time);

        // Update trust using EMA
        let alpha = 0.2; // Learning rate
        if impact > 0.0 {
            state.trust_level = state.trust_level + alpha * impact * (1.0 - state.trust_level);
        } else {
            state.trust_level = state.trust_level + alpha * impact * state.trust_level;
        }
        state.trust_level = state.trust_level.clamp(0.0, 1.0);

        // Update confidence
        state.confidence = (state.confidence + confidence_impact).min(1.0);
        state.interaction_count += 1;
        state.last_update = current_time;

        Ok(state.compute_tcdm_score())
    }

    /// Apply decay to all agents
    pub fn apply_global_decay(&mut self, current_time: i64) {
        for state in self.states.values_mut() {
            state.apply_decay(current_time);
        }
    }

    /// Get TCDM score for an agent
    pub fn get_tcdm_score(&self, agent_id: &str) -> MatlResult<f32> {
        self.states
            .get(agent_id)
            .map(|s| s.compute_tcdm_score())
            .ok_or(MatlError::AgentNotFound {
                agent_id: agent_id.to_string(),
            })
    }

    /// Get all agents
    pub fn agents(&self) -> Vec<&String> {
        self.states.keys().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> TCDMStats {
        let agent_count = self.states.len();
        let total_trust: f32 = self.states.values().map(|s| s.trust_level).sum();
        let total_confidence: f32 = self.states.values().map(|s| s.confidence).sum();

        let avg_trust = if agent_count > 0 {
            total_trust / agent_count as f32
        } else {
            0.0
        };
        let avg_confidence = if agent_count > 0 {
            total_confidence / agent_count as f32
        } else {
            0.0
        };

        let high_trust_count = self.states.values().filter(|s| s.trust_level >= 0.7).count();
        let low_trust_count = self.states.values().filter(|s| s.trust_level < 0.3).count();

        TCDMStats {
            agent_count,
            avg_trust,
            avg_confidence,
            high_trust_count,
            low_trust_count,
            total_interactions: self.history.len(),
        }
    }

    /// Get recent interactions for an agent
    pub fn recent_interactions(&self, agent_id: &str, limit: usize) -> Vec<&Interaction> {
        self.history
            .iter()
            .rev()
            .filter(|i| i.agent_id == agent_id)
            .take(limit)
            .collect()
    }
}

impl Default for TCDMTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// TCDM statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCDMStats {
    pub agent_count: usize,
    pub avg_trust: f32,
    pub avg_confidence: f32,
    pub high_trust_count: usize,
    pub low_trust_count: usize,
    pub total_interactions: usize,
}

/// Get current Unix timestamp
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_interaction_increases_trust() {
        let mut tracker = TCDMTracker::new();

        let initial = tracker.get_or_create("agent-1").trust_level;

        let interaction = Interaction::new(
            "agent-1".to_string(),
            InteractionType::HighQualityContribution,
        );
        tracker.record_interaction(interaction).unwrap();

        let final_trust = tracker.get("agent-1").unwrap().trust_level;
        assert!(final_trust > initial);
    }

    #[test]
    fn test_negative_interaction_decreases_trust() {
        let mut tracker = TCDMTracker::new();

        // Start with some trust
        let state = tracker.get_or_create("agent-1");
        state.trust_level = 0.8;

        let interaction = Interaction::new(
            "agent-1".to_string(),
            InteractionType::ByzantineBehavior,
        );
        tracker.record_interaction(interaction).unwrap();

        let final_trust = tracker.get("agent-1").unwrap().trust_level;
        assert!(final_trust < 0.8);
    }

    #[test]
    fn test_confidence_increases_with_interactions() {
        let mut tracker = TCDMTracker::new();

        let initial_confidence = tracker.get_or_create("agent-1").confidence;

        for _ in 0..5 {
            let interaction = Interaction::new(
                "agent-1".to_string(),
                InteractionType::FLParticipation,
            );
            tracker.record_interaction(interaction).unwrap();
        }

        let final_confidence = tracker.get("agent-1").unwrap().confidence;
        assert!(final_confidence > initial_confidence);
    }

    #[test]
    fn test_tcdm_score_calculation() {
        let mut tracker = TCDMTracker::new();

        let state = tracker.get_or_create("agent-1");
        state.trust_level = 0.8;
        state.confidence = 0.5;

        let score = tracker.get_tcdm_score("agent-1").unwrap();
        assert!((score - 0.4).abs() < 0.01); // 0.8 * 0.5 = 0.4
    }
}
