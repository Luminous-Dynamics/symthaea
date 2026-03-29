// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FL-Agent Bridge
//!
//! Connects Federated Learning outcomes to agent K-Vector updates.
//! Enables AI agents participating in FL to have their trust profiles
//! updated based on gradient quality and Byzantine detection results.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                 FL → Agent Trust Pipeline                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  FL Round                Agent                   KREDIT          │
//! │  ┌──────────┐           ┌──────────┐            ┌──────────┐   │
//! │  │ Gradient │──────────▶│ K-Vector │───────────▶│ Cap      │   │
//! │  │ Quality  │  Bridge   │ Update   │  Derive    │ Adjust   │   │
//! │  │ Signals  │           │          │            │          │   │
//! │  └──────────┘           └──────────┘            └──────────┘   │
//! │       │                      ▲                                  │
//! │       ▼                      │                                  │
//! │  ┌──────────┐           ┌──────────┐                           │
//! │  │ Byzantine│───────────│ Penalty  │                           │
//! │  │ Detection│           │ Apply    │                           │
//! │  └──────────┘           └──────────┘                           │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::agentic::fl_bridge::{
//!     FLAgentBridge, apply_fl_feedback_to_agent
//! };
//! use mycelix_sdk::fl::matl_feedback::FLMatlFeedback;
//!
//! // After FL round completes
//! let feedback: FLMatlFeedback = compute_round_feedback(...);
//!
//! // Apply to agent
//! apply_fl_feedback_to_agent(&mut agent, &feedback, sponsor_civ);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{kredit::calculate_kredit_cap_from_trust, ActionOutcome, InstrumentalActor};
use crate::fl::matl_feedback::{FLMatlFeedback, GradientQualitySignals, KVectorDelta};
use crate::matl::KVector;

/// Configuration for FL-Agent bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLAgentBridgeConfig {
    /// Maximum reputation change per FL round
    pub max_reputation_delta: f32,
    /// Whether to immediately update KREDIT cap after K-Vector change
    pub auto_update_kredit: bool,
    /// Minimum rounds before reputation changes take full effect
    pub reputation_warmup_rounds: u32,
    /// Scale factor for Byzantine penalties
    pub byzantine_penalty_scale: f32,
    /// Scale factor for quality rewards
    pub quality_reward_scale: f32,
}

impl Default for FLAgentBridgeConfig {
    fn default() -> Self {
        Self {
            max_reputation_delta: 0.1,
            auto_update_kredit: true,
            reputation_warmup_rounds: 5,
            byzantine_penalty_scale: 1.0,
            quality_reward_scale: 1.0,
        }
    }
}

/// Result of applying FL feedback to an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLAgentUpdateResult {
    /// Agent ID
    pub agent_id: String,
    /// K-Vector before update
    pub k_vector_before: KVector,
    /// K-Vector after update
    pub k_vector_after: KVector,
    /// Trust score before
    pub trust_before: f32,
    /// Trust score after
    pub trust_after: f32,
    /// KREDIT cap before (if updated)
    pub kredit_cap_before: Option<u64>,
    /// KREDIT cap after (if updated)
    pub kredit_cap_after: Option<u64>,
    /// Applied delta
    pub applied_delta: KVectorDelta,
    /// Whether this was a penalty
    pub was_penalty: bool,
}

/// Bridge for connecting FL outcomes to agent updates
pub struct FLAgentBridge {
    config: FLAgentBridgeConfig,
    /// Track participation count per agent for warmup
    participation_counts: HashMap<String, u32>,
}

impl FLAgentBridge {
    /// Create a new bridge with default config
    pub fn new() -> Self {
        Self::with_config(FLAgentBridgeConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: FLAgentBridgeConfig) -> Self {
        Self {
            config,
            participation_counts: HashMap::new(),
        }
    }

    /// Apply FL round feedback to an agent
    pub fn apply_feedback(
        &mut self,
        agent: &mut InstrumentalActor,
        feedback: &FLMatlFeedback,
        sponsor_civ: f64,
    ) -> Option<FLAgentUpdateResult> {
        let agent_id = agent.agent_id.as_str().to_string();

        // Get delta for this agent
        let delta = feedback.kvector_deltas.get(&agent_id)?;

        // Track participation
        let count = self
            .participation_counts
            .entry(agent_id.clone())
            .or_insert(0);
        *count += 1;

        // Calculate warmup factor (gradual ramp-up of reputation changes)
        let warmup_factor = if *count < self.config.reputation_warmup_rounds {
            *count as f32 / self.config.reputation_warmup_rounds as f32
        } else {
            1.0
        };

        // Scale the delta
        let scaled_delta = self.scale_delta(delta, warmup_factor);

        // Capture before state
        let k_vector_before = agent.k_vector;
        let trust_before = k_vector_before.trust_score();
        let kredit_cap_before = if self.config.auto_update_kredit {
            Some(agent.kredit_cap)
        } else {
            None
        };

        // Apply the delta
        agent.k_vector = scaled_delta.apply(&agent.k_vector);

        let trust_after = agent.k_vector.trust_score();

        // Update KREDIT cap if configured
        let kredit_cap_after = if self.config.auto_update_kredit {
            match calculate_kredit_cap_from_trust(&agent.k_vector, agent.agent_class, sponsor_civ) {
                Ok(new_cap) => {
                    agent.kredit_cap = new_cap;
                    Some(new_cap)
                }
                Err(_) => kredit_cap_before,
            }
        } else {
            None
        };

        // Record the action
        let outcome = if scaled_delta.is_penalty {
            ActionOutcome::ConstraintViolation
        } else {
            ActionOutcome::Success
        };
        agent.record_action("fl_round_participation", 0, outcome);

        Some(FLAgentUpdateResult {
            agent_id: agent_id.clone(),
            k_vector_before,
            k_vector_after: agent.k_vector,
            trust_before,
            trust_after,
            kredit_cap_before,
            kredit_cap_after,
            applied_delta: scaled_delta,
            was_penalty: delta.is_penalty,
        })
    }

    /// Scale a delta based on config and warmup
    fn scale_delta(&self, delta: &KVectorDelta, warmup_factor: f32) -> KVectorDelta {
        let scale = if delta.is_penalty {
            self.config.byzantine_penalty_scale
        } else {
            self.config.quality_reward_scale
        } * warmup_factor;

        let mut scaled = delta.clone();

        // Scale all deltas
        scaled.reputation_delta = (scaled.reputation_delta * scale).clamp(
            -self.config.max_reputation_delta,
            self.config.max_reputation_delta,
        );
        scaled.activity_delta *= scale;
        scaled.integrity_delta *= scale;
        scaled.performance_delta *= scale;
        scaled.historical_delta *= scale;
        scaled.coherence_delta *= scale;

        scaled
    }

    /// Get participation count for an agent
    pub fn participation_count(&self, agent_id: &str) -> u32 {
        self.participation_counts
            .get(agent_id)
            .copied()
            .unwrap_or(0)
    }

    /// Reset participation tracking
    pub fn reset_participation(&mut self) {
        self.participation_counts.clear();
    }

    /// Get current configuration
    pub fn config(&self) -> &FLAgentBridgeConfig {
        &self.config
    }
}

impl Default for FLAgentBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply FL feedback to an agent (convenience function)
pub fn apply_fl_feedback_to_agent(
    agent: &mut InstrumentalActor,
    feedback: &FLMatlFeedback,
    sponsor_civ: f64,
) -> Option<FLAgentUpdateResult> {
    let mut bridge = FLAgentBridge::new();
    bridge.apply_feedback(agent, feedback, sponsor_civ)
}

/// Apply FL feedback to multiple agents
pub fn apply_fl_feedback_to_agents(
    agents: &mut HashMap<String, InstrumentalActor>,
    feedback: &FLMatlFeedback,
    sponsor_civ: f64,
) -> Vec<FLAgentUpdateResult> {
    let mut bridge = FLAgentBridge::new();
    let mut results = Vec::new();

    for (agent_id, agent) in agents.iter_mut() {
        if feedback.kvector_deltas.contains_key(agent_id) {
            if let Some(result) = bridge.apply_feedback(agent, feedback, sponsor_civ) {
                results.push(result);
            }
        }
    }

    results
}

/// Create a K-Vector delta from gradient quality analysis
pub fn delta_from_gradient_quality(
    agent_id: &str,
    signals: &GradientQualitySignals,
    was_included: bool,
) -> KVectorDelta {
    KVectorDelta::from_quality_signals(agent_id, signals, was_included)
}

/// Summary of FL round impact on agents
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FLRoundAgentImpact {
    /// Total agents affected
    pub agents_affected: usize,
    /// Agents with improved trust
    pub trust_improved: usize,
    /// Agents with degraded trust
    pub trust_degraded: usize,
    /// Agents with unchanged trust
    pub trust_unchanged: usize,
    /// Average trust change
    pub avg_trust_change: f32,
    /// Total KREDIT cap change (positive = increase)
    pub total_kredit_change: i64,
}

impl FLRoundAgentImpact {
    /// Compute from update results
    pub fn from_results(results: &[FLAgentUpdateResult]) -> Self {
        if results.is_empty() {
            return Self::default();
        }

        let mut impact = Self {
            agents_affected: results.len(),
            ..Default::default()
        };

        let mut total_change = 0.0f32;

        for result in results {
            let trust_change = result.trust_after - result.trust_before;
            total_change += trust_change;

            if trust_change > 0.001 {
                impact.trust_improved += 1;
            } else if trust_change < -0.001 {
                impact.trust_degraded += 1;
            } else {
                impact.trust_unchanged += 1;
            }

            if let (Some(before), Some(after)) = (result.kredit_cap_before, result.kredit_cap_after)
            {
                impact.total_kredit_change += after as i64 - before as i64;
            }
        }

        impact.avg_trust_change = total_change / results.len() as f32;
        impact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::{
        AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats, UncertaintyCalibration,
    };
    use crate::fl::matl_feedback::FeedbackStats;

    fn create_test_agent(id: &str) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 1000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    fn create_test_feedback(agent_id: &str, positive: bool) -> FLMatlFeedback {
        let mut kvector_deltas = HashMap::new();

        let delta = if positive {
            KVectorDelta {
                participant_id: agent_id.to_string(),
                reputation_delta: 0.05,
                activity_delta: 0.01,
                integrity_delta: 0.02,
                performance_delta: 0.03,
                historical_delta: 0.01,
                coherence_delta: 0.0,
                reason: "Good gradient quality".to_string(),
                is_penalty: false,
            }
        } else {
            KVectorDelta {
                participant_id: agent_id.to_string(),
                reputation_delta: -0.08,
                activity_delta: 0.0,
                integrity_delta: -0.05,
                performance_delta: -0.03,
                historical_delta: -0.02,
                coherence_delta: 0.0,
                reason: "Byzantine behavior detected".to_string(),
                is_penalty: true,
            }
        };

        kvector_deltas.insert(agent_id.to_string(), delta);

        FLMatlFeedback {
            round_id: 1,
            kvector_deltas,
            quality_signals: HashMap::new(),
            stats: FeedbackStats::default(),
        }
    }

    #[test]
    fn test_apply_positive_feedback() {
        let mut agent = create_test_agent("agent-1");
        let feedback = create_test_feedback("agent-1", true);

        let trust_before = agent.k_vector.trust_score();

        let result = apply_fl_feedback_to_agent(&mut agent, &feedback, 0.8);

        assert!(result.is_some());
        let result = result.unwrap();

        assert!(!result.was_penalty);
        assert!(result.trust_after > result.trust_before);
        assert!(agent.k_vector.trust_score() > trust_before);
    }

    #[test]
    fn test_apply_negative_feedback() {
        let mut agent = create_test_agent("agent-2");
        // Start with some reputation
        agent.k_vector.k_r = 0.7;

        let feedback = create_test_feedback("agent-2", false);

        let trust_before = agent.k_vector.trust_score();

        let result = apply_fl_feedback_to_agent(&mut agent, &feedback, 0.8);

        assert!(result.is_some());
        let result = result.unwrap();

        assert!(result.was_penalty);
        assert!(result.trust_after < result.trust_before);
        assert!(agent.k_vector.trust_score() < trust_before);
    }

    #[test]
    fn test_warmup_scaling() {
        let mut bridge = FLAgentBridge::with_config(FLAgentBridgeConfig {
            reputation_warmup_rounds: 5,
            ..Default::default()
        });

        let mut agent = create_test_agent("agent-3");
        let feedback = create_test_feedback("agent-3", true);

        // First round - should have reduced effect
        let result1 = bridge.apply_feedback(&mut agent, &feedback, 0.8).unwrap();
        let change1 = result1.trust_after - result1.trust_before;

        // Reset agent for comparison
        agent.k_vector = KVector::new_participant();

        // Simulate 4 more rounds to complete warmup
        for _ in 0..4 {
            bridge.apply_feedback(&mut agent, &feedback, 0.8);
            agent.k_vector = KVector::new_participant();
        }

        // After warmup (6th round) - should have full effect
        let result2 = bridge.apply_feedback(&mut agent, &feedback, 0.8).unwrap();
        let change2 = result2.trust_after - result2.trust_before;

        // Full warmup should have larger effect
        assert!(
            change2 > change1,
            "Full warmup ({}) should have larger effect than initial ({})",
            change2,
            change1
        );
    }

    #[test]
    fn test_kredit_cap_update() {
        let mut agent = create_test_agent("agent-4");
        agent.k_vector = KVector::new(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4);

        // Get trust-derived initial cap
        let initial_trust = agent.k_vector.trust_score();
        let initial_cap =
            calculate_kredit_cap_from_trust(&agent.k_vector, agent.agent_class, 0.8).unwrap();
        agent.kredit_cap = initial_cap;

        // Use a single bridge to properly accumulate participation count
        let mut bridge = FLAgentBridge::with_config(FLAgentBridgeConfig {
            reputation_warmup_rounds: 1, // Skip warmup
            ..Default::default()
        });

        // Apply positive feedback multiple times to increase trust
        for _ in 0..10 {
            let feedback = create_test_feedback("agent-4", true);
            bridge.apply_feedback(&mut agent, &feedback, 0.8);
        }

        let final_trust = agent.k_vector.trust_score();

        // Trust should have increased
        assert!(
            final_trust > initial_trust,
            "Trust should increase: {} > {}",
            final_trust,
            initial_trust
        );

        // KREDIT cap should have increased with trust
        assert!(
            agent.kredit_cap > initial_cap,
            "KREDIT cap should increase with trust: {} > {}",
            agent.kredit_cap,
            initial_cap
        );
    }

    #[test]
    fn test_apply_to_multiple_agents() {
        let mut agents = HashMap::new();
        agents.insert("agent-a".to_string(), create_test_agent("agent-a"));
        agents.insert("agent-b".to_string(), create_test_agent("agent-b"));
        agents.insert("agent-c".to_string(), create_test_agent("agent-c"));

        // Create feedback for only 2 agents
        let mut feedback = create_test_feedback("agent-a", true);
        feedback.kvector_deltas.insert(
            "agent-b".to_string(),
            KVectorDelta {
                participant_id: "agent-b".to_string(),
                reputation_delta: -0.05,
                activity_delta: 0.0,
                integrity_delta: -0.02,
                performance_delta: -0.01,
                historical_delta: 0.0,
                coherence_delta: 0.0,
                reason: "Poor quality".to_string(),
                is_penalty: true,
            },
        );

        let results = apply_fl_feedback_to_agents(&mut agents, &feedback, 0.8);

        // Should have 2 results (agent-a and agent-b, but not agent-c)
        assert_eq!(results.len(), 2);

        let impact = FLRoundAgentImpact::from_results(&results);
        assert_eq!(impact.agents_affected, 2);
        assert_eq!(impact.trust_improved, 1);
        assert_eq!(impact.trust_degraded, 1);
    }

    #[test]
    fn test_max_reputation_delta_clamping() {
        let config = FLAgentBridgeConfig {
            max_reputation_delta: 0.02,  // Very restrictive
            reputation_warmup_rounds: 1, // Skip warmup
            ..Default::default()
        };

        let mut bridge = FLAgentBridge::with_config(config);

        // Create feedback with large delta
        let mut feedback = FLMatlFeedback {
            round_id: 1,
            kvector_deltas: HashMap::new(),
            quality_signals: HashMap::new(),
            stats: FeedbackStats::default(),
        };
        feedback.kvector_deltas.insert(
            "agent-x".to_string(),
            KVectorDelta {
                participant_id: "agent-x".to_string(),
                reputation_delta: 0.5, // Large delta
                activity_delta: 0.0,
                integrity_delta: 0.0,
                performance_delta: 0.0,
                historical_delta: 0.0,
                coherence_delta: 0.0,
                reason: "Test".to_string(),
                is_penalty: false,
            },
        );

        let mut agent = create_test_agent("agent-x");
        let result = bridge.apply_feedback(&mut agent, &feedback, 0.8).unwrap();

        // Should be clamped to max
        assert!(
            result.applied_delta.reputation_delta <= 0.02,
            "Delta should be clamped: {}",
            result.applied_delta.reputation_delta
        );
    }
}
