// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Main MATL Bridge implementation
//!
//! Combines PoGQ, TCDM, and Entropy into a unified trust computation system.

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use mycelix_core_types::{KVector, TrustScore};
use crate::entropy::EntropyScore;
use crate::error::{MatlError, MatlResult};
use crate::pogq::{GradientContribution, PoGQOracle};
use crate::sync::{TrustRecord, TrustSource, TrustStore};
use crate::tcdm::{Interaction, InteractionType, TCDMTracker};
use crate::{ENTROPY_WEIGHT, POGQ_WEIGHT, TCDM_WEIGHT};

/// MATL Bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlBridgeConfig {
    /// Weight for PoGQ component
    pub pogq_weight: f32,
    /// Weight for TCDM component
    pub tcdm_weight: f32,
    /// Weight for Entropy component
    pub entropy_weight: f32,
    /// TCDM decay rate
    pub decay_rate: f32,
    /// PoGQ epsilon requirement
    pub pogq_epsilon: f32,
    /// PoGQ sigma requirement
    pub pogq_sigma: f32,
}

impl Default for MatlBridgeConfig {
    fn default() -> Self {
        Self {
            pogq_weight: POGQ_WEIGHT,
            tcdm_weight: TCDM_WEIGHT,
            entropy_weight: ENTROPY_WEIGHT,
            decay_rate: 0.99,
            pogq_epsilon: 10.0,
            pogq_sigma: 5.0,
        }
    }
}

/// The main MATL Bridge
#[derive(Debug)]
pub struct MatlBridge {
    /// Configuration
    config: MatlBridgeConfig,
    /// PoGQ Oracle for gradient validation
    pogq_oracle: PoGQOracle,
    /// TCDM Tracker for trust evolution
    tcdm_tracker: TCDMTracker,
    /// Trust store for syncing
    trust_store: TrustStore,
    /// Current agent states
    agent_states: std::collections::HashMap<String, AgentState>,
}

impl MatlBridge {
    /// Create a new MATL bridge with default config
    pub fn new() -> Self {
        Self::with_config(MatlBridgeConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: MatlBridgeConfig) -> Self {
        Self {
            pogq_oracle: PoGQOracle::with_config(
                config.pogq_epsilon,
                config.pogq_sigma,
                10.0, // max norm
            ),
            tcdm_tracker: TCDMTracker::with_decay_rate(config.decay_rate),
            trust_store: TrustStore::new(),
            agent_states: std::collections::HashMap::new(),
            config,
        }
    }

    /// Compute trust score for an agent
    ///
    /// Formula: T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
    pub fn compute_trust(&self, agent_id: &str) -> MatlResult<TrustScore> {
        let state = self.agent_states.get(agent_id).ok_or(MatlError::AgentNotFound {
            agent_id: agent_id.to_string(),
        })?;

        let pogq = state.last_pogq_score;
        let tcdm = self.tcdm_tracker.get_tcdm_score(agent_id).unwrap_or(0.5);
        let entropy = state.entropy_score.score();

        let total = self.config.pogq_weight * pogq
            + self.config.tcdm_weight * tcdm
            + self.config.entropy_weight * entropy;

        Ok(TrustScore {
            pogq,
            tcdm,
            entropy,
            total,
        })
    }

    /// Process a gradient contribution
    pub fn process_gradient(&mut self, contribution: GradientContribution) -> MatlResult<TrustScore> {
        let agent_id = contribution.agent_id.clone();

        // Validate via PoGQ
        let pogq_metrics = self.pogq_oracle.validate(&contribution)?;

        debug!(
            agent_id = %agent_id,
            quality = pogq_metrics.quality_score,
            "Validated gradient contribution"
        );

        // Record interaction based on quality
        let interaction_type = if pogq_metrics.quality_score >= 0.7 {
            InteractionType::HighQualityContribution
        } else if pogq_metrics.quality_score >= 0.5 {
            InteractionType::FLParticipation
        } else {
            InteractionType::LowQualityContribution
        };

        let interaction = Interaction::new(agent_id.clone(), interaction_type)
            .with_round(contribution.round)
            .with_context(format!("quality={:.2}", pogq_metrics.quality_score));

        self.tcdm_tracker.record_interaction(interaction)?;

        // Update agent state
        if !self.agent_states.contains_key(&agent_id) {
            self.agent_states.insert(agent_id.clone(), AgentState::new(agent_id.clone()));
        }

        {
            let state = self.agent_states.get_mut(&agent_id)
                .expect("agent_states entry just inserted above");
            state.last_pogq_score = pogq_metrics.quality_score;
            state.contributions_count += 1;
        }

        // Recompute trust (needs immutable borrow)
        let trust = self.compute_trust(&agent_id)?;

        // Update K-Vector from trust (needs mutable borrow)
        let state = self.agent_states.get_mut(&agent_id)
            .expect("agent_states entry inserted above");
        state.k_vector.k_r = trust.total;
        state.k_vector.k_p = pogq_metrics.quality_score;

        // Recalculate entropy
        state.entropy_score = EntropyScore::from_k_vector(&state.k_vector);

        // Store in trust store
        let record = TrustRecord::new(
            agent_id.clone(),
            state.k_vector.clone(),
            trust.clone(),
            TrustSource::PoGQOracle,
        );
        self.trust_store.upsert(record);

        info!(
            agent_id = %agent_id,
            trust = trust.total,
            "Updated agent trust"
        );

        Ok(trust)
    }

    /// Record a missed round for an agent
    pub fn record_missed_round(&mut self, agent_id: &str, round: u64) -> MatlResult<()> {
        let interaction = Interaction::new(agent_id.to_string(), InteractionType::MissedRound)
            .with_round(round);

        self.tcdm_tracker.record_interaction(interaction)?;

        warn!(agent_id = %agent_id, round, "Agent missed round");

        Ok(())
    }

    /// Record Byzantine behavior detection
    pub fn record_byzantine(&mut self, agent_id: &str, evidence: &str) -> MatlResult<()> {
        let interaction = Interaction::new(agent_id.to_string(), InteractionType::ByzantineBehavior)
            .with_context(evidence.to_string());

        self.tcdm_tracker.record_interaction(interaction)?;

        warn!(agent_id = %agent_id, evidence, "Byzantine behavior detected");

        // Heavily penalize in agent state
        if let Some(state) = self.agent_states.get_mut(agent_id) {
            state.k_vector.k_i *= 0.5; // Halve integrity
            state.k_vector.k_r *= 0.5; // Halve reputation
            state.byzantine_flags += 1;
        }

        Ok(())
    }

    /// Get agent trust
    pub fn get_trust(&self, agent_id: &str) -> MatlResult<TrustScore> {
        self.compute_trust(agent_id)
    }

    /// Get K-Vector for an agent
    pub fn get_k_vector(&self, agent_id: &str) -> Option<&KVector> {
        self.agent_states.get(agent_id).map(|s| &s.k_vector)
    }

    /// Get all agents above trust threshold
    pub fn trusted_agents(&self, threshold: f32) -> Vec<(&str, TrustScore)> {
        self.agent_states
            .keys()
            .filter_map(|id| {
                self.compute_trust(id)
                    .ok()
                    .filter(|t| t.total >= threshold)
                    .map(|t| (id.as_str(), t))
            })
            .collect()
    }

    /// Apply global decay
    pub fn apply_decay(&mut self, current_time: i64) {
        self.tcdm_tracker.apply_global_decay(current_time);
    }

    /// Get bridge statistics
    pub fn stats(&self) -> MatlBridgeStats {
        let tcdm_stats = self.tcdm_tracker.stats();
        let store_stats = self.trust_store.stats();

        let avg_pogq: f32 = if !self.agent_states.is_empty() {
            self.agent_states.values().map(|s| s.last_pogq_score).sum::<f32>()
                / self.agent_states.len() as f32
        } else {
            0.0
        };

        let avg_entropy: f32 = if !self.agent_states.is_empty() {
            self.agent_states.values().map(|s| s.entropy_score.normalized).sum::<f32>()
                / self.agent_states.len() as f32
        } else {
            0.0
        };

        MatlBridgeStats {
            agent_count: self.agent_states.len(),
            avg_pogq,
            avg_tcdm: tcdm_stats.avg_trust,
            avg_entropy,
            total_contributions: self.agent_states.values().map(|s| s.contributions_count).sum(),
            byzantine_flags: self.agent_states.values().map(|s| s.byzantine_flags).sum(),
            pending_sync: store_stats.pending_count,
        }
    }

    /// Get trust store for syncing
    pub fn trust_store(&self) -> &TrustStore {
        &self.trust_store
    }

    /// Get mutable trust store
    pub fn trust_store_mut(&mut self) -> &mut TrustStore {
        &mut self.trust_store
    }
}

impl Default for MatlBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Agent ID
    pub agent_id: String,
    /// K-Vector representation
    pub k_vector: KVector,
    /// Last PoGQ score
    pub last_pogq_score: f32,
    /// Entropy score
    pub entropy_score: EntropyScore,
    /// Number of contributions
    pub contributions_count: u64,
    /// Number of Byzantine flags
    pub byzantine_flags: u32,
}

impl AgentState {
    /// Create new agent state
    pub fn new(agent_id: String) -> Self {
        let k_vector = KVector::neutral();
        Self {
            agent_id,
            entropy_score: EntropyScore::from_k_vector(&k_vector),
            k_vector,
            last_pogq_score: 0.5,
            contributions_count: 0,
            byzantine_flags: 0,
        }
    }
}

/// MATL Bridge statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlBridgeStats {
    pub agent_count: usize,
    pub avg_pogq: f32,
    pub avg_tcdm: f32,
    pub avg_entropy: f32,
    pub total_contributions: u64,
    pub byzantine_flags: u32,
    pub pending_sync: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_contribution(agent_id: &str, round: u64) -> GradientContribution {
        GradientContribution::new(agent_id.to_string(), round, "hash123".to_string())
            .with_stats(5.0, 1000, 0.0, 1.0, 0.1, 3.1)
            .with_dp(5.0, 10.0)
    }

    #[test]
    fn test_process_gradient() {
        let mut bridge = MatlBridge::new();
        let contribution = create_valid_contribution("agent-1", 1);

        let result = bridge.process_gradient(contribution);
        assert!(result.is_ok());

        let trust = result.unwrap();
        assert!(trust.total > 0.0);
    }

    #[test]
    fn test_trusted_agents() {
        let mut bridge = MatlBridge::new();

        // Add some agents
        for i in 0..5 {
            let contribution = create_valid_contribution(&format!("agent-{}", i), 1);
            bridge.process_gradient(contribution).unwrap();
        }

        let trusted = bridge.trusted_agents(0.3);
        assert!(!trusted.is_empty());
    }

    #[test]
    fn test_byzantine_penalty() {
        let mut bridge = MatlBridge::new();

        // Add agent with good contribution
        let contribution = create_valid_contribution("agent-1", 1);
        let initial_trust = bridge.process_gradient(contribution).unwrap();

        // Record Byzantine behavior
        bridge.record_byzantine("agent-1", "double voting").unwrap();

        // Trust should decrease
        let state = bridge.agent_states.get("agent-1").unwrap();
        assert!(state.k_vector.k_r < initial_trust.total);
    }

    // ==================== CRITICAL TESTS ====================
    // Added per CODE_REVIEW_FINDINGS_2026-01-19.md Section 4.2

    #[test]
    fn test_matl_weights_sum_to_one() {
        // CRITICAL: Verify MATL formula weights sum to 1.0
        // T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
        let sum = POGQ_WEIGHT + TCDM_WEIGHT + ENTROPY_WEIGHT;
        assert!(
            (sum - 1.0).abs() < 0.0001,
            "MATL weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_trust_formula_with_known_values() {
        // CRITICAL: Verify trust formula computes correctly
        // T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
        let config = MatlBridgeConfig::default();

        // Test case: PoGQ=0.8, TCDM=0.6, Entropy=0.5
        let expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.5;
        let actual = config.pogq_weight * 0.8 + config.tcdm_weight * 0.6 + config.entropy_weight * 0.5;

        assert!(
            (expected - actual).abs() < 0.0001,
            "Trust formula mismatch: expected {}, got {}",
            expected,
            actual
        );
        assert!((expected - 0.65).abs() < 0.0001, "Expected 0.65, got {}", expected);
    }

    #[test]
    fn test_trust_boundary_all_zeros() {
        // CRITICAL: Boundary condition - all components at minimum
        let config = MatlBridgeConfig::default();
        let trust = config.pogq_weight * 0.0 + config.tcdm_weight * 0.0 + config.entropy_weight * 0.0;

        assert!(
            trust.abs() < 0.0001,
            "All-zero components should yield zero trust, got {}",
            trust
        );
    }

    #[test]
    fn test_trust_boundary_all_ones() {
        // CRITICAL: Boundary condition - all components at maximum
        let config = MatlBridgeConfig::default();
        let trust = config.pogq_weight * 1.0 + config.tcdm_weight * 1.0 + config.entropy_weight * 1.0;

        assert!(
            (trust - 1.0).abs() < 0.0001,
            "All-one components should yield 1.0 trust, got {}",
            trust
        );
    }

    #[test]
    fn test_trust_pogq_dominance() {
        // CRITICAL: PoGQ has highest weight (0.4), verify it dominates
        let config = MatlBridgeConfig::default();

        // High PoGQ, zero others
        let pogq_only = config.pogq_weight * 1.0 + config.tcdm_weight * 0.0 + config.entropy_weight * 0.0;

        // Zero PoGQ, high others
        let others_only = config.pogq_weight * 0.0 + config.tcdm_weight * 1.0 + config.entropy_weight * 1.0;

        // PoGQ=1.0 should give 0.4, others=1.0 should give 0.6
        assert!((pogq_only - 0.4).abs() < 0.0001, "PoGQ-only should be 0.4, got {}", pogq_only);
        assert!((others_only - 0.6).abs() < 0.0001, "Others-only should be 0.6, got {}", others_only);
    }

    #[test]
    fn test_custom_weight_configuration() {
        // Test that custom weights are respected
        let config = MatlBridgeConfig {
            pogq_weight: 0.5,
            tcdm_weight: 0.3,
            entropy_weight: 0.2,
            ..Default::default()
        };

        let bridge = MatlBridge::with_config(config.clone());

        assert!((bridge.config.pogq_weight - 0.5).abs() < 0.0001);
        assert!((bridge.config.tcdm_weight - 0.3).abs() < 0.0001);
        assert!((bridge.config.entropy_weight - 0.2).abs() < 0.0001);

        // Custom weights should also sum to 1.0 for a valid config
        let sum = config.pogq_weight + config.tcdm_weight + config.entropy_weight;
        assert!(
            (sum - 1.0).abs() < 0.0001,
            "Custom weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_agent_not_found_error() {
        // CRITICAL: Proper error handling for unknown agents
        let bridge = MatlBridge::new();

        let result = bridge.compute_trust("nonexistent-agent");
        assert!(result.is_err(), "Should return error for unknown agent");

        match result {
            Err(MatlError::AgentNotFound { agent_id }) => {
                assert_eq!(agent_id, "nonexistent-agent");
            }
            _ => panic!("Expected AgentNotFound error"),
        }
    }

    #[test]
    fn test_trust_consistency_across_rounds() {
        // CRITICAL: Trust should be consistent and monotonic for good behavior
        let mut bridge = MatlBridge::new();

        // Initial contribution
        let contribution1 = create_valid_contribution("agent-1", 1);
        let trust1 = bridge.process_gradient(contribution1).unwrap();

        // Second good contribution should not decrease trust significantly
        let contribution2 = create_valid_contribution("agent-1", 2);
        let trust2 = bridge.process_gradient(contribution2).unwrap();

        // Third good contribution
        let contribution3 = create_valid_contribution("agent-1", 3);
        let trust3 = bridge.process_gradient(contribution3).unwrap();

        // Trust should generally be maintained or increase with good contributions
        // Allow for small fluctuations due to entropy changes
        assert!(
            trust3.total >= trust1.total * 0.9,
            "Trust should not dramatically decrease with good contributions: {} vs {}",
            trust3.total,
            trust1.total
        );
        assert!(
            trust2.total > 0.0 && trust3.total > 0.0,
            "Trust should remain positive for good contributors"
        );
    }

    #[test]
    fn test_multiple_byzantine_penalties_compound() {
        // CRITICAL: Multiple Byzantine events should compound penalties
        let mut bridge = MatlBridge::new();

        // Add agent with good contribution
        let contribution = create_valid_contribution("agent-1", 1);
        bridge.process_gradient(contribution).unwrap();

        let state_before = bridge.agent_states.get("agent-1").unwrap().clone();

        // First Byzantine event
        bridge.record_byzantine("agent-1", "double voting").unwrap();
        let state_after_1 = bridge.agent_states.get("agent-1").unwrap().clone();

        // Second Byzantine event
        bridge.record_byzantine("agent-1", "gradient poisoning").unwrap();
        let state_after_2 = bridge.agent_states.get("agent-1").unwrap();

        // Each Byzantine event halves k_r, so after two events: k_r * 0.5 * 0.5 = k_r * 0.25
        assert!(
            state_after_2.k_vector.k_r < state_after_1.k_vector.k_r,
            "Second Byzantine penalty should further reduce trust"
        );
        assert!(
            state_after_2.byzantine_flags == 2,
            "Should have 2 Byzantine flags, got {}",
            state_after_2.byzantine_flags
        );
        assert!(
            state_after_2.k_vector.k_r < state_before.k_vector.k_r * 0.5,
            "Trust should be less than half after two Byzantine events"
        );
    }

    #[test]
    fn test_stats_accuracy() {
        // CRITICAL: Bridge statistics should be accurate
        let mut bridge = MatlBridge::new();

        // Add multiple agents
        for i in 0..3 {
            let contribution = create_valid_contribution(&format!("agent-{}", i), 1);
            bridge.process_gradient(contribution).unwrap();
        }

        // Add one more contribution to first agent
        let contribution = create_valid_contribution("agent-0", 2);
        bridge.process_gradient(contribution).unwrap();

        let stats = bridge.stats();

        assert_eq!(stats.agent_count, 3, "Should have 3 agents");
        assert_eq!(stats.total_contributions, 4, "Should have 4 total contributions");
        assert!(stats.avg_pogq > 0.0, "Average PoGQ should be positive");
        assert_eq!(stats.byzantine_flags, 0, "Should have no Byzantine flags");
    }
}
