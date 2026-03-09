//! Trust-Integrated Patterns: K-Vector Trust Weighting
//!
//! Component 12 integrates K-Vector trust scores into pattern evaluation.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::SymthaeaId;
use crate::KVector;

/// Unique identifier for agents in the trust system
pub type AgentId = u64;

// ==============================================================================
// COMPONENT 12: TRUST-INTEGRATED PATTERNS - K-Vector Trust Weighting
// ==============================================================================
//
// This component connects patterns to the K-Vector trust system, enabling:
//
// 1. Patterns weighted by learner's trust score
// 2. Trust-adjusted pattern rankings and recommendations
// 3. Pattern credibility that evolves with agent reputation
// 4. Trust decay when agents behave inconsistently
//
// Key insight: A pattern from a high-trust agent should carry more weight
// than the same pattern from an unknown agent. This aligns with MATL's
// reputation-weighted validation.

/// Configuration for trust-weighted pattern evaluation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrustWeightConfig {
    /// Minimum trust score for patterns to be considered (0.0-1.0)
    pub min_trust_threshold: f32,

    /// How much weight trust has in final score (0.0-1.0)
    /// 0.0 = ignore trust, 1.0 = trust dominates
    pub trust_weight: f32,

    /// Whether to apply trust decay over time
    pub enable_trust_decay: bool,

    /// Trust decay half-life in time units (e.g., 86400 for 1 day in seconds)
    pub trust_decay_halflife: u64,

    /// Bonus multiplier for patterns from high-integrity agents (k_i > 0.8)
    pub integrity_bonus: f32,

    /// Penalty multiplier for patterns from low-performance agents (k_p < 0.3)
    pub low_performance_penalty: f32,
}

impl Default for TrustWeightConfig {
    fn default() -> Self {
        Self {
            min_trust_threshold: 0.1,
            trust_weight: 0.3, // 30% trust, 70% pattern performance
            enable_trust_decay: true,
            trust_decay_halflife: 604800, // 1 week in seconds
            integrity_bonus: 1.2,
            low_performance_penalty: 0.7,
        }
    }
}

/// Trust context for a specific agent
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgentTrustContext {
    /// The agent's K-Vector trust representation
    pub k_vector: KVector,

    /// Timestamp when trust was last updated
    pub last_updated: u64,

    /// Number of patterns contributed by this agent
    pub patterns_contributed: u32,

    /// Success rate of patterns from this agent
    pub pattern_success_rate: f32,

    /// Whether this agent has been manually vouched for
    pub vouched: bool,

    /// Vouch strength (0.0-1.0) if vouched
    pub vouch_strength: f32,
}

impl AgentTrustContext {
    /// Create a new trust context for an agent
    pub fn new(k_vector: KVector, timestamp: u64) -> Self {
        Self {
            k_vector,
            last_updated: timestamp,
            patterns_contributed: 0,
            pattern_success_rate: 0.5, // Neutral
            vouched: false,
            vouch_strength: 0.0,
        }
    }

    /// Create a context for a new/unknown agent (neutral trust)
    pub fn new_agent(timestamp: u64) -> Self {
        Self::new(KVector::neutral(), timestamp)
    }

    /// Create a context for a highly trusted agent
    pub fn trusted_agent(timestamp: u64) -> Self {
        Self::new(
            KVector::from_array([0.9, 0.8, 0.9, 0.85, 0.7, 0.6, 0.9, 0.7]),
            timestamp,
        )
    }

    /// Get the overall trust score
    pub fn trust_score(&self) -> f32 {
        let base = self.k_vector.trust_score();

        // Apply vouch bonus if applicable
        if self.vouched {
            (base + self.vouch_strength * 0.2).min(1.0)
        } else {
            base
        }
    }

    /// Apply time-based decay to trust
    pub fn apply_decay(&mut self, current_time: u64, halflife: u64) {
        if current_time <= self.last_updated || halflife == 0 {
            return;
        }

        let elapsed = current_time - self.last_updated;
        let decay_factor = 0.5_f32.powf(elapsed as f32 / halflife as f32);

        // Decay toward neutral (0.5) rather than zero
        self.k_vector = self
            .k_vector
            .ema_merge(&KVector::neutral(), 1.0 - decay_factor);
    }

    /// Update trust based on pattern outcome
    pub fn update_from_pattern_outcome(&mut self, success: bool, timestamp: u64) {
        self.patterns_contributed += 1;

        // Update pattern success rate with exponential moving average
        let outcome = if success { 1.0 } else { 0.0 };
        self.pattern_success_rate = self.pattern_success_rate * 0.9 + outcome * 0.1;

        // Update K-Vector performance component
        self.k_vector.k_p = self.k_vector.k_p * 0.95 + self.pattern_success_rate * 0.05;

        // Update integrity if consistent
        if (self.pattern_success_rate - 0.5).abs() > 0.2 {
            // Agent has a consistent track record (good or bad)
            self.k_vector.k_h = (self.k_vector.k_h * 0.95 + 0.05).min(1.0);
        }

        self.last_updated = timestamp;
    }

    /// Vouch for this agent (increase trust)
    pub fn apply_vouch(&mut self, strength: f32) {
        self.vouched = true;
        self.vouch_strength = strength.clamp(0.0, 1.0);
    }

    /// Revoke vouch
    pub fn revoke_vouch(&mut self) {
        self.vouched = false;
        self.vouch_strength = 0.0;
    }
}

/// Result of a trust-weighted pattern evaluation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrustWeightedScore {
    /// The pattern's raw success rate
    pub raw_success_rate: f32,

    /// The learner's trust score
    pub learner_trust: f32,

    /// Trust-weighted final score
    pub weighted_score: f32,

    /// Confidence in this score (based on usage + trust)
    pub confidence: f32,

    /// Human-readable trust level
    pub trust_level: TrustLevel,

    /// Whether this pattern meets the minimum trust threshold
    pub meets_threshold: bool,
}

/// Categorical trust level for display
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TrustLevel {
    /// Trust < 0.2
    Untrusted,
    /// Trust 0.2-0.4
    Low,
    /// Trust 0.4-0.6
    Neutral,
    /// Trust 0.6-0.8
    High,
    /// Trust > 0.8
    VeryHigh,
}

impl TrustLevel {
    /// Get trust level from score
    pub fn from_score(score: f32) -> Self {
        if score < 0.2 {
            Self::Untrusted
        } else if score < 0.4 {
            Self::Low
        } else if score < 0.6 {
            Self::Neutral
        } else if score < 0.8 {
            Self::High
        } else {
            Self::VeryHigh
        }
    }

    /// Get a descriptive string
    pub fn description(&self) -> &'static str {
        match self {
            Self::Untrusted => "untrusted",
            Self::Low => "low trust",
            Self::Neutral => "neutral",
            Self::High => "trusted",
            Self::VeryHigh => "highly trusted",
        }
    }

    /// Check if this trust level meets or exceeds the threshold
    pub fn meets_threshold(&self, threshold: TrustLevel) -> bool {
        *self >= threshold
    }
}

/// Registry for tracking agent trust across the system
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgentTrustRegistry {
    /// Map of agent ID to trust context
    agents: HashMap<SymthaeaId, AgentTrustContext>,

    /// Configuration for trust weighting
    pub config: TrustWeightConfig,
}

impl AgentTrustRegistry {
    /// Create a new trust registry with default config
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            config: TrustWeightConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: TrustWeightConfig) -> Self {
        Self {
            agents: HashMap::new(),
            config,
        }
    }

    /// Register a new agent with initial K-Vector
    pub fn register_agent(&mut self, agent_id: SymthaeaId, k_vector: KVector, timestamp: u64) {
        self.agents
            .insert(agent_id, AgentTrustContext::new(k_vector, timestamp));
    }

    /// Register a new agent with neutral trust
    pub fn register_new_agent(&mut self, agent_id: SymthaeaId, timestamp: u64) {
        self.agents
            .insert(agent_id, AgentTrustContext::new_agent(timestamp));
    }

    /// Get trust context for an agent
    pub fn get(&self, agent_id: SymthaeaId) -> Option<&AgentTrustContext> {
        self.agents.get(&agent_id)
    }

    /// Get mutable trust context for an agent
    pub fn get_mut(&mut self, agent_id: SymthaeaId) -> Option<&mut AgentTrustContext> {
        self.agents.get_mut(&agent_id)
    }

    /// Get or create trust context for an agent
    pub fn get_or_create(
        &mut self,
        agent_id: SymthaeaId,
        timestamp: u64,
    ) -> &mut AgentTrustContext {
        self.agents
            .entry(agent_id)
            .or_insert_with(|| AgentTrustContext::new_agent(timestamp))
    }

    /// Get trust score for an agent (returns neutral 0.5 if unknown)
    pub fn trust_score(&self, agent_id: SymthaeaId) -> f32 {
        self.agents
            .get(&agent_id)
            .map(|ctx| ctx.trust_score())
            .unwrap_or(0.5)
    }

    /// Update agent trust from pattern outcome
    pub fn update_from_outcome(&mut self, agent_id: SymthaeaId, success: bool, timestamp: u64) {
        let ctx = self.get_or_create(agent_id, timestamp);
        ctx.update_from_pattern_outcome(success, timestamp);
    }

    /// Apply decay to all agents
    pub fn apply_decay_all(&mut self, current_time: u64) {
        if !self.config.enable_trust_decay {
            return;
        }

        for ctx in self.agents.values_mut() {
            ctx.apply_decay(current_time, self.config.trust_decay_halflife);
        }
    }

    /// Calculate trust-weighted score for a pattern
    pub fn calculate_weighted_score(
        &self,
        pattern_success_rate: f32,
        pattern_usage: u64,
        learner_id: SymthaeaId,
    ) -> TrustWeightedScore {
        let learner_trust = self.trust_score(learner_id);
        let learner_ctx = self.agents.get(&learner_id);

        // Apply integrity bonus or performance penalty
        let mut trust_multiplier = 1.0;
        if let Some(ctx) = learner_ctx {
            if ctx.k_vector.k_i > 0.8 {
                trust_multiplier = self.config.integrity_bonus;
            } else if ctx.k_vector.k_p < 0.3 {
                trust_multiplier = self.config.low_performance_penalty;
            }
        }

        let adjusted_trust = (learner_trust * trust_multiplier).clamp(0.0, 1.0);

        // Weighted combination of pattern performance and trust
        let weighted_score = pattern_success_rate * (1.0 - self.config.trust_weight)
            + adjusted_trust * self.config.trust_weight;

        // Confidence based on usage and trust
        let usage_confidence = (pattern_usage as f32 / 100.0).sqrt().min(1.0);
        let trust_confidence = adjusted_trust;
        let confidence = (usage_confidence + trust_confidence) / 2.0;

        TrustWeightedScore {
            raw_success_rate: pattern_success_rate,
            learner_trust: adjusted_trust,
            weighted_score,
            confidence,
            trust_level: TrustLevel::from_score(adjusted_trust),
            meets_threshold: adjusted_trust >= self.config.min_trust_threshold,
        }
    }

    /// Get statistics about the trust registry
    pub fn stats(&self) -> TrustRegistryStats {
        let total = self.agents.len();
        let mut sum_trust = 0.0;
        let mut high_trust = 0;
        let mut low_trust = 0;

        for ctx in self.agents.values() {
            let score = ctx.trust_score();
            sum_trust += score;
            if score > 0.7 {
                high_trust += 1;
            } else if score < 0.3 {
                low_trust += 1;
            }
        }

        TrustRegistryStats {
            total_agents: total,
            average_trust: if total > 0 {
                sum_trust / total as f32
            } else {
                0.5
            },
            high_trust_agents: high_trust,
            low_trust_agents: low_trust,
            vouched_agents: self.agents.values().filter(|c| c.vouched).count(),
        }
    }

    /// Get number of tracked agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Alias for register_agent (convenience method)
    pub fn register(&mut self, agent_id: SymthaeaId, k_vector: KVector, timestamp: u64) {
        self.register_agent(agent_id, k_vector, timestamp);
    }

    /// Alias for register_new_agent (convenience method)
    pub fn register_new(&mut self, agent_id: SymthaeaId, timestamp: u64) {
        self.register_new_agent(agent_id, timestamp);
    }

    /// Vouch for an agent, increasing their trust
    pub fn vouch(&mut self, agent_id: SymthaeaId, strength: f32, timestamp: u64) {
        let ctx = self.get_or_create(agent_id, timestamp);
        ctx.vouched = true;
        ctx.vouch_strength = strength.clamp(0.0, 1.0);
        // Boost all K-Vector dimensions proportionally
        let boost = 1.0 + (strength * 0.2); // Up to 20% boost
        ctx.k_vector.k_r = (ctx.k_vector.k_r * boost).clamp(0.0, 1.0);
        ctx.k_vector.k_i = (ctx.k_vector.k_i * boost).clamp(0.0, 1.0);
        ctx.k_vector.k_h = (ctx.k_vector.k_h * boost).clamp(0.0, 1.0);
        ctx.last_updated = timestamp;
    }

    /// Revoke a vouch, reducing trust back
    pub fn revoke_vouch(&mut self, agent_id: SymthaeaId) {
        if let Some(ctx) = self.get_mut(agent_id) {
            if ctx.vouched {
                // Reduce boost
                let reduction = 1.0 / (1.0 + (ctx.vouch_strength * 0.2));
                ctx.k_vector.k_r = (ctx.k_vector.k_r * reduction).clamp(0.0, 1.0);
                ctx.k_vector.k_i = (ctx.k_vector.k_i * reduction).clamp(0.0, 1.0);
                ctx.k_vector.k_h = (ctx.k_vector.k_h * reduction).clamp(0.0, 1.0);
                ctx.vouched = false;
                ctx.vouch_strength = 0.0;
            }
        }
    }

    /// Apply decay to all agents (convenience wrapper)
    pub fn apply_decay(&mut self, current_time: u64) {
        self.apply_decay_all(current_time);
    }
}

/// Statistics about the trust registry
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrustRegistryStats {
    /// Total number of tracked agents
    pub total_agents: usize,

    /// Average trust score across all agents
    pub average_trust: f32,

    /// Number of high-trust agents (> 0.7)
    pub high_trust_agents: usize,

    /// Number of low-trust agents (< 0.3)
    pub low_trust_agents: usize,

    /// Number of vouched agents
    pub vouched_agents: usize,
}
