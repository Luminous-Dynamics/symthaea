// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi-Weighted Collective Consensus
//!
//! Bridges collective Phi measurement with trust-weighted consensus to create
//! decisions that are both accurate (trust-weighted) AND harmonious (phi-weighted).
//!
//! ## Philosophy
//!
//! Traditional consensus weights votes by individual trustworthiness.
//! Phi-weighted consensus adds a second dimension: how much does each agent
//! contribute to collective coherence?
//!
//! An agent who is individually trustworthy BUT diverges from the collective
//! gets lower weight than an agent who harmonizes with the whole.
//!
//! ## Key Concepts
//!
//! - **Phi Contribution**: How much an agent adds to collective coherence
//! - **Population Phi Gate**: Minimum collective coherence to proceed
//! - **Coherence-Boosted Confidence**: Higher collective Phi = higher confidence
//! - **Harmonic Weight**: trust × phi_contribution × epistemic_factors

use super::coherence_integration::{
    measure_collective_coherence, CollectiveCoherenceLevel, CollectiveCoherenceResult,
};
use super::multi_agent::{AgentVote, ConsensusConfig, ConsensusResult};
use super::{AgentStatus, InstrumentalActor};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for phi-weighted consensus
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiConsensusConfig {
    /// Base consensus configuration
    pub base_config: ConsensusConfig,

    /// Minimum population Phi to proceed with consensus
    /// If collective coherence is below this, consensus is deferred
    pub min_population_coherence: f64,

    /// Weight of phi contribution in final vote weight (0.0-1.0)
    /// Higher = harmony matters more, Lower = individual trust matters more
    pub phi_contribution_weight: f64,

    /// Threshold for flagging divergent agents
    /// Agents with phi contribution below this are flagged
    pub divergence_threshold: f64,

    /// Boost factor for confidence when population Phi is high
    pub coherence_confidence_boost: f64,

    /// Whether to require minimum coherence level
    pub require_coherence_level: Option<CollectiveCoherenceLevel>,
}

impl Default for PhiConsensusConfig {
    fn default() -> Self {
        Self {
            base_config: ConsensusConfig::default(),
            min_population_coherence: 0.3,
            phi_contribution_weight: 0.4,
            divergence_threshold: 0.3,
            coherence_confidence_boost: 0.2,
            require_coherence_level: None,
        }
    }
}

impl PhiConsensusConfig {
    /// Create config for high-stakes decisions (requires higher coherence)
    pub fn high_stakes() -> Self {
        Self {
            base_config: ConsensusConfig {
                min_trust_threshold: 0.5,
                max_dissent_threshold: 0.2,
                min_participants: 5,
                ..Default::default()
            },
            min_population_coherence: 0.6,
            phi_contribution_weight: 0.5,
            divergence_threshold: 0.4,
            coherence_confidence_boost: 0.3,
            require_coherence_level: Some(CollectiveCoherenceLevel::HighlyIntegrated),
        }
    }

    /// Create config for low-stakes decisions (more permissive)
    pub fn low_stakes() -> Self {
        Self {
            base_config: ConsensusConfig {
                min_trust_threshold: 0.2,
                max_dissent_threshold: 0.4,
                min_participants: 2,
                ..Default::default()
            },
            min_population_coherence: 0.2,
            phi_contribution_weight: 0.2,
            divergence_threshold: 0.2,
            coherence_confidence_boost: 0.1,
            require_coherence_level: None,
        }
    }
}

// ============================================================================
// Phi Contribution
// ============================================================================

/// Measures how much an agent contributes to collective coherence
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiContribution {
    /// Agent ID
    pub agent_id: String,

    /// Base phi contribution (0.0-1.0)
    /// How much does including this agent increase collective Phi?
    pub contribution: f64,

    /// Alignment with population centroid
    pub alignment: f64,

    /// Whether this agent diverges significantly from the collective
    pub is_divergent: bool,

    /// Individual Phi of this agent
    pub individual_phi: f64,
}

/// Compute phi contributions for all agents
pub fn compute_phi_contributions(
    agents: &HashMap<String, InstrumentalActor>,
    config: &PhiConsensusConfig,
) -> (CollectiveCoherenceResult, HashMap<String, PhiContribution>) {
    let active_agents: Vec<&InstrumentalActor> = agents
        .values()
        .filter(|a| a.status == AgentStatus::Active)
        .collect();

    if active_agents.is_empty() {
        return (
            CollectiveCoherenceResult {
                population_coherence: 0.0,
                average_individual_coherence: 0.0,
                coherence_variance: 0.0,
                emergent_integration: 0.0,
                coherence_level: CollectiveCoherenceLevel::Fragmented,
                agent_count: 0,
                coherence_distribution: [0; 5],
            },
            HashMap::new(),
        );
    }

    // Measure collective Phi with all agents
    let collective = measure_collective_coherence(&active_agents.to_vec());

    // Compute centroid K-Vector
    let centroid = compute_kvector_centroid(&active_agents);

    // Compute individual contributions
    let mut contributions = HashMap::new();

    for agent in &active_agents {
        let agent_id = agent.agent_id.as_str().to_string();

        // Compute individual phi
        let individual_phi = compute_individual_phi(agent);

        // Compute alignment with centroid
        let alignment = kvector_alignment(&agent.k_vector, &centroid);

        // Compute contribution: how much does including this agent help the collective?
        // Higher alignment + higher individual phi = higher contribution
        let contribution = (alignment * 0.6 + individual_phi * 0.4).clamp(0.0, 1.0);

        let is_divergent = contribution < config.divergence_threshold;

        contributions.insert(
            agent_id.clone(),
            PhiContribution {
                agent_id,
                contribution,
                alignment,
                is_divergent,
                individual_phi,
            },
        );
    }

    (collective, contributions)
}

/// Compute the centroid (average) K-Vector of all agents
fn compute_kvector_centroid(agents: &[&InstrumentalActor]) -> KVector {
    if agents.is_empty() {
        return KVector::default();
    }

    let n = agents.len() as f32;
    let mut sum = [0.0f32; 8];

    for agent in agents {
        let kv = &agent.k_vector;
        sum[0] += kv.k_r;
        sum[1] += kv.k_a;
        sum[2] += kv.k_i;
        sum[3] += kv.k_p;
        sum[4] += kv.k_m;
        sum[5] += kv.k_s;
        sum[6] += kv.k_h;
        sum[7] += kv.k_topo;
    }

    KVector::new(
        sum[0] / n,
        sum[1] / n,
        sum[2] / n,
        sum[3] / n,
        sum[4] / n,
        sum[5] / n,
        sum[6] / n,
        sum[7] / n,
        0.0, // k_v placeholder
        0.5, // k_coherence - default moderate coherence for consensus
    )
}

fn compute_individual_phi(agent: &InstrumentalActor) -> f64 {
    let kv = &agent.k_vector;
    let values = [
        kv.k_r as f64,
        kv.k_a as f64,
        kv.k_i as f64,
        kv.k_p as f64,
        kv.k_m as f64,
        kv.k_s as f64,
        kv.k_h as f64,
        kv.k_topo as f64,
    ];

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

    (1.0 - variance.sqrt().min(1.0)).max(0.0)
}

/// Compute alignment between two K-Vectors using normalized Euclidean distance
/// Returns value in [0, 1] where 1 = identical, 0 = maximally different
fn kvector_alignment(a: &KVector, b: &KVector) -> f64 {
    let av = [
        a.k_r as f64,
        a.k_a as f64,
        a.k_i as f64,
        a.k_p as f64,
        a.k_m as f64,
        a.k_s as f64,
        a.k_h as f64,
        a.k_topo as f64,
    ];
    let bv = [
        b.k_r as f64,
        b.k_a as f64,
        b.k_i as f64,
        b.k_p as f64,
        b.k_m as f64,
        b.k_s as f64,
        b.k_h as f64,
        b.k_topo as f64,
    ];

    // Compute Euclidean distance
    let squared_dist: f64 = av.iter().zip(&bv).map(|(x, y)| (x - y).powi(2)).sum();
    let distance = squared_dist.sqrt();

    // Maximum possible distance (corner to corner of 8D unit hypercube)
    let max_distance = (8.0_f64).sqrt(); // sqrt(8) ≈ 2.83

    // Convert to similarity: 1 = identical, 0 = maximally different
    (1.0 - distance / max_distance).max(0.0)
}

// ============================================================================
// Phi-Weighted Consensus Result
// ============================================================================

/// Extended consensus result with Phi metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiConsensusResult {
    /// Base consensus result
    pub consensus: ConsensusResult,

    /// Collective Phi metrics
    pub collective_phi: CollectiveCoherenceResult,

    /// Status of the phi-weighted consensus
    pub status: PhiConsensusStatus,

    /// Phi contributions by agent
    pub phi_contributions: HashMap<String, f64>,

    /// Agents flagged as divergent
    pub divergent_agents: Vec<String>,

    /// Harmonic confidence (boosted by collective coherence)
    pub harmonic_confidence: f64,

    /// Harmonic weights used (trust × phi_contribution)
    pub harmonic_weights: HashMap<String, f64>,
}

/// Status of phi-weighted consensus
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum PhiConsensusStatus {
    /// Consensus reached with sufficient collective coherence
    Reached,
    /// Consensus reached but with low coherence (proceed with caution)
    ReachedLowCoherence,
    /// Deferred due to insufficient collective coherence
    DeferredLowCoherence,
    /// Deferred due to required coherence level not met
    DeferredCoherenceLevelNotMet,
    /// Failed to reach consensus (dissent too high)
    FailedHighDissent,
    /// Not enough participants
    InsufficientParticipants,
}

// ============================================================================
// Main Consensus Function
// ============================================================================

/// Compute phi-weighted consensus from agent votes
///
/// This function combines traditional trust-weighted consensus with collective
/// coherence metrics to produce decisions that are both accurate and harmonious.
///
/// # Arguments
///
/// * `votes` - Votes from participating agents
/// * `agents` - All agents in the system (for context)
/// * `config` - Phi consensus configuration
///
/// # Returns
///
/// PhiConsensusResult with full metrics on consensus and coherence
pub fn compute_phi_weighted_consensus(
    votes: &[AgentVote],
    agents: &HashMap<String, InstrumentalActor>,
    config: &PhiConsensusConfig,
) -> PhiConsensusResult {
    // Step 1: Compute phi contributions for all agents
    let (collective_phi, phi_contributions) = compute_phi_contributions(agents, config);

    // Step 2: Check population phi gate
    if collective_phi.population_coherence < config.min_population_coherence {
        return PhiConsensusResult {
            consensus: empty_consensus_result(votes.len()),
            collective_phi,
            status: PhiConsensusStatus::DeferredLowCoherence,
            phi_contributions: phi_contributions
                .iter()
                .map(|(k, v)| (k.clone(), v.contribution))
                .collect(),
            divergent_agents: phi_contributions
                .iter()
                .filter(|(_, v)| v.is_divergent)
                .map(|(k, _)| k.clone())
                .collect(),
            harmonic_confidence: 0.0,
            harmonic_weights: HashMap::new(),
        };
    }

    // Step 3: Check required coherence level
    if let Some(required_level) = config.require_coherence_level {
        if collective_phi.population_coherence < required_level.threshold() {
            return PhiConsensusResult {
                consensus: empty_consensus_result(votes.len()),
                collective_phi,
                status: PhiConsensusStatus::DeferredCoherenceLevelNotMet,
                phi_contributions: phi_contributions
                    .iter()
                    .map(|(k, v)| (k.clone(), v.contribution))
                    .collect(),
                divergent_agents: phi_contributions
                    .iter()
                    .filter(|(_, v)| v.is_divergent)
                    .map(|(k, _)| k.clone())
                    .collect(),
                harmonic_confidence: 0.0,
                harmonic_weights: HashMap::new(),
            };
        }
    }

    // Step 4: Filter and weight votes using harmonic weights
    let valid_votes: Vec<_> = votes
        .iter()
        .filter(|v| {
            agents
                .get(&v.agent_id)
                .map(|a| a.k_vector.trust_score() >= config.base_config.min_trust_threshold)
                .unwrap_or(false)
        })
        .collect();

    if valid_votes.len() < config.base_config.min_participants {
        return PhiConsensusResult {
            consensus: empty_consensus_result(valid_votes.len()),
            collective_phi,
            status: PhiConsensusStatus::InsufficientParticipants,
            phi_contributions: phi_contributions
                .iter()
                .map(|(k, v)| (k.clone(), v.contribution))
                .collect(),
            divergent_agents: vec![],
            harmonic_confidence: 0.0,
            harmonic_weights: HashMap::new(),
        };
    }

    // Step 5: Compute harmonic weights
    let mut harmonic_weights: HashMap<String, f64> = HashMap::new();
    let mut weighted_values: Vec<(String, f64, f64)> = Vec::new(); // (agent_id, weight, value)
    let mut total_weight = 0.0;

    for vote in &valid_votes {
        if let Some(agent) = agents.get(&vote.agent_id) {
            let trust = agent.k_vector.trust_score() as f64;

            // Get phi contribution (default to 0.5 if not computed)
            let phi_contrib = phi_contributions
                .get(&vote.agent_id)
                .map(|p| p.contribution)
                .unwrap_or(0.5);

            // Epistemic and confidence bonuses
            let epistemic_bonus =
                (vote.epistemic_level as f64 / 4.0) * config.base_config.epistemic_weight;
            let confidence_bonus = vote.confidence * config.base_config.confidence_weight;

            // HARMONIC WEIGHT: trust × phi_contribution × (1 + bonuses)
            let harmonic_weight = trust
                * (1.0 - config.phi_contribution_weight
                    + config.phi_contribution_weight * phi_contrib)
                * (1.0 + epistemic_bonus + confidence_bonus);

            harmonic_weights.insert(vote.agent_id.clone(), harmonic_weight);
            weighted_values.push((vote.agent_id.clone(), harmonic_weight, vote.value));
            total_weight += harmonic_weight;
        }
    }

    if total_weight < 1e-10 {
        return PhiConsensusResult {
            consensus: empty_consensus_result(valid_votes.len()),
            collective_phi,
            status: PhiConsensusStatus::InsufficientParticipants,
            phi_contributions: phi_contributions
                .iter()
                .map(|(k, v)| (k.clone(), v.contribution))
                .collect(),
            divergent_agents: vec![],
            harmonic_confidence: 0.0,
            harmonic_weights,
        };
    }

    // Step 6: Compute weighted consensus value
    let consensus_value: f64 =
        weighted_values.iter().map(|(_, w, v)| w * v).sum::<f64>() / total_weight;

    // Step 7: Compute dissent
    let variance: f64 = weighted_values
        .iter()
        .map(|(_, w, v)| w * (v - consensus_value).powi(2))
        .sum::<f64>()
        / total_weight;
    let dissent = variance.sqrt().min(1.0);

    // Step 8: Compute contributions
    let contributions: HashMap<String, f64> = weighted_values
        .iter()
        .map(|(id, w, _)| (id.clone(), w / total_weight))
        .collect();

    // Step 9: Compute harmonic confidence (boosted by collective Phi)
    let agreement_factor = 1.0 - dissent;
    let trust_factor = (total_weight / valid_votes.len() as f64).min(1.0);
    let base_confidence = agreement_factor * trust_factor;

    // Boost confidence by collective coherence
    let coherence_boost = collective_phi.population_coherence * config.coherence_confidence_boost;
    let harmonic_confidence = (base_confidence * (1.0 + coherence_boost)).min(1.0);

    // Step 10: Determine status
    let consensus_reached = dissent <= config.base_config.max_dissent_threshold;
    let status = if !consensus_reached {
        PhiConsensusStatus::FailedHighDissent
    } else if collective_phi.coherence_level == CollectiveCoherenceLevel::Fragmented
        || collective_phi.coherence_level == CollectiveCoherenceLevel::WeaklyCoordinated
    {
        PhiConsensusStatus::ReachedLowCoherence
    } else {
        PhiConsensusStatus::Reached
    };

    // Collect divergent agents
    let divergent_agents: Vec<String> = phi_contributions
        .iter()
        .filter(|(id, v)| v.is_divergent && contributions.contains_key(*id))
        .map(|(k, _)| k.clone())
        .collect();

    PhiConsensusResult {
        consensus: ConsensusResult {
            consensus_value,
            confidence: base_confidence,
            participant_count: valid_votes.len(),
            total_trust_weight: total_weight,
            dissent,
            consensus_reached,
            contributions,
        },
        collective_phi,
        status,
        phi_contributions: phi_contributions
            .iter()
            .map(|(k, v)| (k.clone(), v.contribution))
            .collect(),
        divergent_agents,
        harmonic_confidence,
        harmonic_weights,
    }
}

fn empty_consensus_result(participant_count: usize) -> ConsensusResult {
    ConsensusResult {
        consensus_value: 0.5,
        confidence: 0.0,
        participant_count,
        total_trust_weight: 0.0,
        dissent: 1.0,
        consensus_reached: false,
        contributions: HashMap::new(),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a decision should proceed based on phi consensus result
pub fn should_proceed(result: &PhiConsensusResult) -> bool {
    matches!(
        result.status,
        PhiConsensusStatus::Reached | PhiConsensusStatus::ReachedLowCoherence
    )
}

/// Get recommendation based on phi consensus status
pub fn get_recommendation(result: &PhiConsensusResult) -> PhiConsensusRecommendation {
    match result.status {
        PhiConsensusStatus::Reached => PhiConsensusRecommendation::Proceed,
        PhiConsensusStatus::ReachedLowCoherence => PhiConsensusRecommendation::ProceedWithCaution {
            reason: "Low collective coherence - decision may not reflect unified perspective"
                .to_string(),
        },
        PhiConsensusStatus::DeferredLowCoherence => PhiConsensusRecommendation::Defer {
            reason: format!(
                "Population Phi ({:.2}) below threshold ({:.2})",
                result.collective_phi.population_coherence,
                0.3 // Would need config here
            ),
            suggested_wait: 300, // 5 minutes
        },
        PhiConsensusStatus::DeferredCoherenceLevelNotMet => PhiConsensusRecommendation::Defer {
            reason: format!(
                "Required coherence level not met (current: {:?})",
                result.collective_phi.coherence_level
            ),
            suggested_wait: 600, // 10 minutes
        },
        PhiConsensusStatus::FailedHighDissent => PhiConsensusRecommendation::Escalate {
            reason: format!(
                "High dissent ({:.2}) - agents disagree significantly",
                result.consensus.dissent
            ),
        },
        PhiConsensusStatus::InsufficientParticipants => PhiConsensusRecommendation::Wait {
            reason: "Not enough participants".to_string(),
            required: 3,
            current: result.consensus.participant_count,
        },
    }
}

/// Recommendation for how to handle phi consensus result
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum PhiConsensusRecommendation {
    /// Proceed with the decision
    Proceed,
    /// Proceed but with additional scrutiny
    ProceedWithCaution {
        /// Reason for caution
        reason: String,
    },
    /// Defer decision until coherence improves
    Defer {
        /// Reason for deferral
        reason: String,
        /// Suggested wait time in seconds
        suggested_wait: u64,
    },
    /// Wait for more participants
    Wait {
        /// Reason for waiting
        reason: String,
        /// Required number of participants
        required: usize,
        /// Current number of participants
        current: usize,
    },
    /// Escalate to human oversight
    Escalate {
        /// Reason for escalation
        reason: String,
    },
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{AgentClass, AgentConstraints, AgentId, EpistemicStats};
    use crate::matl::KVector;

    fn create_test_agent(id: &str, k_vector: KVector) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 1000,
            actions_this_hour: 0,
            k_vector,
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_phi_contributions() {
        let mut agents = HashMap::new();

        // Create similar agents (should have high alignment)
        agents.insert(
            "a1".to_string(),
            create_test_agent(
                "a1",
                KVector::new(0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.6, 0.3, 0.7, 0.65),
            ),
        );
        agents.insert(
            "a2".to_string(),
            create_test_agent(
                "a2",
                KVector::new(0.75, 0.65, 0.82, 0.72, 0.32, 0.42, 0.62, 0.32, 0.72, 0.7),
            ),
        );
        agents.insert(
            "a3".to_string(),
            create_test_agent(
                "a3",
                KVector::new(0.68, 0.58, 0.78, 0.68, 0.28, 0.38, 0.58, 0.28, 0.68, 0.6),
            ),
        );

        let config = PhiConsensusConfig::default();
        let (collective, contributions) = compute_phi_contributions(&agents, &config);

        // Should have high population phi (similar agents)
        assert!(collective.population_coherence > 0.5);
        assert_eq!(contributions.len(), 3);

        // All should have high contribution (similar to each other)
        for (_, contrib) in &contributions {
            assert!(contrib.contribution > 0.5);
            assert!(!contrib.is_divergent);
        }
    }

    #[test]
    fn test_divergent_agent_detection() {
        let mut agents = HashMap::new();

        // Three similar high-value agents
        agents.insert(
            "a1".to_string(),
            create_test_agent(
                "a1",
                KVector::new(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
            ),
        );
        agents.insert(
            "a2".to_string(),
            create_test_agent(
                "a2",
                KVector::new(0.82, 0.78, 0.81, 0.79, 0.81, 0.80, 0.79, 0.82, 0.80, 0.8),
            ),
        );
        agents.insert(
            "a3".to_string(),
            create_test_agent(
                "a3",
                KVector::new(0.79, 0.81, 0.80, 0.80, 0.79, 0.81, 0.80, 0.79, 0.80, 0.8),
            ),
        );

        // One agent with very low values (different magnitude = lower cosine similarity with centroid)
        agents.insert(
            "divergent".to_string(),
            create_test_agent(
                "divergent",
                KVector::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            ),
        );

        let config = PhiConsensusConfig::default();
        let (_, contributions) = compute_phi_contributions(&agents, &config);

        // The divergent agent should have lower alignment than the similar agents
        let divergent = contributions.get("divergent").unwrap();
        let a1 = contributions.get("a1").unwrap();

        // Divergent should have lower contribution than aligned agents
        assert!(
            divergent.contribution < a1.contribution,
            "Divergent contribution ({}) should be less than aligned ({})",
            divergent.contribution,
            a1.contribution
        );
    }

    #[test]
    fn test_phi_weighted_consensus_success() {
        let mut agents = HashMap::new();

        // Create aligned agents with high trust
        for i in 0..5 {
            let id = format!("agent-{}", i);
            agents.insert(
                id.clone(),
                create_test_agent(
                    &id,
                    KVector::new(0.7, 0.6, 0.8, 0.7, 0.5, 0.5, 0.6, 0.5, 0.7, 0.65),
                ),
            );
        }

        // Create votes that mostly agree
        let votes: Vec<AgentVote> = (0..5)
            .map(|i| AgentVote {
                agent_id: format!("agent-{}", i),
                value: 0.7 + (i as f64 * 0.02), // Slight variation
                confidence: 0.8,
                epistemic_level: 3,
                reasoning: None,
                timestamp: 1000,
            })
            .collect();

        let config = PhiConsensusConfig::default();
        let result = compute_phi_weighted_consensus(&votes, &agents, &config);

        assert!(matches!(result.status, PhiConsensusStatus::Reached));
        assert!(result.consensus.consensus_reached);
        assert!(result.harmonic_confidence > result.consensus.confidence);
        assert!(result.divergent_agents.is_empty());
    }

    #[test]
    fn test_phi_weighted_consensus_low_coherence() {
        let mut agents = HashMap::new();

        // Create agents with very different K-Vector patterns (low inter-agent alignment)
        // Each agent has different peaks in different dimensions
        agents.insert(
            "a0".to_string(),
            create_test_agent(
                "a0",
                KVector::new(0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2),
            ),
        );
        agents.insert(
            "a1".to_string(),
            create_test_agent(
                "a1",
                KVector::new(0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2),
            ),
        );
        agents.insert(
            "a2".to_string(),
            create_test_agent(
                "a2",
                KVector::new(0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2),
            ),
        );
        agents.insert(
            "a3".to_string(),
            create_test_agent(
                "a3",
                KVector::new(0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2),
            ),
        );
        agents.insert(
            "a4".to_string(),
            create_test_agent(
                "a4",
                KVector::new(0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.2, 0.2),
            ),
        );

        let votes: Vec<AgentVote> = (0..5)
            .map(|i| AgentVote {
                agent_id: format!("a{}", i),
                value: 0.5,
                confidence: 0.8,
                epistemic_level: 2,
                reasoning: None,
                timestamp: 1000,
            })
            .collect();

        // Use config with threshold we know will fail based on the divergent agents
        let config = PhiConsensusConfig {
            base_config: ConsensusConfig::default(),
            min_population_coherence: 0.95, // Very high threshold - almost impossible to reach
            phi_contribution_weight: 0.4,
            divergence_threshold: 0.3,
            coherence_confidence_boost: 0.2,
            require_coherence_level: None,
        };
        let result = compute_phi_weighted_consensus(&votes, &agents, &config);

        // Should be deferred due to population phi below 0.95
        assert!(
            matches!(result.status, PhiConsensusStatus::DeferredLowCoherence),
            "Expected DeferredLowCoherence, got {:?} with population_coherence={:.3}",
            result.status,
            result.collective_phi.population_coherence
        );
    }

    #[test]
    fn test_harmonic_weight_calculation() {
        let mut agents = HashMap::new();

        // High-trust, high-alignment agent
        agents.insert(
            "aligned".to_string(),
            create_test_agent(
                "aligned",
                KVector::new(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
            ),
        );

        // High-trust but divergent agent
        agents.insert(
            "divergent".to_string(),
            create_test_agent(
                "divergent",
                KVector::new(0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.5, 0.4),
            ),
        );

        // Medium agent for baseline
        agents.insert(
            "medium".to_string(),
            create_test_agent(
                "medium",
                KVector::new(0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75),
            ),
        );

        let votes: Vec<AgentVote> = vec!["aligned", "divergent", "medium"]
            .iter()
            .map(|id| AgentVote {
                agent_id: id.to_string(),
                value: 0.6,
                confidence: 0.8,
                epistemic_level: 3,
                reasoning: None,
                timestamp: 1000,
            })
            .collect();

        let config = PhiConsensusConfig::default();
        let result = compute_phi_weighted_consensus(&votes, &agents, &config);

        // Aligned agent should have higher harmonic weight than divergent
        let aligned_weight = result.harmonic_weights.get("aligned").unwrap();
        let divergent_weight = result.harmonic_weights.get("divergent").unwrap();

        assert!(aligned_weight > divergent_weight);
    }

    #[test]
    fn test_recommendation() {
        let mut agents = HashMap::new();
        for i in 0..3 {
            let id = format!("a{}", i);
            agents.insert(
                id.clone(),
                create_test_agent(
                    &id,
                    KVector::new(0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
                ),
            );
        }

        let votes: Vec<AgentVote> = (0..3)
            .map(|i| AgentVote {
                agent_id: format!("a{}", i),
                value: 0.5,
                confidence: 0.7,
                epistemic_level: 2,
                reasoning: None,
                timestamp: 1000,
            })
            .collect();

        let config = PhiConsensusConfig::default();
        let result = compute_phi_weighted_consensus(&votes, &agents, &config);

        let recommendation = get_recommendation(&result);
        assert!(matches!(
            recommendation,
            PhiConsensusRecommendation::Proceed
                | PhiConsensusRecommendation::ProceedWithCaution { .. }
        ));
    }
}
