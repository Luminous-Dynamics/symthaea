// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Coherence Integration for Agent Coherence
//!
//! Integration of coherence metrics for multi-agent systems.
//!
//! ## IMPORTANT: Coherence Usage Guidelines
//!
//! Coherence should be used sparingly:
//!
//! **Meaningful uses (keep):**
//! - Gating high-stakes ZK operations (see `coherence_bridge.rs`)
//! - One-time coherence checks before critical decisions
//!
//! **Over-engineered uses (removed):**
//! - `EmergentBehaviorDetector` - use simple clustering instead
//! - `PhiEvolutionTracker` - use variance tracking instead
//! - `collective coherence` in dashboards - informational only, don't act on it
//!
//! ## Features
//!
//! - **Coherence-Gated Actions**: Gate high-stakes actions on coherence thresholds
//! - **Collective Coherence**: Measure population-level coherence (informational)

use super::InstrumentalActor;
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Collective Coherence Measurement
// ============================================================================

/// Collective coherence measurement for agent populations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CollectiveCoherenceResult {
    /// Population-level coherence (integration of all agents)
    pub population_coherence: f64,
    /// Average individual coherence
    pub average_individual_coherence: f64,
    /// Variance in individual coherence values
    pub coherence_variance: f64,
    /// Emergent integration (population coherence - sum of individual coherence)
    pub emergent_integration: f64,
    /// Coherence level classification
    pub coherence_level: CollectiveCoherenceLevel,
    /// Number of agents analyzed
    pub agent_count: usize,
    /// Agents grouped by coherence
    pub coherence_distribution: [usize; 5], // Incoherent, Low, Moderate, High, Very High
}

/// Classification of collective coherence
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum CollectiveCoherenceLevel {
    /// Fragmented - agents operating independently
    Fragmented,
    /// Weakly coordinated
    WeaklyCoordinated,
    /// Moderately synchronized
    ModeratelySynchronized,
    /// Highly integrated
    HighlyIntegrated,
    /// Emergent collective behavior
    EmergentCollective,
}

impl CollectiveCoherenceLevel {
    /// Get threshold for this level
    pub fn threshold(&self) -> f64 {
        match self {
            Self::Fragmented => 0.0,
            Self::WeaklyCoordinated => 0.3,
            Self::ModeratelySynchronized => 0.5,
            Self::HighlyIntegrated => 0.7,
            Self::EmergentCollective => 0.85,
        }
    }

    /// Classify from coherence value
    pub fn from_phi(phi: f64) -> Self {
        if phi >= 0.85 {
            Self::EmergentCollective
        } else if phi >= 0.7 {
            Self::HighlyIntegrated
        } else if phi >= 0.5 {
            Self::ModeratelySynchronized
        } else if phi >= 0.3 {
            Self::WeaklyCoordinated
        } else {
            Self::Fragmented
        }
    }
}

/// Measure collective coherence for a population of agents
pub fn measure_collective_coherence(agents: &[&InstrumentalActor]) -> CollectiveCoherenceResult {
    if agents.is_empty() {
        return CollectiveCoherenceResult {
            population_coherence: 0.0,
            average_individual_coherence: 0.0,
            coherence_variance: 0.0,
            emergent_integration: 0.0,
            coherence_level: CollectiveCoherenceLevel::Fragmented,
            agent_count: 0,
            coherence_distribution: [0; 5],
        };
    }

    // Compute individual coherence values
    let individual_coherences: Vec<f64> =
        agents.iter().map(|a| compute_agent_coherence(a)).collect();

    let sum_individual: f64 = individual_coherences.iter().sum();
    let average_individual_coherence = sum_individual / agents.len() as f64;

    // Compute variance
    let coherence_variance = individual_coherences
        .iter()
        .map(|p| (p - average_individual_coherence).powi(2))
        .sum::<f64>()
        / agents.len() as f64;

    // Compute population-level coherence by creating a combined state
    let population_coherence = compute_population_coherence(agents);

    // Emergent integration: how much more coherent is the collective vs individuals
    let emergent_integration = population_coherence - average_individual_coherence;

    // Classify agents by coherence
    let mut coherence_distribution = [0usize; 5];
    for coherence in &individual_coherences {
        let level = individual_coherence_level(*coherence);
        coherence_distribution[level] += 1;
    }

    CollectiveCoherenceResult {
        population_coherence,
        average_individual_coherence,
        coherence_variance,
        emergent_integration,
        coherence_level: CollectiveCoherenceLevel::from_phi(population_coherence),
        agent_count: agents.len(),
        coherence_distribution,
    }
}

fn individual_coherence_level(coherence: f64) -> usize {
    if coherence >= 0.8 {
        4
    } else if coherence >= 0.6 {
        3
    } else if coherence >= 0.4 {
        2
    } else if coherence >= 0.2 {
        1
    } else {
        0
    }
}

fn compute_agent_coherence(agent: &InstrumentalActor) -> f64 {
    // Use K-Vector as proxy for internal state coherence
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

    // Compute coherence as inverse variance (more uniform = more coherent)
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

    // Transform to 0-1 range (low variance = high coherence)
    let std_dev = variance.sqrt();
    (1.0 - std_dev.min(1.0)).max(0.0)
}

fn compute_population_coherence(agents: &[&InstrumentalActor]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }

    // Aggregate K-Vectors
    let mut aggregated = [0.0f64; 8];
    for agent in agents {
        let kv = &agent.k_vector;
        aggregated[0] += kv.k_r as f64;
        aggregated[1] += kv.k_a as f64;
        aggregated[2] += kv.k_i as f64;
        aggregated[3] += kv.k_p as f64;
        aggregated[4] += kv.k_m as f64;
        aggregated[5] += kv.k_s as f64;
        aggregated[6] += kv.k_h as f64;
        aggregated[7] += kv.k_topo as f64;
    }

    let n = agents.len() as f64;
    for v in &mut aggregated {
        *v /= n;
    }

    // Compute coherence of aggregated state
    let mean: f64 = aggregated.iter().sum::<f64>() / 8.0;
    let variance: f64 = aggregated.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 8.0;

    // Also measure inter-agent alignment
    let alignment = compute_inter_agent_alignment(agents);

    // Combined metric: coherence of average + alignment between agents
    let coherence = 1.0 - variance.sqrt().min(1.0);
    (coherence * 0.4 + alignment * 0.6).clamp(0.0, 1.0)
}

fn compute_inter_agent_alignment(agents: &[&InstrumentalActor]) -> f64 {
    if agents.len() < 2 {
        return 1.0; // Single agent is perfectly aligned with itself
    }

    // Compute average cosine similarity between agent K-Vectors
    let mut total_similarity = 0.0;
    let mut count = 0;

    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let sim = kvector_cosine_similarity(&agents[i].k_vector, &agents[j].k_vector);
            total_similarity += sim;
            count += 1;
        }
    }

    if count > 0 {
        total_similarity / count as f64
    } else {
        0.0
    }
}

fn kvector_cosine_similarity(a: &KVector, b: &KVector) -> f64 {
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

    let dot: f64 = av.iter().zip(&bv).map(|(x, y)| x * y).sum();
    let norm_a: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = bv.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ============================================================================
// Emergent Behavior Detection (types kept for signal processing)
// ============================================================================

/// Types of emergent behaviors that can be detected
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum EmergentBehaviorType {
    /// Agents spontaneously synchronizing behavior
    Synchronization,
    /// Formation of distinct agent clusters
    Clustering,
    /// Hierarchical structure emerging
    Hierarchy,
    /// Collective oscillation in behavior
    Oscillation,
    /// Cascade effects (one agent affects many)
    Cascade,
    /// Self-organized specialization
    Specialization,
    /// Coordinated attack pattern
    CoordinatedAttack,
}

/// Detected emergent behavior with evidence
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct EmergentBehavior {
    /// Type of behavior
    pub behavior_type: EmergentBehaviorType,
    /// Confidence in detection (0-1)
    pub confidence: f64,
    /// Agents involved
    pub involved_agents: Vec<String>,
    /// Evidence supporting detection
    pub evidence: Vec<String>,
    /// Timestamp of detection
    pub detected_at: u64,
    /// Is this behavior concerning?
    pub concerning: bool,
}

// ============================================================================
// Coherence-Gated Actions
// ============================================================================

/// Configuration for coherence-gated actions
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CoherenceGatingConfig {
    /// Minimum coherence for low-stakes actions
    pub low_stakes_threshold: f64,
    /// Minimum coherence for medium-stakes actions
    pub medium_stakes_threshold: f64,
    /// Minimum coherence for high-stakes actions
    pub high_stakes_threshold: f64,
    /// Minimum coherence for critical actions
    pub critical_threshold: f64,
}

impl Default for CoherenceGatingConfig {
    fn default() -> Self {
        Self {
            low_stakes_threshold: 0.2,
            medium_stakes_threshold: 0.4,
            high_stakes_threshold: 0.6,
            critical_threshold: 0.8,
        }
    }
}

/// Stakes level for an action
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum StakesLevel {
    /// Low-stakes (routine actions)
    Low,
    /// Medium-stakes (significant but reversible)
    Medium,
    /// High-stakes (significant impact)
    High,
    /// Critical (irreversible or major impact)
    Critical,
}

/// Result of coherence gating check
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CoherenceGatingResult {
    /// Whether the action is permitted
    pub permitted: bool,
    /// Current coherence value
    pub current_coherence: f64,
    /// Required coherence for this action
    pub required_coherence: f64,
    /// Deficit (if not permitted)
    pub deficit: f64,
    /// Recommendation
    pub recommendation: CoherenceGatingRecommendation,
}

/// Recommendation from coherence gating
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum CoherenceGatingRecommendation {
    /// Proceed with action
    Proceed,
    /// Proceed with caution
    ProceedWithCaution,
    /// Delay until coherence improves
    Delay,
    /// Escalate to human sponsor
    Escalate,
    /// Action denied
    Deny,
}

/// Check if an action should be permitted based on agent's coherence
pub fn check_coherence_gating(
    agent: &InstrumentalActor,
    stakes: StakesLevel,
    config: &CoherenceGatingConfig,
) -> CoherenceGatingResult {
    let current_coherence = compute_agent_coherence(agent);

    let required_coherence = match stakes {
        StakesLevel::Low => config.low_stakes_threshold,
        StakesLevel::Medium => config.medium_stakes_threshold,
        StakesLevel::High => config.high_stakes_threshold,
        StakesLevel::Critical => config.critical_threshold,
    };

    let permitted = current_coherence >= required_coherence;
    let deficit = (required_coherence - current_coherence).max(0.0);

    let recommendation = if permitted {
        if current_coherence >= required_coherence + 0.2 {
            CoherenceGatingRecommendation::Proceed
        } else {
            CoherenceGatingRecommendation::ProceedWithCaution
        }
    } else if deficit < 0.1 {
        CoherenceGatingRecommendation::Delay
    } else if stakes == StakesLevel::Critical {
        CoherenceGatingRecommendation::Escalate
    } else {
        CoherenceGatingRecommendation::Deny
    };

    CoherenceGatingResult {
        permitted,
        current_coherence,
        required_coherence,
        deficit,
        recommendation,
    }
}

// ============================================================================
// Agent Clustering by Coherence
// ============================================================================

/// Result of clustering agents by coherence similarity
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CoherenceClusterResult {
    /// Clusters of agent IDs
    pub clusters: Vec<CoherenceCluster>,
    /// Agents not in any cluster (singletons)
    pub singletons: Vec<String>,
    /// Total number of clusters
    pub cluster_count: usize,
    /// Average cluster size
    pub avg_cluster_size: f64,
    /// Clustering quality metric (silhouette-like)
    pub clustering_quality: f64,
}

/// A cluster of agents with similar coherence profiles
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CoherenceCluster {
    /// Cluster ID
    pub id: usize,
    /// Agent IDs in this cluster
    pub members: Vec<String>,
    /// Centroid (average K-Vector)
    pub centroid: [f64; 8],
    /// Average coherence of members
    pub avg_coherence: f64,
    /// Intra-cluster cohesion
    pub cohesion: f64,
}

/// Cluster agents based on K-Vector similarity
pub fn cluster_agents_by_coherence(
    agents: &HashMap<String, InstrumentalActor>,
    similarity_threshold: f64,
) -> CoherenceClusterResult {
    let agent_ids: Vec<_> = agents.keys().cloned().collect();
    let n = agent_ids.len();

    if n == 0 {
        return CoherenceClusterResult {
            clusters: vec![],
            singletons: vec![],
            cluster_count: 0,
            avg_cluster_size: 0.0,
            clustering_quality: 0.0,
        };
    }

    // Simple agglomerative clustering
    let mut cluster_assignments: Vec<Option<usize>> = vec![None; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for i in 0..n {
        if cluster_assignments[i].is_some() {
            continue;
        }

        let cluster_id = clusters.len();
        let mut cluster = vec![i];
        cluster_assignments[i] = Some(cluster_id);

        for j in (i + 1)..n {
            if cluster_assignments[j].is_some() {
                continue;
            }

            let agent_i = &agents[&agent_ids[i]];
            let agent_j = &agents[&agent_ids[j]];

            let sim = kvector_cosine_similarity(&agent_i.k_vector, &agent_j.k_vector);
            if sim >= similarity_threshold {
                cluster.push(j);
                cluster_assignments[j] = Some(cluster_id);
            }
        }

        clusters.push(cluster);
    }

    // Build result
    let mut coherence_clusters = Vec::new();
    let mut singletons = Vec::new();

    for (id, member_indices) in clusters.iter().enumerate() {
        if member_indices.len() == 1 {
            singletons.push(agent_ids[member_indices[0]].clone());
            continue;
        }

        let members: Vec<String> = member_indices
            .iter()
            .map(|&i| agent_ids[i].clone())
            .collect();

        // Compute centroid
        let mut centroid = [0.0f64; 8];
        let mut total_coherence = 0.0;

        for &idx in member_indices {
            let agent = &agents[&agent_ids[idx]];
            let kv = &agent.k_vector;
            centroid[0] += kv.k_r as f64;
            centroid[1] += kv.k_a as f64;
            centroid[2] += kv.k_i as f64;
            centroid[3] += kv.k_p as f64;
            centroid[4] += kv.k_m as f64;
            centroid[5] += kv.k_s as f64;
            centroid[6] += kv.k_h as f64;
            centroid[7] += kv.k_topo as f64;
            total_coherence += compute_agent_coherence(agent);
        }

        let n_members = member_indices.len() as f64;
        for v in &mut centroid {
            *v /= n_members;
        }
        let avg_coherence = total_coherence / n_members;

        // Compute cohesion (average intra-cluster similarity)
        let mut cohesion_sum = 0.0;
        let mut cohesion_count = 0;

        for i in 0..member_indices.len() {
            for j in (i + 1)..member_indices.len() {
                let agent_i = &agents[&agent_ids[member_indices[i]]];
                let agent_j = &agents[&agent_ids[member_indices[j]]];
                cohesion_sum += kvector_cosine_similarity(&agent_i.k_vector, &agent_j.k_vector);
                cohesion_count += 1;
            }
        }

        let cohesion = if cohesion_count > 0 {
            cohesion_sum / cohesion_count as f64
        } else {
            1.0
        };

        coherence_clusters.push(CoherenceCluster {
            id,
            members,
            centroid,
            avg_coherence,
            cohesion,
        });
    }

    let cluster_count = coherence_clusters.len();
    let avg_cluster_size = if cluster_count > 0 {
        coherence_clusters
            .iter()
            .map(|c| c.members.len())
            .sum::<usize>() as f64
            / cluster_count as f64
    } else {
        0.0
    };

    // Clustering quality: average cohesion
    let clustering_quality = if cluster_count > 0 {
        coherence_clusters.iter().map(|c| c.cohesion).sum::<f64>() / cluster_count as f64
    } else {
        0.0
    };

    CoherenceClusterResult {
        clusters: coherence_clusters,
        singletons,
        cluster_count,
        avg_cluster_size,
        clustering_quality,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats};

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
    fn test_collective_coherence_measurement() {
        // Create agents with similar K-Vectors (should have high collective coherence)
        let agent1 = create_test_agent(
            "a1",
            KVector::new(0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.6, 0.3, 0.7, 0.65),
        );
        let agent2 = create_test_agent(
            "a2",
            KVector::new(0.75, 0.65, 0.82, 0.72, 0.32, 0.42, 0.62, 0.32, 0.72, 0.7),
        );
        let agent3 = create_test_agent(
            "a3",
            KVector::new(0.68, 0.58, 0.78, 0.68, 0.28, 0.38, 0.58, 0.28, 0.68, 0.6),
        );

        let result = measure_collective_coherence(&[&agent1, &agent2, &agent3]);

        assert_eq!(result.agent_count, 3);
        assert!(
            result.population_coherence > 0.5,
            "Similar agents should have high collective coherence"
        );
        assert!(
            result.coherence_variance < 0.1,
            "Similar agents should have low variance"
        );
    }

    #[test]
    fn test_collective_coherence_diverse_agents() {
        // Create agents with diverse K-Vectors (should have lower collective coherence)
        let agent1 = create_test_agent(
            "a1",
            KVector::new(0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.3),
        );
        let agent2 = create_test_agent(
            "a2",
            KVector::new(0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.5, 0.3),
        );

        let result = measure_collective_coherence(&[&agent1, &agent2]);

        assert_eq!(result.agent_count, 2);
        // Diverse agents should have lower alignment
        assert!(
            result.population_coherence < 0.7,
            "Diverse agents should have lower collective coherence"
        );
    }

    #[test]
    fn test_coherence_gating() {
        // Create agent with moderate coherence
        let agent = create_test_agent(
            "a1",
            KVector::new(0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.55, 0.5),
        );
        let config = CoherenceGatingConfig::default();

        // Low stakes should be permitted
        let result = check_coherence_gating(&agent, StakesLevel::Low, &config);
        assert!(result.permitted);

        // Critical stakes might not be permitted
        let _result = check_coherence_gating(&agent, StakesLevel::Critical, &config);
        // Whether it's permitted depends on the computed coherence
    }

    #[test]
    fn test_agent_clustering() {
        let mut agents = HashMap::new();

        // Create two clusters
        // Cluster 1: High trust agents
        agents.insert(
            "a1".to_string(),
            create_test_agent(
                "a1",
                KVector::new(0.8, 0.7, 0.9, 0.8, 0.5, 0.6, 0.7, 0.5, 0.8, 0.75),
            ),
        );
        agents.insert(
            "a2".to_string(),
            create_test_agent(
                "a2",
                KVector::new(0.82, 0.72, 0.88, 0.78, 0.52, 0.58, 0.72, 0.52, 0.78, 0.75),
            ),
        );

        // Cluster 2: Low trust agents
        agents.insert(
            "a3".to_string(),
            create_test_agent(
                "a3",
                KVector::new(0.2, 0.3, 0.25, 0.3, 0.1, 0.15, 0.2, 0.1, 0.2, 0.2),
            ),
        );
        agents.insert(
            "a4".to_string(),
            create_test_agent(
                "a4",
                KVector::new(0.22, 0.28, 0.27, 0.32, 0.12, 0.17, 0.22, 0.12, 0.22, 0.2),
            ),
        );

        let result = cluster_agents_by_coherence(&agents, 0.9);

        assert!(
            result.cluster_count >= 1,
            "Should find at least one cluster"
        );
    }

    #[test]
    fn test_coherence_level_classification() {
        assert_eq!(
            CollectiveCoherenceLevel::from_phi(0.1),
            CollectiveCoherenceLevel::Fragmented
        );
        assert_eq!(
            CollectiveCoherenceLevel::from_phi(0.4),
            CollectiveCoherenceLevel::WeaklyCoordinated
        );
        assert_eq!(
            CollectiveCoherenceLevel::from_phi(0.6),
            CollectiveCoherenceLevel::ModeratelySynchronized
        );
        assert_eq!(
            CollectiveCoherenceLevel::from_phi(0.75),
            CollectiveCoherenceLevel::HighlyIntegrated
        );
        assert_eq!(
            CollectiveCoherenceLevel::from_phi(0.9),
            CollectiveCoherenceLevel::EmergentCollective
        );
    }
}
