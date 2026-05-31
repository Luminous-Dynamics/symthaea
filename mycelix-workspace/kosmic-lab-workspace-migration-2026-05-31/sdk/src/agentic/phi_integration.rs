// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Enhanced Phi Integration for Emergent Behavior Detection
//!
//! Deep integration of consciousness metrics (Phi) for detecting emergent
//! behaviors in multi-agent systems.
//!
//! ## Features
//!
//! - **Phi-Gated Actions**: Gate high-stakes actions on coherence thresholds
//! - **Collective Phi**: Measure population-level coherence and emergence
//! - **Emergent Behavior Detection**: Detect patterns that emerge from interactions
//! - **Agent Clustering**: Group agents by Phi similarity
//! - **Temporal Phi Analysis**: Track coherence evolution over time

use super::InstrumentalActor;
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Collective Phi Measurement
// ============================================================================

/// Collective Phi measurement for agent populations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CollectivePhiResult {
    /// Population-level Phi (integration of all agents)
    pub population_phi: f64,
    /// Average individual Phi
    pub average_individual_phi: f64,
    /// Variance in individual Phi values
    pub phi_variance: f64,
    /// Emergent integration (population Phi - sum of individual Phi)
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

    /// Classify from Phi value
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

/// Measure collective Phi for a population of agents
pub fn measure_collective_phi(agents: &[&InstrumentalActor]) -> CollectivePhiResult {
    if agents.is_empty() {
        return CollectivePhiResult {
            population_phi: 0.0,
            average_individual_phi: 0.0,
            phi_variance: 0.0,
            emergent_integration: 0.0,
            coherence_level: CollectiveCoherenceLevel::Fragmented,
            agent_count: 0,
            coherence_distribution: [0; 5],
        };
    }

    // Compute individual Phi values
    let individual_phis: Vec<f64> = agents.iter().map(|a| compute_agent_phi(a)).collect();

    let sum_individual: f64 = individual_phis.iter().sum();
    let average_individual_phi = sum_individual / agents.len() as f64;

    // Compute variance
    let phi_variance = individual_phis
        .iter()
        .map(|p| (p - average_individual_phi).powi(2))
        .sum::<f64>()
        / agents.len() as f64;

    // Compute population-level Phi by creating a combined state
    let population_phi = compute_population_phi(agents);

    // Emergent integration: how much more coherent is the collective vs individuals
    let emergent_integration = population_phi - average_individual_phi;

    // Classify agents by coherence
    let mut coherence_distribution = [0usize; 5];
    for phi in &individual_phis {
        let level = individual_coherence_level(*phi);
        coherence_distribution[level] += 1;
    }

    CollectivePhiResult {
        population_phi,
        average_individual_phi,
        phi_variance,
        emergent_integration,
        coherence_level: CollectiveCoherenceLevel::from_phi(population_phi),
        agent_count: agents.len(),
        coherence_distribution,
    }
}

fn individual_coherence_level(phi: f64) -> usize {
    if phi >= 0.8 {
        4
    } else if phi >= 0.6 {
        3
    } else if phi >= 0.4 {
        2
    } else if phi >= 0.2 {
        1
    } else {
        0
    }
}

fn compute_agent_phi(agent: &InstrumentalActor) -> f64 {
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

    // Transform to 0-1 range (low variance = high Phi)
    let std_dev = variance.sqrt();
    (1.0 - std_dev.min(1.0)).max(0.0)
}

fn compute_population_phi(agents: &[&InstrumentalActor]) -> f64 {
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

    // Compute Phi of aggregated state
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
// Emergent Behavior Detection
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

/// Detector for emergent behaviors
pub struct EmergentBehaviorDetector {
    /// History of collective Phi measurements
    phi_history: Vec<(u64, f64)>,
    /// Previous agent clusters
    prev_clusters: Vec<Vec<String>>,
    /// Synchronization detection threshold
    sync_threshold: f64,
    /// Cluster stability threshold
    cluster_threshold: f64,
}

impl EmergentBehaviorDetector {
    /// Create a new detector
    pub fn new() -> Self {
        Self {
            phi_history: Vec::new(),
            prev_clusters: Vec::new(),
            sync_threshold: 0.8,
            cluster_threshold: 0.7,
        }
    }

    /// Analyze population for emergent behaviors
    pub fn analyze(
        &mut self,
        agents: &HashMap<String, InstrumentalActor>,
        timestamp: u64,
    ) -> Vec<EmergentBehavior> {
        let mut behaviors = Vec::new();

        let agent_refs: Vec<_> = agents.values().collect();

        // Measure collective Phi
        let collective = measure_collective_phi(&agent_refs);
        self.phi_history
            .push((timestamp, collective.population_phi));

        // Detect synchronization (sudden increase in population Phi)
        if let Some(sync) = self.detect_synchronization(&collective, timestamp) {
            behaviors.push(sync);
        }

        // Detect clustering
        let clusters = self.compute_clusters(agents);
        if let Some(cluster_behavior) =
            self.detect_clustering_emergence(&clusters, agents, timestamp)
        {
            behaviors.push(cluster_behavior);
        }
        self.prev_clusters = clusters;

        // Detect oscillation in Phi
        if let Some(osc) = self.detect_oscillation(timestamp) {
            behaviors.push(osc);
        }

        // Detect coordinated attack patterns
        if let Some(attack) = self.detect_coordinated_attack(agents, timestamp) {
            behaviors.push(attack);
        }

        behaviors
    }

    fn detect_synchronization(
        &self,
        collective: &CollectivePhiResult,
        timestamp: u64,
    ) -> Option<EmergentBehavior> {
        if self.phi_history.len() < 5 {
            return None;
        }

        // Check if recent Phi values show sudden increase
        let recent: Vec<f64> = self
            .phi_history
            .iter()
            .rev()
            .take(5)
            .map(|(_, phi)| *phi)
            .collect();

        let old_avg = recent[2..].iter().sum::<f64>() / 3.0;
        let new_avg = recent[..2].iter().sum::<f64>() / 2.0;

        if new_avg > old_avg + 0.2 && collective.population_phi > self.sync_threshold {
            Some(EmergentBehavior {
                behavior_type: EmergentBehaviorType::Synchronization,
                confidence: (new_avg - old_avg).min(1.0),
                involved_agents: vec![], // All agents
                evidence: vec![
                    format!(
                        "Population Phi increased from {:.2} to {:.2}",
                        old_avg, new_avg
                    ),
                    format!("Coherence level: {:?}", collective.coherence_level),
                ],
                detected_at: timestamp,
                concerning: false, // Synchronization isn't inherently bad
            })
        } else {
            None
        }
    }

    fn compute_clusters(&self, agents: &HashMap<String, InstrumentalActor>) -> Vec<Vec<String>> {
        // Simple K-Vector based clustering using similarity threshold
        let agent_ids: Vec<_> = agents.keys().cloned().collect();
        let mut visited = vec![false; agent_ids.len()];
        let mut clusters = Vec::new();

        for i in 0..agent_ids.len() {
            if visited[i] {
                continue;
            }

            let mut cluster = vec![agent_ids[i].clone()];
            visited[i] = true;

            for j in (i + 1)..agent_ids.len() {
                if visited[j] {
                    continue;
                }

                let agent_i = &agents[&agent_ids[i]];
                let agent_j = &agents[&agent_ids[j]];

                let sim = kvector_cosine_similarity(&agent_i.k_vector, &agent_j.k_vector);
                if sim > self.cluster_threshold {
                    cluster.push(agent_ids[j].clone());
                    visited[j] = true;
                }
            }

            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }

        clusters
    }

    fn detect_clustering_emergence(
        &self,
        clusters: &[Vec<String>],
        agents: &HashMap<String, InstrumentalActor>,
        timestamp: u64,
    ) -> Option<EmergentBehavior> {
        // Detect new significant clusters
        if clusters.len() > self.prev_clusters.len() && !clusters.is_empty() {
            let largest = clusters.iter().max_by_key(|c| c.len())?;

            if largest.len() >= 3 {
                return Some(EmergentBehavior {
                    behavior_type: EmergentBehaviorType::Clustering,
                    confidence: largest.len() as f64 / agents.len().max(1) as f64,
                    involved_agents: largest.clone(),
                    evidence: vec![
                        format!("{} agents formed a cluster", largest.len()),
                        format!("Total clusters: {}", clusters.len()),
                    ],
                    detected_at: timestamp,
                    concerning: false,
                });
            }
        }

        None
    }

    fn detect_oscillation(&self, timestamp: u64) -> Option<EmergentBehavior> {
        if self.phi_history.len() < 10 {
            return None;
        }

        // Check for alternating high-low pattern
        let recent: Vec<f64> = self
            .phi_history
            .iter()
            .rev()
            .take(10)
            .map(|(_, phi)| *phi)
            .collect();

        let mut sign_changes = 0;
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;

        for i in 1..recent.len() {
            let prev_above = recent[i - 1] > mean;
            let curr_above = recent[i] > mean;
            if prev_above != curr_above {
                sign_changes += 1;
            }
        }

        // High sign changes indicate oscillation
        if sign_changes >= 6 {
            Some(EmergentBehavior {
                behavior_type: EmergentBehaviorType::Oscillation,
                confidence: sign_changes as f64 / 10.0,
                involved_agents: vec![],
                evidence: vec![
                    format!("Population Phi oscillating around {:.2}", mean),
                    format!("{} direction changes in last 10 measurements", sign_changes),
                ],
                detected_at: timestamp,
                concerning: true, // Oscillation might indicate instability
            })
        } else {
            None
        }
    }

    fn detect_coordinated_attack(
        &self,
        agents: &HashMap<String, InstrumentalActor>,
        timestamp: u64,
    ) -> Option<EmergentBehavior> {
        // Look for agents with very similar behavior patterns indicating coordination
        let suspicious: Vec<_> = agents
            .iter()
            .filter(|(_, a)| {
                // Agents with suspiciously similar success patterns and timing
                let log_len = a.behavior_log.len();
                if log_len < 5 {
                    return false;
                }

                // High success rate is suspicious
                let successes = a
                    .behavior_log
                    .iter()
                    .filter(|e| e.outcome == super::ActionOutcome::Success)
                    .count();
                successes as f64 / log_len as f64 > 0.95
            })
            .map(|(id, _)| id.clone())
            .collect();

        if suspicious.len() >= 3 {
            // Check if they're also clustered
            let suspicious_agents: Vec<_> =
                suspicious.iter().filter_map(|id| agents.get(id)).collect();

            if !suspicious_agents.is_empty() {
                let collective = measure_collective_phi(&suspicious_agents);

                if collective.population_phi > 0.9 {
                    return Some(EmergentBehavior {
                        behavior_type: EmergentBehaviorType::CoordinatedAttack,
                        confidence: collective.population_phi,
                        involved_agents: suspicious,
                        evidence: vec![
                            "Multiple agents with >95% success rate".to_string(),
                            format!(
                                "Collective Phi of suspicious agents: {:.2}",
                                collective.population_phi
                            ),
                        ],
                        detected_at: timestamp,
                        concerning: true,
                    });
                }
            }
        }

        None
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.phi_history.clear();
        self.prev_clusters.clear();
    }
}

impl Default for EmergentBehaviorDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Phi-Gated Actions
// ============================================================================

/// Configuration for Phi-gated actions
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiGatingConfig {
    /// Minimum Phi for low-stakes actions
    pub low_stakes_threshold: f64,
    /// Minimum Phi for medium-stakes actions
    pub medium_stakes_threshold: f64,
    /// Minimum Phi for high-stakes actions
    pub high_stakes_threshold: f64,
    /// Minimum Phi for critical actions
    pub critical_threshold: f64,
}

impl Default for PhiGatingConfig {
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

/// Result of Phi gating check
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiGatingResult {
    /// Whether the action is permitted
    pub permitted: bool,
    /// Current Phi value
    pub current_phi: f64,
    /// Required Phi for this action
    pub required_phi: f64,
    /// Deficit (if not permitted)
    pub deficit: f64,
    /// Recommendation
    pub recommendation: PhiGatingRecommendation,
}

/// Recommendation from Phi gating
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum PhiGatingRecommendation {
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

/// Check if an action should be permitted based on agent's Phi
pub fn check_phi_gating(
    agent: &InstrumentalActor,
    stakes: StakesLevel,
    config: &PhiGatingConfig,
) -> PhiGatingResult {
    let current_phi = compute_agent_phi(agent);

    let required_phi = match stakes {
        StakesLevel::Low => config.low_stakes_threshold,
        StakesLevel::Medium => config.medium_stakes_threshold,
        StakesLevel::High => config.high_stakes_threshold,
        StakesLevel::Critical => config.critical_threshold,
    };

    let permitted = current_phi >= required_phi;
    let deficit = (required_phi - current_phi).max(0.0);

    let recommendation = if permitted {
        if current_phi >= required_phi + 0.2 {
            PhiGatingRecommendation::Proceed
        } else {
            PhiGatingRecommendation::ProceedWithCaution
        }
    } else if deficit < 0.1 {
        PhiGatingRecommendation::Delay
    } else if stakes == StakesLevel::Critical {
        PhiGatingRecommendation::Escalate
    } else {
        PhiGatingRecommendation::Deny
    };

    PhiGatingResult {
        permitted,
        current_phi,
        required_phi,
        deficit,
        recommendation,
    }
}

// ============================================================================
// Agent Clustering by Phi
// ============================================================================

/// Result of clustering agents by Phi similarity
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiClusterResult {
    /// Clusters of agent IDs
    pub clusters: Vec<PhiCluster>,
    /// Agents not in any cluster (singletons)
    pub singletons: Vec<String>,
    /// Total number of clusters
    pub cluster_count: usize,
    /// Average cluster size
    pub avg_cluster_size: f64,
    /// Clustering quality metric (silhouette-like)
    pub clustering_quality: f64,
}

/// A cluster of agents with similar Phi profiles
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiCluster {
    /// Cluster ID
    pub id: usize,
    /// Agent IDs in this cluster
    pub members: Vec<String>,
    /// Centroid (average K-Vector)
    pub centroid: [f64; 8],
    /// Average Phi of members
    pub avg_phi: f64,
    /// Intra-cluster cohesion
    pub cohesion: f64,
}

/// Cluster agents based on K-Vector similarity
pub fn cluster_agents_by_phi(
    agents: &HashMap<String, InstrumentalActor>,
    similarity_threshold: f64,
) -> PhiClusterResult {
    let agent_ids: Vec<_> = agents.keys().cloned().collect();
    let n = agent_ids.len();

    if n == 0 {
        return PhiClusterResult {
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
    let mut phi_clusters = Vec::new();
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
        let mut total_phi = 0.0;

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
            total_phi += compute_agent_phi(agent);
        }

        let n_members = member_indices.len() as f64;
        for v in &mut centroid {
            *v /= n_members;
        }
        let avg_phi = total_phi / n_members;

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

        phi_clusters.push(PhiCluster {
            id,
            members,
            centroid,
            avg_phi,
            cohesion,
        });
    }

    let cluster_count = phi_clusters.len();
    let avg_cluster_size = if cluster_count > 0 {
        phi_clusters.iter().map(|c| c.members.len()).sum::<usize>() as f64 / cluster_count as f64
    } else {
        0.0
    };

    // Clustering quality: average cohesion
    let clustering_quality = if cluster_count > 0 {
        phi_clusters.iter().map(|c| c.cohesion).sum::<f64>() / cluster_count as f64
    } else {
        0.0
    };

    PhiClusterResult {
        clusters: phi_clusters,
        singletons,
        cluster_count,
        avg_cluster_size,
        clustering_quality,
    }
}

// ============================================================================
// Temporal Phi Analysis
// ============================================================================

/// Tracks Phi evolution over time for an agent
#[derive(Clone, Debug)]
pub struct PhiEvolutionTracker {
    /// History of (timestamp, phi) pairs
    history: Vec<(u64, f64)>,
    /// Maximum history size
    max_history: usize,
    /// Smoothing window size
    window_size: usize,
}

impl PhiEvolutionTracker {
    /// Create a new tracker
    pub fn new(max_history: usize, window_size: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
            window_size,
        }
    }

    /// Record a Phi observation
    pub fn record(&mut self, timestamp: u64, phi: f64) {
        self.history.push((timestamp, phi));
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get trend (positive = improving, negative = declining)
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let n = self.history.len();
        let recent = &self.history[n.saturating_sub(self.window_size)..];

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n_f = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().map(|(_, phi)| *phi).sum();
        let sum_xy: f64 = recent
            .iter()
            .enumerate()
            .map(|(i, (_, phi))| i as f64 * phi)
            .sum();
        let sum_x2: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n_f * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n_f * sum_xy - sum_x * sum_y) / denominator
    }

    /// Get volatility (standard deviation of recent values)
    pub fn volatility(&self) -> f64 {
        let n = self.history.len();
        let recent = &self.history[n.saturating_sub(self.window_size)..];

        if recent.len() < 2 {
            return 0.0;
        }

        let mean: f64 = recent.iter().map(|(_, phi)| *phi).sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent
            .iter()
            .map(|(_, phi)| (phi - mean).powi(2))
            .sum::<f64>()
            / recent.len() as f64;

        variance.sqrt()
    }

    /// Get current smoothed Phi
    pub fn smoothed_phi(&self) -> f64 {
        let n = self.history.len();
        let recent = &self.history[n.saturating_sub(self.window_size)..];

        if recent.is_empty() {
            return 0.0;
        }

        recent.iter().map(|(_, phi)| *phi).sum::<f64>() / recent.len() as f64
    }

    /// Check if Phi is stable
    pub fn is_stable(&self, threshold: f64) -> bool {
        self.volatility() < threshold
    }

    /// Get the evolution summary
    pub fn summary(&self) -> PhiEvolutionSummary {
        PhiEvolutionSummary {
            current: self.history.last().map(|(_, phi)| *phi).unwrap_or(0.0),
            smoothed: self.smoothed_phi(),
            trend: self.trend(),
            volatility: self.volatility(),
            observations: self.history.len(),
        }
    }
}

/// Summary of Phi evolution
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct PhiEvolutionSummary {
    /// Current (latest) Phi value
    pub current: f64,
    /// Smoothed Phi value
    pub smoothed: f64,
    /// Trend (slope of recent values)
    pub trend: f64,
    /// Volatility (standard deviation)
    pub volatility: f64,
    /// Number of observations
    pub observations: usize,
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
    fn test_collective_phi_measurement() {
        // Create agents with similar K-Vectors (should have high collective Phi)
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

        let result = measure_collective_phi(&[&agent1, &agent2, &agent3]);

        assert_eq!(result.agent_count, 3);
        assert!(
            result.population_phi > 0.5,
            "Similar agents should have high collective Phi"
        );
        assert!(
            result.phi_variance < 0.1,
            "Similar agents should have low variance"
        );
    }

    #[test]
    fn test_collective_phi_diverse_agents() {
        // Create agents with diverse K-Vectors (should have lower collective Phi)
        let agent1 = create_test_agent(
            "a1",
            KVector::new(0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.3),
        );
        let agent2 = create_test_agent(
            "a2",
            KVector::new(0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.5, 0.3),
        );

        let result = measure_collective_phi(&[&agent1, &agent2]);

        assert_eq!(result.agent_count, 2);
        // Diverse agents should have lower alignment
        assert!(
            result.population_phi < 0.7,
            "Diverse agents should have lower collective Phi"
        );
    }

    #[test]
    fn test_phi_gating() {
        // Create agent with moderate coherence
        let agent = create_test_agent(
            "a1",
            KVector::new(0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.55, 0.5),
        );
        let config = PhiGatingConfig::default();

        // Low stakes should be permitted
        let result = check_phi_gating(&agent, StakesLevel::Low, &config);
        assert!(result.permitted);

        // Critical stakes might not be permitted
        let result = check_phi_gating(&agent, StakesLevel::Critical, &config);
        // Whether it's permitted depends on the computed Phi
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

        let result = cluster_agents_by_phi(&agents, 0.9);

        assert!(
            result.cluster_count >= 1,
            "Should find at least one cluster"
        );
    }

    #[test]
    fn test_phi_evolution_tracker() {
        let mut tracker = PhiEvolutionTracker::new(100, 5);

        // Record improving Phi
        for i in 0..10 {
            tracker.record(i as u64 * 100, 0.3 + i as f64 * 0.05);
        }

        let summary = tracker.summary();
        assert!(
            summary.trend > 0.0,
            "Trend should be positive for improving Phi"
        );
        assert!(
            summary.current > summary.smoothed - 0.1,
            "Current should be near smoothed"
        );
    }

    #[test]
    fn test_emergent_behavior_detector() {
        let mut detector = EmergentBehaviorDetector::new();
        let mut agents = HashMap::new();

        // Create normal agents
        for i in 0..5 {
            agents.insert(
                format!("agent-{}", i),
                create_test_agent(&format!("agent-{}", i), KVector::new_participant()),
            );
        }

        // Run detection
        let behaviors = detector.analyze(&agents, 1000);
        // Initially should not detect emergent behaviors
        assert!(behaviors.is_empty() || behaviors.iter().all(|b| !b.concerning));
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
