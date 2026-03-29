// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Agent Trust System
//!
//! Enables trust-weighted consensus, agent collaboration, and cross-agent
//! calibration learning for collective intelligence.
//!
//! ## Key Features
//!
//! - **Trust-Weighted Consensus**: Aggregate multi-agent decisions by K-Vector trust
//! - **Cross-Agent Calibration**: Agents learn from well-calibrated peers
//! - **Collaboration Protocols**: Structured multi-agent task coordination
//! - **Collective Prediction**: Epistemic-weighted ensemble predictions
//! - **Reputation Propagation**: Trust flows through agent interaction graphs

use super::calibration_engine::{AgentCalibrationProfile, CalibrationQuality};
use super::InstrumentalActor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Trust-Weighted Consensus
// ============================================================================

/// A vote or prediction from an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentVote {
    /// Agent casting the vote
    pub agent_id: String,
    /// The vote value (0.0-1.0 for continuous, or discrete choice index)
    pub value: f64,
    /// Confidence in this vote (0.0-1.0)
    pub confidence: f64,
    /// Epistemic level of the vote (0-4 for E0-E4)
    pub epistemic_level: u8,
    /// Optional reasoning/justification
    pub reasoning: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of trust-weighted consensus
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ConsensusResult {
    /// Final consensus value
    pub consensus_value: f64,
    /// Confidence in consensus (based on agreement and trust)
    pub confidence: f64,
    /// Number of agents participating
    pub participant_count: usize,
    /// Total trust weight
    pub total_trust_weight: f64,
    /// Dissent measure (0.0 = full agreement, 1.0 = maximum disagreement)
    pub dissent: f64,
    /// Whether consensus was reached (dissent below threshold)
    pub consensus_reached: bool,
    /// Individual contributions by agent
    pub contributions: HashMap<String, f64>,
}

/// Configuration for consensus computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ConsensusConfig {
    /// Minimum trust score to participate
    pub min_trust_threshold: f32,
    /// Maximum dissent for consensus to be valid
    pub max_dissent_threshold: f64,
    /// Weight for epistemic level in trust calculation
    pub epistemic_weight: f64,
    /// Weight for confidence in trust calculation
    pub confidence_weight: f64,
    /// Minimum participants for valid consensus
    pub min_participants: usize,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            min_trust_threshold: 0.3,
            max_dissent_threshold: 0.3,
            epistemic_weight: 0.3,
            confidence_weight: 0.2,
            min_participants: 3,
        }
    }
}

/// Compute trust-weighted consensus from multiple agent votes
pub fn compute_consensus(
    votes: &[AgentVote],
    agents: &HashMap<String, InstrumentalActor>,
    config: &ConsensusConfig,
) -> ConsensusResult {
    // Filter votes from agents meeting trust threshold
    let valid_votes: Vec<_> = votes
        .iter()
        .filter(|v| {
            agents
                .get(&v.agent_id)
                .map(|a| a.k_vector.trust_score() >= config.min_trust_threshold)
                .unwrap_or(false)
        })
        .collect();

    if valid_votes.len() < config.min_participants {
        return ConsensusResult {
            consensus_value: 0.5,
            confidence: 0.0,
            participant_count: valid_votes.len(),
            total_trust_weight: 0.0,
            dissent: 1.0,
            consensus_reached: false,
            contributions: HashMap::new(),
        };
    }

    // Calculate weights for each vote
    let mut weights: Vec<(String, f64, f64)> = Vec::new(); // (agent_id, weight, value)
    let mut total_weight = 0.0;

    for vote in &valid_votes {
        if let Some(agent) = agents.get(&vote.agent_id) {
            let trust = agent.k_vector.trust_score() as f64;
            let epistemic_bonus = (vote.epistemic_level as f64 / 4.0) * config.epistemic_weight;
            let confidence_bonus = vote.confidence * config.confidence_weight;

            let weight = trust * (1.0 + epistemic_bonus + confidence_bonus);
            weights.push((vote.agent_id.clone(), weight, vote.value));
            total_weight += weight;
        }
    }

    if total_weight == 0.0 {
        return ConsensusResult {
            consensus_value: 0.5,
            confidence: 0.0,
            participant_count: valid_votes.len(),
            total_trust_weight: 0.0,
            dissent: 1.0,
            consensus_reached: false,
            contributions: HashMap::new(),
        };
    }

    // Compute weighted average
    let consensus_value: f64 = weights.iter().map(|(_, w, v)| w * v).sum::<f64>() / total_weight;

    // Compute dissent (weighted variance)
    let variance: f64 = weights
        .iter()
        .map(|(_, w, v)| w * (v - consensus_value).powi(2))
        .sum::<f64>()
        / total_weight;
    let dissent = variance.sqrt().min(1.0);

    // Compute contributions
    let contributions: HashMap<String, f64> = weights
        .iter()
        .map(|(id, w, _)| (id.clone(), w / total_weight))
        .collect();

    // Compute confidence based on agreement and total trust
    let agreement_factor = 1.0 - dissent;
    let trust_factor = (total_weight / valid_votes.len() as f64).min(1.0);
    let confidence = agreement_factor * trust_factor;

    ConsensusResult {
        consensus_value,
        confidence,
        participant_count: valid_votes.len(),
        total_trust_weight: total_weight,
        dissent,
        consensus_reached: dissent <= config.max_dissent_threshold,
        contributions,
    }
}

// ============================================================================
// Cross-Agent Calibration Learning
// ============================================================================

/// Calibration knowledge that can be shared between agents
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationKnowledge {
    /// Domain this calibration applies to
    pub domain: String,
    /// Recalibration curve (predicted -> adjusted)
    pub recalibration_points: Vec<(f64, f64)>,
    /// Quality of this calibration knowledge
    pub quality: CalibrationQuality,
    /// Number of predictions this is based on
    pub sample_size: u64,
    /// Source agent's trust score when shared
    pub source_trust: f32,
}

/// Cross-agent calibration learning system
#[derive(Clone, Debug, Default)]
pub struct CrossAgentCalibration {
    /// Calibration knowledge by domain
    pub domain_knowledge: HashMap<String, Vec<CalibrationKnowledge>>,
    /// Agent interaction history (who learned from whom)
    pub learning_history: HashMap<String, Vec<String>>,
}

impl CrossAgentCalibration {
    /// Create a new cross-agent calibration tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Share calibration knowledge from a well-calibrated agent
    pub fn share_calibration(
        &mut self,
        source_agent: &InstrumentalActor,
        profile: &AgentCalibrationProfile,
        domain: &str,
    ) -> Option<CalibrationKnowledge> {
        // Only share from well-calibrated agents
        let quality = profile.overall_curve.quality_rating();
        if !matches!(
            quality,
            CalibrationQuality::Excellent | CalibrationQuality::Good
        ) {
            return None;
        }

        // Extract recalibration points from the curve
        let recalibration_points: Vec<(f64, f64)> = profile
            .overall_curve
            .bins
            .iter()
            .filter(|bin| bin.count >= 10)
            .map(|bin| (bin.avg_predicted, bin.actual_frequency))
            .collect();

        if recalibration_points.len() < 3 {
            return None;
        }

        let knowledge = CalibrationKnowledge {
            domain: domain.to_string(),
            recalibration_points,
            quality,
            sample_size: profile.overall_curve.total_predictions,
            source_trust: source_agent.k_vector.trust_score(),
        };

        // Store in domain knowledge
        self.domain_knowledge
            .entry(domain.to_string())
            .or_default()
            .push(knowledge.clone());

        Some(knowledge)
    }

    /// Learn calibration from peer agents
    pub fn learn_from_peers(
        &mut self,
        learner_id: &str,
        domain: &str,
        min_trust: f32,
    ) -> Option<Vec<(f64, f64)>> {
        let knowledge = self.domain_knowledge.get(domain)?;

        // Filter by trust and aggregate
        let valid_knowledge: Vec<_> = knowledge
            .iter()
            .filter(|k| k.source_trust >= min_trust)
            .collect();

        if valid_knowledge.is_empty() {
            return None;
        }

        // Weight by trust and sample size
        let mut aggregated: HashMap<i32, (f64, f64, f64)> = HashMap::new(); // bucket -> (sum_adjusted, sum_weight, count)

        for k in valid_knowledge {
            let weight = k.source_trust as f64 * (k.sample_size as f64).sqrt();
            for (pred, actual) in &k.recalibration_points {
                let bucket = (*pred * 10.0) as i32;
                let entry = aggregated.entry(bucket).or_insert((0.0, 0.0, 0.0));
                entry.0 += actual * weight;
                entry.1 += weight;
                entry.2 += 1.0;
            }
        }

        // Convert to recalibration curve
        let mut curve: Vec<(f64, f64)> = aggregated
            .into_iter()
            .filter(|(_, (_, w, _))| *w > 0.0)
            .map(|(bucket, (sum, weight, _))| {
                let predicted = (bucket as f64 + 0.5) / 10.0;
                let adjusted = sum / weight;
                (predicted, adjusted)
            })
            .collect();

        curve.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Record learning
        self.learning_history
            .entry(learner_id.to_string())
            .or_default()
            .push(domain.to_string());

        Some(curve)
    }

    /// Apply learned recalibration to a prediction
    pub fn apply_learned_recalibration(curve: &[(f64, f64)], predicted: f64) -> f64 {
        if curve.is_empty() {
            return predicted;
        }

        // Find surrounding points and interpolate
        let predicted = predicted.clamp(0.0, 1.0);

        // Find the two closest points
        let mut lower: Option<&(f64, f64)> = None;
        let mut upper: Option<&(f64, f64)> = None;

        for point in curve {
            if point.0 <= predicted {
                lower = Some(point);
            }
            if point.0 >= predicted && upper.is_none() {
                upper = Some(point);
            }
        }

        match (lower, upper) {
            (Some(l), Some(u)) if (u.0 - l.0).abs() > 0.001 => {
                // Linear interpolation
                let t = (predicted - l.0) / (u.0 - l.0);
                l.1 + t * (u.1 - l.1)
            }
            (Some(l), _) => l.1,
            (_, Some(u)) => u.1,
            _ => predicted,
        }
    }
}

// ============================================================================
// Agent Collaboration Protocols
// ============================================================================

/// A collaborative task that multiple agents work on
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CollaborativeTask {
    /// Task identifier
    pub task_id: String,
    /// Task type
    pub task_type: CollaborativeTaskType,
    /// Required minimum trust to participate
    pub min_trust: f32,
    /// Required minimum participants
    pub min_participants: usize,
    /// Current participants
    pub participants: Vec<String>,
    /// Task status
    pub status: CollaborativeTaskStatus,
    /// Deadline timestamp
    pub deadline: u64,
    /// Collected contributions
    pub contributions: Vec<TaskContribution>,
    /// Final result (if completed)
    pub result: Option<CollaborativeResult>,
}

/// Types of collaborative tasks
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum CollaborativeTaskType {
    /// Multiple agents vote on a decision
    Voting,
    /// Agents contribute to aggregate prediction
    Prediction,
    /// Agents verify each other's outputs
    PeerReview,
    /// Agents contribute different parts of a task
    Division,
    /// Agents compete, best result wins
    Competition,
}

/// Task status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum CollaborativeTaskStatus {
    /// Accepting participants
    Open,
    /// Minimum reached, accepting contributions
    Active,
    /// Deadline passed, computing result
    Computing,
    /// Task completed
    Completed,
    /// Task failed (not enough participants, etc.)
    Failed,
}

/// A contribution to a collaborative task
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TaskContribution {
    /// Contributing agent
    pub agent_id: String,
    /// Agent's trust score at contribution time
    pub trust_score: f32,
    /// The contribution value (skipped in TS - use JSON parsing)
    #[cfg_attr(feature = "ts-export", ts(skip))]
    pub value: serde_json::Value,
    /// Epistemic classification of contribution
    pub epistemic_level: u8,
    /// Confidence
    pub confidence: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of a collaborative task
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CollaborativeResult {
    /// Final value/decision (skipped in TS - use JSON parsing)
    #[cfg_attr(feature = "ts-export", ts(skip))]
    pub value: serde_json::Value,
    /// Confidence in result
    pub confidence: f64,
    /// Participating agents
    pub participants: Vec<String>,
    /// Trust-weighted contribution scores
    #[cfg_attr(feature = "ts-export", ts(skip))]
    pub contribution_scores: HashMap<String, f64>,
    /// Completion timestamp
    pub completed_at: u64,
}

/// Manager for collaborative tasks
#[derive(Clone, Debug, Default)]
pub struct CollaborationManager {
    /// Active tasks
    pub tasks: HashMap<String, CollaborativeTask>,
    /// Agent participation history
    pub participation_history: HashMap<String, Vec<String>>,
    /// Collaboration quality scores (how well agents collaborate)
    pub collaboration_scores: HashMap<String, f64>,
}

impl CollaborationManager {
    /// Create a new collaboration manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new collaborative task
    pub fn create_task(
        &mut self,
        task_id: String,
        task_type: CollaborativeTaskType,
        min_trust: f32,
        min_participants: usize,
        deadline: u64,
    ) -> &CollaborativeTask {
        let task = CollaborativeTask {
            task_id: task_id.clone(),
            task_type,
            min_trust,
            min_participants,
            participants: Vec::new(),
            status: CollaborativeTaskStatus::Open,
            deadline,
            contributions: Vec::new(),
            result: None,
        };

        self.tasks.entry(task_id).or_insert(task)
    }

    /// Agent joins a task
    pub fn join_task(
        &mut self,
        task_id: &str,
        agent: &InstrumentalActor,
    ) -> Result<(), CollaborationError> {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or(CollaborationError::TaskNotFound)?;

        if task.status != CollaborativeTaskStatus::Open
            && task.status != CollaborativeTaskStatus::Active
        {
            return Err(CollaborationError::TaskNotAcceptingParticipants);
        }

        if agent.k_vector.trust_score() < task.min_trust {
            return Err(CollaborationError::InsufficientTrust);
        }

        if task
            .participants
            .contains(&agent.agent_id.as_str().to_string())
        {
            return Err(CollaborationError::AlreadyParticipating);
        }

        task.participants.push(agent.agent_id.as_str().to_string());

        // Update status if minimum reached
        if task.participants.len() >= task.min_participants
            && task.status == CollaborativeTaskStatus::Open
        {
            task.status = CollaborativeTaskStatus::Active;
        }

        // Record participation
        self.participation_history
            .entry(agent.agent_id.as_str().to_string())
            .or_default()
            .push(task_id.to_string());

        Ok(())
    }

    /// Submit a contribution to a task
    pub fn contribute(
        &mut self,
        task_id: &str,
        agent: &InstrumentalActor,
        value: serde_json::Value,
        epistemic_level: u8,
        confidence: f64,
        timestamp: u64,
    ) -> Result<(), CollaborationError> {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or(CollaborationError::TaskNotFound)?;

        if task.status != CollaborativeTaskStatus::Active {
            return Err(CollaborationError::TaskNotAcceptingContributions);
        }

        if !task
            .participants
            .contains(&agent.agent_id.as_str().to_string())
        {
            return Err(CollaborationError::NotParticipating);
        }

        // Check if already contributed
        if task
            .contributions
            .iter()
            .any(|c| c.agent_id == agent.agent_id.as_str())
        {
            return Err(CollaborationError::AlreadyContributed);
        }

        task.contributions.push(TaskContribution {
            agent_id: agent.agent_id.as_str().to_string(),
            trust_score: agent.k_vector.trust_score(),
            value,
            epistemic_level,
            confidence,
            timestamp,
        });

        Ok(())
    }

    /// Finalize a task and compute result
    pub fn finalize_task(
        &mut self,
        task_id: &str,
        timestamp: u64,
    ) -> Result<CollaborativeResult, CollaborationError> {
        // First, extract the data we need
        let (task_type, contributions) = {
            let task = self
                .tasks
                .get_mut(task_id)
                .ok_or(CollaborationError::TaskNotFound)?;

            if task.contributions.is_empty() {
                task.status = CollaborativeTaskStatus::Failed;
                return Err(CollaborationError::NoContributions);
            }

            task.status = CollaborativeTaskStatus::Computing;
            (task.task_type.clone(), task.contributions.clone())
        };

        // Compute trust-weighted result based on task type
        let result = match task_type {
            CollaborativeTaskType::Voting | CollaborativeTaskType::Prediction => {
                Self::compute_weighted_aggregate(&contributions, timestamp)
            }
            CollaborativeTaskType::PeerReview => {
                Self::compute_peer_review_result(&contributions, timestamp)
            }
            CollaborativeTaskType::Division => {
                Self::compute_division_result(&contributions, timestamp)
            }
            CollaborativeTaskType::Competition => {
                Self::compute_competition_result(&contributions, timestamp)
            }
        };

        // Update the task with the result
        if let Some(task) = self.tasks.get_mut(task_id) {
            task.result = Some(result.clone());
            task.status = CollaborativeTaskStatus::Completed;
        }

        // Update collaboration scores
        for (agent_id, score) in &result.contribution_scores {
            let entry = self
                .collaboration_scores
                .entry(agent_id.clone())
                .or_insert(0.5);
            *entry = (*entry * 0.9) + (score * 0.1); // EMA update
        }

        Ok(result)
    }

    fn compute_weighted_aggregate(
        contributions: &[TaskContribution],
        timestamp: u64,
    ) -> CollaborativeResult {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        let mut contribution_scores = HashMap::new();

        for contrib in contributions {
            // Try to extract numeric value
            let value = contrib.value.as_f64().unwrap_or(0.5);
            let weight = contrib.trust_score as f64
                * (1.0 + contrib.epistemic_level as f64 * 0.1)
                * contrib.confidence;

            weighted_sum += value * weight;
            total_weight += weight;
            contribution_scores.insert(contrib.agent_id.clone(), weight);
        }

        let final_value = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.5
        };

        // Normalize contribution scores
        for score in contribution_scores.values_mut() {
            *score /= total_weight.max(0.001);
        }

        CollaborativeResult {
            value: serde_json::json!(final_value),
            confidence: (total_weight / contributions.len() as f64).min(1.0),
            participants: contributions.iter().map(|c| c.agent_id.clone()).collect(),
            contribution_scores,
            completed_at: timestamp,
        }
    }

    fn compute_peer_review_result(
        contributions: &[TaskContribution],
        timestamp: u64,
    ) -> CollaborativeResult {
        // For peer review, look for consensus on approval/rejection
        let mut approvals = 0.0;
        let mut total_weight = 0.0;
        let mut contribution_scores = HashMap::new();

        for contrib in contributions {
            let approved = contrib.value.as_bool().unwrap_or(false);
            let weight = contrib.trust_score as f64 * contrib.confidence;

            if approved {
                approvals += weight;
            }
            total_weight += weight;
            contribution_scores.insert(contrib.agent_id.clone(), weight / total_weight.max(0.001));
        }

        let approval_ratio = approvals / total_weight.max(0.001);
        let approved = approval_ratio >= 0.5;

        CollaborativeResult {
            value: serde_json::json!({"approved": approved, "ratio": approval_ratio}),
            confidence: (approval_ratio - 0.5).abs() * 2.0, // Higher confidence when clear majority
            participants: contributions.iter().map(|c| c.agent_id.clone()).collect(),
            contribution_scores,
            completed_at: timestamp,
        }
    }

    fn compute_division_result(
        contributions: &[TaskContribution],
        timestamp: u64,
    ) -> CollaborativeResult {
        // For division tasks, aggregate all contributions
        let mut all_values = Vec::new();
        let mut contribution_scores = HashMap::new();

        for contrib in contributions {
            all_values.push(contrib.value.clone());
            contribution_scores.insert(
                contrib.agent_id.clone(),
                contrib.trust_score as f64 / contributions.len() as f64,
            );
        }

        CollaborativeResult {
            value: serde_json::json!(all_values),
            confidence: contribution_scores.values().sum::<f64>() / contributions.len() as f64,
            participants: contributions.iter().map(|c| c.agent_id.clone()).collect(),
            contribution_scores,
            completed_at: timestamp,
        }
    }

    fn compute_competition_result(
        contributions: &[TaskContribution],
        timestamp: u64,
    ) -> CollaborativeResult {
        // For competition, select highest-scoring contribution
        let winner = contributions.iter().max_by(|a, b| {
            let score_a = a.trust_score as f64 * a.confidence;
            let score_b = b.trust_score as f64 * b.confidence;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut contribution_scores = HashMap::new();
        if let Some(w) = winner {
            contribution_scores.insert(w.agent_id.clone(), 1.0);
            for contrib in contributions {
                if contrib.agent_id != w.agent_id {
                    contribution_scores.insert(contrib.agent_id.clone(), 0.0);
                }
            }

            CollaborativeResult {
                value: w.value.clone(),
                confidence: w.trust_score as f64 * w.confidence,
                participants: contributions.iter().map(|c| c.agent_id.clone()).collect(),
                contribution_scores,
                completed_at: timestamp,
            }
        } else {
            CollaborativeResult {
                value: serde_json::Value::Null,
                confidence: 0.0,
                participants: vec![],
                contribution_scores,
                completed_at: timestamp,
            }
        }
    }
}

/// Errors in collaboration
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollaborationError {
    /// Task ID does not exist
    TaskNotFound,
    /// Task is not accepting new participants
    TaskNotAcceptingParticipants,
    /// Task is not accepting contributions yet
    TaskNotAcceptingContributions,
    /// Agent's trust score too low for task
    InsufficientTrust,
    /// Agent already participating in task
    AlreadyParticipating,
    /// Agent not part of task
    NotParticipating,
    /// Agent already contributed to task
    AlreadyContributed,
    /// No contributions available for aggregation
    NoContributions,
}

// ============================================================================
// Reputation Propagation
// ============================================================================

/// Agent interaction for reputation propagation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentInteraction {
    /// Source agent
    pub from_agent: String,
    /// Target agent
    pub to_agent: String,
    /// Interaction type
    pub interaction_type: InteractionType,
    /// Quality rating (-1.0 to 1.0)
    pub quality: f64,
    /// Weight of this interaction
    pub weight: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Types of agent interactions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Collaborated on task
    Collaboration,
    /// Verified output
    Verification,
    /// Learned calibration from
    CalibrationLearning,
    /// Referred/recommended
    Referral,
    /// Disputed/challenged
    Dispute,
}

/// Reputation propagation system using PageRank-like algorithm
#[derive(Clone, Debug, Default)]
pub struct ReputationPropagation {
    /// Interaction history
    pub interactions: Vec<AgentInteraction>,
    /// Propagated reputation scores
    pub propagated_scores: HashMap<String, f64>,
    /// Damping factor for PageRank
    pub damping_factor: f64,
}

impl ReputationPropagation {
    /// Create a new reputation propagation engine
    pub fn new(damping_factor: f64) -> Self {
        Self {
            interactions: Vec::new(),
            propagated_scores: HashMap::new(),
            damping_factor: damping_factor.clamp(0.0, 1.0),
        }
    }

    /// Record an interaction
    pub fn record_interaction(&mut self, interaction: AgentInteraction) {
        self.interactions.push(interaction);
    }

    /// Compute propagated reputation scores
    pub fn propagate(&mut self, agents: &HashMap<String, InstrumentalActor>, iterations: usize) {
        // Initialize with base trust scores
        let mut scores: HashMap<String, f64> = agents
            .iter()
            .map(|(id, a)| (id.clone(), a.k_vector.trust_score() as f64))
            .collect();

        // Build adjacency list with weights
        let mut outgoing: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for interaction in &self.interactions {
            let weight = interaction.weight * interaction.quality.max(0.0);
            outgoing
                .entry(interaction.from_agent.clone())
                .or_default()
                .push((interaction.to_agent.clone(), weight));
        }

        // PageRank-like iteration
        for _ in 0..iterations {
            let mut new_scores: HashMap<String, f64> = HashMap::new();
            let _n = scores.len() as f64;

            for agent_id in scores.keys() {
                let base_score = agents
                    .get(agent_id)
                    .map(|a| a.k_vector.trust_score() as f64)
                    .unwrap_or(0.5);

                // Random walk contribution
                let random_contribution = (1.0 - self.damping_factor) * base_score;

                // Link contribution
                let mut link_contribution = 0.0;
                for (from_id, edges) in &outgoing {
                    let total_weight: f64 = edges.iter().map(|(_, w)| w).sum();
                    if total_weight > 0.0 {
                        for (to_id, weight) in edges {
                            if to_id == agent_id {
                                let from_score = scores.get(from_id).unwrap_or(&0.5);
                                link_contribution +=
                                    self.damping_factor * from_score * weight / total_weight;
                            }
                        }
                    }
                }

                new_scores.insert(agent_id.clone(), random_contribution + link_contribution);
            }

            scores = new_scores;
        }

        // Normalize scores to 0-1 range
        let max_score = scores.values().cloned().fold(0.0f64, f64::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }

        self.propagated_scores = scores;
    }

    /// Get propagated reputation for an agent
    pub fn get_propagated_reputation(&self, agent_id: &str) -> Option<f64> {
        self.propagated_scores.get(agent_id).copied()
    }

    /// Get combined reputation (base + propagated)
    pub fn get_combined_reputation(
        &self,
        agent: &InstrumentalActor,
        propagation_weight: f64,
    ) -> f64 {
        let base = agent.k_vector.trust_score() as f64;
        let propagated = self
            .propagated_scores
            .get(agent.agent_id.as_str())
            .copied()
            .unwrap_or(base);

        base * (1.0 - propagation_weight) + propagated * propagation_weight
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
    use crate::matl::KVector;

    fn create_test_agent(id: &str, trust: f32) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new(trust, trust, trust, trust, 0.5, 0.5, trust, 0.5, trust, 0.5),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_trust_weighted_consensus() {
        let mut agents = HashMap::new();
        agents.insert("agent-1".to_string(), create_test_agent("agent-1", 0.9));
        agents.insert("agent-2".to_string(), create_test_agent("agent-2", 0.7));
        agents.insert("agent-3".to_string(), create_test_agent("agent-3", 0.5));

        let votes = vec![
            AgentVote {
                agent_id: "agent-1".to_string(),
                value: 0.8,
                confidence: 0.9,
                epistemic_level: 3,
                reasoning: None,
                timestamp: 1000,
            },
            AgentVote {
                agent_id: "agent-2".to_string(),
                value: 0.7,
                confidence: 0.8,
                epistemic_level: 2,
                reasoning: None,
                timestamp: 1000,
            },
            AgentVote {
                agent_id: "agent-3".to_string(),
                value: 0.6,
                confidence: 0.7,
                epistemic_level: 1,
                reasoning: None,
                timestamp: 1000,
            },
        ];

        let result = compute_consensus(&votes, &agents, &ConsensusConfig::default());

        // Higher trust agents should pull consensus toward their value (0.8)
        assert!(result.consensus_value > 0.7);
        assert!(result.consensus_reached);
        assert_eq!(result.participant_count, 3);
        assert!(
            result.contributions.get("agent-1").unwrap()
                > result.contributions.get("agent-3").unwrap()
        );
    }

    #[test]
    fn test_cross_agent_calibration() {
        let mut cal = CrossAgentCalibration::new();
        let agent = create_test_agent("well-calibrated", 0.9);

        // Create a well-calibrated profile
        let mut profile = AgentCalibrationProfile::new("well-calibrated".to_string());
        for i in 0..100 {
            let p = (i % 10) as f64 / 10.0 + 0.05;
            let outcome = (i as f64 / 100.0) < p;
            profile.record_prediction(p, outcome, None);
        }

        // Share calibration
        let knowledge = cal.share_calibration(&agent, &profile, "science");
        assert!(knowledge.is_some());

        // Learn from it
        let curve = cal.learn_from_peers("learner", "science", 0.5);
        assert!(curve.is_some());
        assert!(!curve.unwrap().is_empty());
    }

    #[test]
    fn test_collaboration_task() {
        let mut manager = CollaborationManager::new();

        let agent1 = create_test_agent("agent-1", 0.8);
        let agent2 = create_test_agent("agent-2", 0.7);
        let agent3 = create_test_agent("agent-3", 0.6);

        // Create task
        manager.create_task(
            "task-1".to_string(),
            CollaborativeTaskType::Prediction,
            0.5,
            2,
            10000,
        );

        // Join task
        assert!(manager.join_task("task-1", &agent1).is_ok());
        assert!(manager.join_task("task-1", &agent2).is_ok());
        assert!(manager.join_task("task-1", &agent3).is_ok());

        // Contribute
        assert!(manager
            .contribute("task-1", &agent1, serde_json::json!(0.8), 3, 0.9, 1000)
            .is_ok());
        assert!(manager
            .contribute("task-1", &agent2, serde_json::json!(0.75), 2, 0.85, 1001)
            .is_ok());
        assert!(manager
            .contribute("task-1", &agent3, serde_json::json!(0.7), 2, 0.8, 1002)
            .is_ok());

        // Finalize
        let result = manager.finalize_task("task-1", 2000).unwrap();

        assert_eq!(result.participants.len(), 3);
        assert!(result.confidence > 0.5);
        // Higher trust agents should have more contribution
        assert!(
            result.contribution_scores.get("agent-1").unwrap()
                > result.contribution_scores.get("agent-3").unwrap()
        );
    }

    #[test]
    fn test_reputation_propagation() {
        let mut agents = HashMap::new();
        agents.insert("hub".to_string(), create_test_agent("hub", 0.9));
        agents.insert("spoke1".to_string(), create_test_agent("spoke1", 0.5));
        agents.insert("spoke2".to_string(), create_test_agent("spoke2", 0.5));

        let mut prop = ReputationPropagation::new(0.85);

        // Hub endorses spokes
        prop.record_interaction(AgentInteraction {
            from_agent: "hub".to_string(),
            to_agent: "spoke1".to_string(),
            interaction_type: InteractionType::Collaboration,
            quality: 1.0,
            weight: 1.0,
            timestamp: 1000,
        });
        prop.record_interaction(AgentInteraction {
            from_agent: "hub".to_string(),
            to_agent: "spoke2".to_string(),
            interaction_type: InteractionType::Collaboration,
            quality: 0.5,
            weight: 1.0,
            timestamp: 1000,
        });

        prop.propagate(&agents, 10);

        // Spoke1 should have higher propagated reputation than spoke2
        let spoke1_rep = prop.get_propagated_reputation("spoke1").unwrap();
        let spoke2_rep = prop.get_propagated_reputation("spoke2").unwrap();
        assert!(spoke1_rep > spoke2_rep);
    }
}
