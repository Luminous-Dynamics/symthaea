// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Consciousness Integrity Zome
//!
//! Tracks belief trajectories over time, enabling analysis of:
//! - Belief stability vs. volatility
//! - Entrenchment detection (beliefs resistant to contradictory evidence)
//! - Epistemic growth patterns
//! - Consciousness evolution across time

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A trajectory tracking the evolution of a belief over time
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BeliefTrajectory {
    /// The thought ID this trajectory tracks
    pub thought_id: String,
    /// Snapshots of the belief at various points in time
    pub snapshots: Vec<BeliefSnapshot>,
    /// Current trajectory type classification
    pub trajectory_type: TrajectoryType,
    /// Whether this belief shows signs of entrenchment
    pub entrenchment: Option<EntrenchmentLevel>,
    /// First recorded timestamp
    pub started_at: Timestamp,
    /// Last updated timestamp
    pub last_updated: Timestamp,
}

/// A snapshot of belief state at a point in time
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BeliefSnapshot {
    /// E/N/M/H classification at this time
    pub epistemic_code: String,
    /// Confidence at this time
    pub confidence: f64,
    /// Phi (integrated information) at this time
    pub phi: f64,
    /// Coherence with rest of knowledge graph
    pub coherence: f64,
    /// Timestamp of this snapshot
    pub timestamp: Timestamp,
    /// What triggered this snapshot (edit, review, contradiction, etc.)
    pub trigger: SnapshotTrigger,
}

/// What triggered the creation of a belief snapshot
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SnapshotTrigger {
    /// Initial creation
    Created,
    /// User edited the belief
    Edited,
    /// Periodic scheduled snapshot
    Scheduled,
    /// Contradiction detected with this belief
    ContradictionDetected,
    /// User explicitly reviewed and confirmed
    ReviewedConfirmed,
    /// User reviewed and modified
    ReviewedModified,
    /// External evidence affected this belief
    EvidenceUpdated,
    /// Coherence analysis changed
    CoherenceChanged,
}

/// Classification of how a belief has evolved
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum TrajectoryType {
    /// Belief hasn't meaningfully changed
    Stable,
    /// Confidence has been consistently increasing
    Growing,
    /// Confidence has been consistently decreasing
    Weakening,
    /// Confidence oscillates back and forth
    Oscillating,
    /// Entrenched despite contradictory evidence
    Entrenched,
    /// Rapidly evolving (many changes in short time)
    Volatile,
    /// Too few snapshots to classify
    Insufficient,
}

/// Level of entrenchment (resistance to change despite evidence)
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum EntrenchmentLevel {
    /// No signs of entrenchment
    None,
    /// Mild entrenchment (one contradiction ignored)
    Mild,
    /// Moderate entrenchment (multiple contradictions ignored)
    Moderate,
    /// High entrenchment (belief strengthened despite contradictions)
    High,
    /// Severe entrenchment (actively resistant to any evidence)
    Severe,
}

/// Analysis of a belief trajectory
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrajectoryAnalysis {
    /// Analysis ID
    pub id: String,
    /// Thought ID analyzed
    pub thought_id: String,
    /// Classified trajectory type
    pub trajectory_type: TrajectoryType,
    /// Confidence trend (-1.0 decreasing to 1.0 increasing)
    pub confidence_trend: f64,
    /// Phi trend (-1.0 decreasing to 1.0 increasing)
    pub phi_trend: f64,
    /// Coherence trend
    pub coherence_trend: f64,
    /// Entrenchment level if detected
    pub entrenchment: Option<EntrenchmentLevel>,
    /// Number of contradictions during trajectory
    pub contradiction_count: u32,
    /// Recommendation for the user
    pub recommendation: String,
    /// Analysis timestamp
    pub analyzed_at: Timestamp,
}

/// Consciousness evolution summary over a time period
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousnessEvolution {
    /// Evolution record ID
    pub id: String,
    /// Start of the period analyzed
    pub period_start: Timestamp,
    /// End of the period analyzed
    pub period_end: Timestamp,
    /// Average phi across period
    pub avg_phi: f64,
    /// Phi trend over period
    pub phi_trend: f64,
    /// Average coherence across period
    pub avg_coherence: f64,
    /// Coherence trend over period
    pub coherence_trend: f64,
    /// Number of stable beliefs
    pub stable_belief_count: u32,
    /// Number of growing beliefs
    pub growing_belief_count: u32,
    /// Number of weakening beliefs
    pub weakening_belief_count: u32,
    /// Number of entrenched beliefs (concern)
    pub entrenched_belief_count: u32,
    /// Key insights from the period
    pub insights: Vec<String>,
    /// Record creation timestamp
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    #[entry_type(name = "BeliefTrajectory", visibility = "private")]
    BeliefTrajectory(BeliefTrajectory),
    #[entry_type(name = "TrajectoryAnalysis", visibility = "private")]
    TrajectoryAnalysis(TrajectoryAnalysis),
    #[entry_type(name = "ConsciousnessEvolution", visibility = "private")]
    ConsciousnessEvolution(ConsciousnessEvolution),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Thought to its trajectory
    ThoughtToTrajectory,
    /// Trajectory to its analyses
    TrajectoryToAnalyses,
    /// Agent to their evolution records
    AgentToEvolution,
    /// Time-based index for evolution records
    EvolutionTimeIndex,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::BeliefTrajectory(t) => {
                    if t.thought_id.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("thought_id required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::TrajectoryAnalysis(a) => {
                    if a.thought_id.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("thought_id required".into()));
                    }
                    if a.confidence_trend < -1.0 || a.confidence_trend > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("confidence_trend must be -1 to 1".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::ConsciousnessEvolution(e) => {
                    if e.avg_phi < 0.0 || e.avg_phi > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("avg_phi must be 0 to 1".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
