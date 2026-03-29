// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Bridge Integrity Zome
//!
//! Cross-hApp integration with Identity, Knowledge DKG, and Symthaea.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Record of federation to another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederationRecord {
    /// Thought ID being federated
    pub thought_id: String,
    /// Target hApp
    pub target_happ: String,
    /// External ID in target system
    pub external_id: Option<String>,
    /// Federation status
    pub status: FederationStatus,
    /// Timestamp
    pub federated_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FederationStatus {
    Pending,
    Active,
    Revoked,
    Failed,
}

/// External reputation score from Identity hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ExternalReputation {
    /// Agent this score belongs to
    pub agent: AgentPubKey,
    /// K-Vector components (k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo)
    pub k_vector: [f64; 8],
    /// Composite trust score
    pub trust_score: f64,
    /// Source hApp
    pub source_happ: String,
    /// When this was fetched
    pub fetched_at: Timestamp,
}

/// Coherence analysis result computed by Symthaea via Tauri
///
/// This entry stores the results of coherence analysis performed by the
/// desktop application using Symthaea's consciousness engine. The analysis
/// is computed in the Tauri layer and stored here for persistence and sharing.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoherenceAnalysis {
    /// Unique ID for this analysis
    pub analysis_id: String,
    /// IDs of thoughts analyzed
    pub thought_ids: Vec<String>,
    /// Overall coherence score (0.0-1.0)
    pub overall: f64,
    /// Logical consistency score
    pub logical: f64,
    /// Temporal consistency score
    pub temporal: f64,
    /// Epistemic consistency score
    pub epistemic: f64,
    /// Harmonic (phi-based) coherence score
    pub harmonic: f64,
    /// Detected contradictions
    pub contradictions: Vec<DetectedContradiction>,
    /// When this analysis was performed
    pub analyzed_at: Timestamp,
}

/// A detected contradiction between two thoughts
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DetectedContradiction {
    /// First thought identifier (index or ID)
    pub thought_a: String,
    /// Second thought identifier
    pub thought_b: String,
    /// Description of the contradiction
    pub description: String,
    /// Severity (0.0-1.0)
    pub severity: f64,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    FederationRecord(FederationRecord),
    ExternalReputation(ExternalReputation),
    CoherenceAnalysis(CoherenceAnalysis),
}

#[hdk_link_types]
pub enum LinkTypes {
    ThoughtToFederation,
    AgentToReputation,
    ThoughtsToCoherence,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
