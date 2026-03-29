// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Reasoning Integrity Zome
//!
//! Coherence checking, contradiction detection, and inference.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A detected contradiction between thoughts
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contradiction {
    /// Unique identifier
    pub id: String,
    /// First thought ID
    pub thought_a_id: String,
    /// Second thought ID
    pub thought_b_id: String,
    /// Type of contradiction
    pub contradiction_type: ContradictionType,
    /// Severity (0.0-1.0)
    pub severity: f64,
    /// Explanation of the contradiction
    pub explanation: String,
    /// Suggested resolution
    pub suggested_resolution: Option<String>,
    /// Whether this has been resolved
    pub resolved: bool,
    /// Resolution notes
    pub resolution_notes: Option<String>,
    /// Detection timestamp
    pub detected_at: Timestamp,
    /// Resolution timestamp
    pub resolved_at: Option<Timestamp>,
    /// Phi (integrated information) at time of detection
    /// From Symthaea coherence analysis - indicates how fundamental the conflict is
    #[serde(default)]
    pub phi_at_detection: Option<f64>,
    /// Semantic similarity between the thoughts (from HDC embeddings)
    /// High similarity + contradiction = more concerning
    #[serde(default)]
    pub semantic_similarity: Option<f64>,
    /// Overall coherence score from Symthaea at time of detection
    #[serde(default)]
    pub coherence_at_detection: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContradictionType {
    /// Direct logical contradiction
    Logical,
    /// Factual inconsistency
    Factual,
    /// Temporal inconsistency
    Temporal,
    /// Source conflict
    SourceConflict,
    /// Confidence inconsistency
    ConfidenceInconsistency,
    /// Fundamental belief conflict (low phi, core disagreement)
    Fundamental,
    /// Surface-level apparent contradiction (may be resolvable)
    Apparent,
    /// Semantic divergence (same topic, opposite conclusions)
    SemanticDivergence,
}

/// A coherence report for a set of thoughts
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoherenceReport {
    /// Report identifier
    pub id: String,
    /// Thought IDs analyzed
    pub thought_ids: Vec<String>,
    /// Overall coherence score (0.0-1.0)
    pub coherence_score: f64,
    /// Phi (integrated information) estimate
    pub phi_estimate: Option<f64>,
    /// Number of contradictions found
    pub contradiction_count: u32,
    /// Knowledge gaps identified
    pub knowledge_gaps: Vec<String>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
    /// Report timestamp
    pub created_at: Timestamp,
}

/// An inference derived from existing thoughts
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Inference {
    /// Unique identifier
    pub id: String,
    /// The inferred claim
    pub content: String,
    /// Premise thought IDs
    pub premise_ids: Vec<String>,
    /// Type of inference
    pub inference_type: InferenceType,
    /// Confidence in the inference
    pub confidence: f64,
    /// Whether user has accepted this inference
    pub accepted: bool,
    /// Inference timestamp
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InferenceType {
    /// Logical deduction
    Deduction,
    /// Inductive generalization
    Induction,
    /// Abductive reasoning (best explanation)
    Abduction,
    /// Analogy-based
    Analogy,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Contradiction(Contradiction),
    CoherenceReport(CoherenceReport),
    Inference(Inference),
}

#[hdk_link_types]
pub enum LinkTypes {
    ThoughtToContradictions,
    AgentToReports,
    AgentToInferences,
    InferenceToPremises,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Contradiction(c) => {
                    if c.severity < 0.0 || c.severity > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("severity must be 0-1".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::CoherenceReport(r) => {
                    if r.coherence_score < 0.0 || r.coherence_score > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("coherence must be 0-1".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::Inference(i) => {
                    if i.confidence < 0.0 || i.confidence > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("confidence must be 0-1".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
