// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG Integrity Zome - Entry Types and Validation
//!
//! Defines the on-chain data structures for the Distributed Knowledge Graph:
//! - ClaimEntry: Verifiable claims (subject-predicate-object triples)
//! - AttestationEntry: Social proof (endorse/challenge/acknowledge)
//! - AnchorEntry: Deterministic anchors for indexing

use hdi::prelude::*;

/// Maximum length for claim subjects
pub const MAX_SUBJECT_LEN: usize = 256;
/// Maximum length for predicates
pub const MAX_PREDICATE_LEN: usize = 256;
/// Maximum length for string objects
pub const MAX_OBJECT_LEN: usize = 1024;
/// Maximum length for evidence text
pub const MAX_EVIDENCE_LEN: usize = 2000;

// ============================================================================
// Entry Types
// ============================================================================

/// A claim stored on the DHT (subject-predicate-object triple)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClaimEntry {
    /// The subject of the claim (what is being described)
    pub subject: String,
    /// The predicate (relationship/property)
    pub predicate: String,
    /// The object value as a string (JSON-encoded for complex types)
    pub object: String,
    /// Object type hint: "text", "number", "integer", "boolean"
    pub object_type: String,
    /// Epistemic classification: "empirical", "normative", "metaphysical"
    pub epistemic_type: String,
    /// Optional domain for context
    pub domain: Option<String>,
    /// Unix timestamp of creation (seconds)
    pub created_at: u64,
}

impl ClaimEntry {
    /// Validate claim fields
    pub fn validate(&self) -> ExternResult<ValidateCallbackResult> {
        if self.subject.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Claim subject cannot be empty".to_string(),
            ));
        }
        if self.subject.len() > MAX_SUBJECT_LEN {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Subject exceeds maximum length of {} bytes",
                MAX_SUBJECT_LEN
            )));
        }
        if self.predicate.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Claim predicate cannot be empty".to_string(),
            ));
        }
        if self.predicate.len() > MAX_PREDICATE_LEN {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Predicate exceeds maximum length of {} bytes",
                MAX_PREDICATE_LEN
            )));
        }
        if self.object.len() > MAX_OBJECT_LEN {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Object exceeds maximum length of {} bytes",
                MAX_OBJECT_LEN
            )));
        }
        // Validate epistemic type
        let valid_types = ["empirical", "normative", "metaphysical"];
        if !valid_types.contains(&self.epistemic_type.as_str()) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid epistemic type: {}. Must be one of: {:?}",
                self.epistemic_type, valid_types
            )));
        }
        // Validate object type
        let valid_obj_types = ["text", "number", "integer", "boolean"];
        if !valid_obj_types.contains(&self.object_type.as_str()) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid object type: {}. Must be one of: {:?}",
                self.object_type, valid_obj_types
            )));
        }
        Ok(ValidateCallbackResult::Valid)
    }
}

/// An attestation stored on the DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AttestationEntry {
    /// Hash of the claim being attested
    pub claim_hash: ActionHash,
    /// Type: "endorse", "challenge", "acknowledge"
    pub attestation_type: String,
    /// Optional evidence/reasoning
    pub evidence: Option<String>,
    /// Unix timestamp (seconds)
    pub created_at: u64,
}

impl AttestationEntry {
    /// Validate attestation fields
    pub fn validate(&self) -> ExternResult<ValidateCallbackResult> {
        let valid_types = ["endorse", "challenge", "acknowledge"];
        if !valid_types.contains(&self.attestation_type.as_str()) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid attestation type: {}. Must be one of: {:?}",
                self.attestation_type, valid_types
            )));
        }
        if let Some(ref evidence) = self.evidence {
            if evidence.len() > MAX_EVIDENCE_LEN {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Evidence exceeds maximum length of {} bytes",
                    MAX_EVIDENCE_LEN
                )));
            }
        }
        Ok(ValidateCallbackResult::Valid)
    }

    /// Check if this is an endorsement
    pub fn is_endorsement(&self) -> bool {
        self.attestation_type == "endorse"
    }

    /// Check if this is a challenge
    pub fn is_challenge(&self) -> bool {
        self.attestation_type == "challenge"
    }
}

/// Anchor entry for indexing (deterministic hashes)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AnchorEntry {
    pub anchor: String,
}

// ============================================================================
// Consensus Types
// ============================================================================

/// Maximum length for dispute reason
pub const MAX_DISPUTE_REASON_LEN: usize = 2000;
/// Maximum number of evidence URIs in a dispute
pub const MAX_DISPUTE_EVIDENCE_COUNT: usize = 5;

/// Lifecycle status of a claim in the consensus process
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimStatus {
    /// Just submitted, below minimum attestation threshold
    Proposed,
    /// Has endorsements above low threshold
    Attested,
    /// Active dispute filed against this claim
    Contested,
    /// Dispute resolved or market-verified
    Resolved,
    /// High-confidence truth: quorum met
    Established,
    /// Temporal decay dropped confidence below threshold
    Decayed,
}

/// Status of a dispute
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    /// Dispute is open and awaiting resolution
    Open,
    /// Dispute was resolved (upheld or dismissed)
    Resolved,
    /// Dispute was dismissed
    Dismissed,
}

/// Snapshot of consensus state at evaluation time
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsensusSnapshot {
    /// Hash of the claim being evaluated
    pub claim_hash: ActionHash,
    /// Current status after evaluation
    pub status: ClaimStatus,
    /// Number of endorsements
    pub endorsement_count: u32,
    /// Number of challenges
    pub challenge_count: u32,
    /// Sum of endorser K-vector trust scores
    pub aggregate_reputation: f64,
    /// Computed confidence value (0.0-1.0)
    pub confidence: f64,
    /// Unix timestamp of evaluation (seconds)
    pub evaluated_at: u64,
}

/// A formal dispute against a claim
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DisputeEntry {
    /// Hash of the claim being disputed
    pub claim_hash: ActionHash,
    /// Agent filing the dispute
    pub challenger: AgentPubKey,
    /// Reason for the dispute (max 2000 chars)
    pub reason: String,
    /// Supporting evidence URIs (max 5)
    pub evidence: Vec<String>,
    /// Current dispute status
    pub status: DisputeStatus,
    /// Resolution text (set when resolved)
    pub resolution: Option<String>,
    /// Unix timestamp of creation (seconds)
    pub created_at: u64,
}

impl ConsensusSnapshot {
    /// Validate consensus snapshot fields
    pub fn validate(&self) -> ExternResult<ValidateCallbackResult> {
        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Confidence must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.aggregate_reputation < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Aggregate reputation cannot be negative".to_string(),
            ));
        }
        Ok(ValidateCallbackResult::Valid)
    }
}

impl DisputeEntry {
    /// Validate dispute fields
    pub fn validate(&self) -> ExternResult<ValidateCallbackResult> {
        if self.reason.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Dispute reason cannot be empty".to_string(),
            ));
        }
        if self.reason.len() > MAX_DISPUTE_REASON_LEN {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Dispute reason exceeds maximum length of {} bytes",
                MAX_DISPUTE_REASON_LEN
            )));
        }
        if self.evidence.len() > MAX_DISPUTE_EVIDENCE_COUNT {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Dispute evidence exceeds maximum of {} URIs",
                MAX_DISPUTE_EVIDENCE_COUNT
            )));
        }
        Ok(ValidateCallbackResult::Valid)
    }
}

// ============================================================================
// Entry Types Enum
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A verifiable claim (subject-predicate-object)
    Claim(ClaimEntry),
    /// An attestation (endorse/challenge/acknowledge)
    Attestation(AttestationEntry),
    /// Anchor for indexing
    Anchor(AnchorEntry),
    /// Consensus evaluation snapshot
    ConsensusSnapshot(ConsensusSnapshot),
    /// Formal dispute against a claim
    Dispute(DisputeEntry),
}

// ============================================================================
// Link Types
// ============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    /// Subject anchor -> Claim entries
    SubjectToClaim,
    /// Claim -> Attestations
    ClaimToAttestation,
    /// Agent -> Their claims (for reputation tracking)
    AgentToClaim,
    /// Agent -> Their attestations
    AgentToAttestation,
    /// Global anchor -> All subjects (for discovery)
    AllSubjects,
    /// Claim -> Consensus snapshots
    ClaimToConsensus,
    /// Claim -> Disputes
    ClaimToDispute,
    /// Agent -> Disputes they filed
    AgentToDispute,
}

// ============================================================================
// Validation Callbacks
// ============================================================================

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Claim(claim) => claim.validate(),
                EntryTypes::Attestation(att) => att.validate(),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ConsensusSnapshot(snap) => snap.validate(),
                EntryTypes::Dispute(dispute) => dispute.validate(),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Claim(claim) => claim.validate(),
                EntryTypes::Attestation(att) => att.validate(),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ConsensusSnapshot(snap) => snap.validate(),
                EntryTypes::Dispute(dispute) => dispute.validate(),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },

        FlatOp::RegisterCreateLink { .. } => {
            Ok(ValidateCallbackResult::Valid)
        }

        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original link creator can delete a link
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete a link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        FlatOp::RegisterUpdate(op_update) => {
            // Only the original author can update an entry
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        FlatOp::RegisterDelete(op_delete) => {
            // Only the original author can delete an entry
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        // Default: allow other operations
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
