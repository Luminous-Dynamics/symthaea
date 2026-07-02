// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Contribution Integrity Zome
//!
//! Defines entry types and validation rules for ZK-verified causal contributions
//! on Holochain 0.6. This zome tracks verified contributions to the federated
//! learning network with RISC Zero proof attestations.

use hdi::prelude::*;

/// Records the causal contribution of an agent for a single round.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CausalContributionRecord {
    /// The agent whose contribution is being measured.
    pub agent: AgentPubKey,
    /// The round or task ID for which the contribution is being measured.
    pub round_id: u64,
    /// The causal contribution score (C_i).
    pub contribution_score: f64,
    /// A link to the RISC Zero receipt (proof) that validates this calculation.
    pub proof_hash: EntryHash,
    /// The timestamp of the contribution.
    pub timestamp: Timestamp,
}

/// Stores the raw RISC0 receipt bytes (opaque to zome) and attestations
/// by committee members for later audit/verification.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReceiptEntry {
    /// Raw receipt data (RISC Zero proof bytes)
    pub data: Vec<u8>,
    /// Committee members who have attested to this receipt
    pub attesters: Vec<AgentPubKey>,
    /// Signatures from attesters
    pub signatures: Vec<Signature>,
    /// Optional opaque metadata (e.g., envelope/spec)
    pub meta: Option<Vec<u8>>,
}

/// Configurable policy values for Civitas execution.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CivitasConfig {
    /// Minimum attestations required for a receipt to be valid
    pub min_attestations: u8,
    /// Administrators who can update the config
    pub admins: Vec<AgentPubKey>,
    /// Maximum L-infinity norm for gradient envelope
    pub max_envelope_linf: Option<f32>,
}

/// Committee membership for a given round.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommitteeEntry {
    /// The round this committee is assigned to
    pub round_id: u64,
    /// Members of the verification committee
    pub members: Vec<AgentPubKey>,
}

/// An aggregated causal score for an agent, updated over time.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregatedCausalScore {
    /// The agent whose score this is.
    pub agent: AgentPubKey,
    /// The aggregated causal score.
    pub total_score: f64,
    /// The number of rounds the agent has participated in.
    pub rounds_participated: u64,
    /// The last time the score was updated.
    pub last_updated: Timestamp,
}

/// Entry types for the causal contribution system
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    CausalContributionRecord(CausalContributionRecord),
    ReceiptEntry(ReceiptEntry),
    CivitasConfig(CivitasConfig),
    CommitteeEntry(CommitteeEntry),
    AggregatedCausalScore(AggregatedCausalScore),
}

/// Link types for the causal contribution system
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from agent to their causal score entries
    AgentToCausalScore,
    /// Links from agent to their contribution records
    AgentToContributionRecords,
    /// Anchor link for Civitas config
    CivitasConfigAnchor,
    /// Anchor link for committees
    CommitteeAnchor,
    /// Anchor link for rounds
    RoundAnchor,
}

// === Validation Functions ===

/// Validation for CausalContributionRecord
pub fn validate_contribution_record(record: CausalContributionRecord) -> ExternResult<ValidateCallbackResult> {
    // Contribution score should be reasonable (not NaN or infinite)
    if record.contribution_score.is_nan() || record.contribution_score.is_infinite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contribution score must be a valid number".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation for ReceiptEntry
pub fn validate_receipt_entry(receipt: ReceiptEntry) -> ExternResult<ValidateCallbackResult> {
    // Receipt data must not be empty
    if receipt.data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Receipt data cannot be empty".to_string(),
        ));
    }
    // Attesters and signatures must match in length
    if receipt.attesters.len() != receipt.signatures.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Number of attesters must match number of signatures".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation for CivitasConfig
pub fn validate_civitas_config(config: CivitasConfig) -> ExternResult<ValidateCallbackResult> {
    if config.min_attestations == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum attestations must be at least 1".to_string(),
        ));
    }
    if config.admins.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Config must have at least one admin".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation for CommitteeEntry
pub fn validate_committee_entry(committee: CommitteeEntry) -> ExternResult<ValidateCallbackResult> {
    if committee.members.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Committee must have at least one member".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation for AggregatedCausalScore
pub fn validate_aggregated_score(score: AggregatedCausalScore) -> ExternResult<ValidateCallbackResult> {
    if score.total_score.is_nan() || score.total_score.is_infinite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Total score must be a valid number".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::CausalContributionRecord(record)) => validate_contribution_record(record),
                    Some(EntryTypes::ReceiptEntry(receipt)) => validate_receipt_entry(receipt),
                    Some(EntryTypes::CivitasConfig(config)) => validate_civitas_config(config),
                    Some(EntryTypes::CommitteeEntry(committee)) => validate_committee_entry(committee),
                    Some(EntryTypes::AggregatedCausalScore(score)) => validate_aggregated_score(score),
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
