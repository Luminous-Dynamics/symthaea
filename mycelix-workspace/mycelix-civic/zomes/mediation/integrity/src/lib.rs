// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mediation Integrity Zome
//!
//! Entry types and validation for community mediation — a pre-case
//! dispute resolution pathway that sits before the formal justice system.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

// ============================================================================
// MEDIATION REQUEST TYPES
// ============================================================================

/// Category of mediation dispute.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MediationCategory {
    Neighbor,
    Property,
    Noise,
    SharedSpace,
    Financial,
    Other,
}

/// Urgency level for a mediation request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MediationUrgency {
    Low,
    Medium,
    High,
}

/// Status of a mediation request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MediationStatus {
    Requested,
    Accepted,
    InProgress,
    Resolved,
    Escalated,
    Withdrawn,
}

/// A request for community mediation.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct MediationRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// DID of the person requesting mediation.
    pub requester_did: String,
    /// DID of the other party.
    pub respondent_did: String,
    /// Brief subject line.
    pub subject: String,
    /// Full description of the dispute.
    pub description: String,
    /// Category of the dispute.
    pub category: MediationCategory,
    /// How urgent this is.
    pub urgency: MediationUrgency,
    /// Current status.
    pub status: MediationStatus,
}

// ============================================================================
// SESSION & AGREEMENT TYPES
// ============================================================================

/// A mediation session record.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct MediationSession {
    /// Request this session is for (request_id).
    pub request_id: String,
    /// Mediator DID.
    pub mediator_did: String,
    /// Session notes.
    pub session_notes: String,
    /// Agreements reached during this session.
    pub agreements: Vec<String>,
    /// When the session took place.
    pub session_at: Timestamp,
    /// Duration of the session in minutes.
    pub duration_minutes: u32,
}

/// A formal mediation agreement.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct MediationAgreement {
    /// Request this agreement is for (request_id).
    pub request_id: String,
    /// Terms of the agreement.
    pub terms: Vec<String>,
    /// DIDs of parties who have agreed.
    pub agreed_by: Vec<String>,
    /// When the agreement was reached.
    pub agreed_at: Timestamp,
    /// Whether this agreement is binding.
    pub binding: bool,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "MediationRequest", visibility = "public")]
    MediationRequest(MediationRequest),
    #[entry_type(name = "MediationSession", visibility = "public")]
    MediationSession(MediationSession),
    #[entry_type(name = "MediationAgreement", visibility = "public")]
    MediationAgreement(MediationAgreement),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllRequests,
    AgentToRequest,
    RequestToSession,
    RequestToAgreement,
    MediatorToSession,
    RequestsByStatus,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::MediationRequest(req) => validate_request(&req),
                    EntryTypes::MediationSession(session) => validate_session(&session),
                    EntryTypes::MediationAgreement(agreement) => validate_agreement(&agreement),
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_request(req: &MediationRequest) -> ExternResult<ValidateCallbackResult> {
    if req.request_id.is_empty() || req.request_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID must be 1-256 chars".to_string(),
        ));
    }
    if req.requester_did.is_empty() || req.requester_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester DID must be 1-256 chars".to_string(),
        ));
    }
    if req.respondent_did.is_empty() || req.respondent_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent DID must be 1-256 chars".to_string(),
        ));
    }
    if req.requester_did == req.respondent_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot request mediation with yourself".to_string(),
        ));
    }
    if req.subject.is_empty() || req.subject.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be 1-512 chars".to_string(),
        ));
    }
    if req.description.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description too long (max 8192)".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_session(session: &MediationSession) -> ExternResult<ValidateCallbackResult> {
    if session.request_id.is_empty() || session.request_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID must be 1-256 chars".to_string(),
        ));
    }
    if session.mediator_did.is_empty() || session.mediator_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mediator DID must be 1-256 chars".to_string(),
        ));
    }
    if session.session_notes.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Session notes too long (max 16384)".to_string(),
        ));
    }
    if session.duration_minutes == 0 || session.duration_minutes > 480 {
        return Ok(ValidateCallbackResult::Invalid(
            "Duration must be 1-480 minutes".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_agreement(agreement: &MediationAgreement) -> ExternResult<ValidateCallbackResult> {
    if agreement.request_id.is_empty() || agreement.request_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID must be 1-256 chars".to_string(),
        ));
    }
    if agreement.terms.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agreement must have at least one term".to_string(),
        ));
    }
    if agreement.terms.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many terms (max 100)".to_string(),
        ));
    }
    for term in &agreement.terms {
        if term.len() > 2048 {
            return Ok(ValidateCallbackResult::Invalid(
                "Individual term too long (max 2048)".to_string(),
            ));
        }
    }
    if agreement.agreed_by.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agreement must have at least one signatory".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
