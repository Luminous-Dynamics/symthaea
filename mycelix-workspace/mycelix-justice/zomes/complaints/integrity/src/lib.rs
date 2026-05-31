// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Complaints Integrity Zome
//! Entry types and validation for dispute complaints

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Complaint {
    pub id: String,
    pub complainant: String,
    pub respondent: String,
    pub title: String,
    pub description: String,
    pub complaint_type: ComplaintType,
    pub status: ComplaintStatus,
    pub evidence_ids: Vec<String>,
    pub relief_sought: String,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ComplaintType {
    ContractDispute,
    PropertyDispute,
    CredentialDispute,
    GovernanceDispute,
    FinancialDispute,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ComplaintStatus {
    Open,
    UnderReview,
    Filed,
    Mediation,
    Arbitration,
    Appeal,
    Resolved,
    Dismissed,
    Withdrawn,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Response {
    pub id: String,
    pub complaint_id: String,
    pub respondent: String,
    pub content: String,
    pub submitted: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Complaint(Complaint),
    Response(Response),
}

#[hdk_link_types]
pub enum LinkTypes {
    ComplainantToComplaint,
    RespondentToComplaint,
    StatusToComplaint,
    ComplaintToResponse,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Complaint(complaint) => validate_complaint(&complaint),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Complaint(complaint) => validate_complaint(&complaint),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_complaint(complaint: &Complaint) -> ExternResult<ValidateCallbackResult> {
    if !complaint.complainant.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Complainant must be a valid DID".into(),
        ));
    }
    if !complaint.respondent.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent must be a valid DID".into(),
        ));
    }
    if complaint.complainant == complaint.respondent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot file complaint against yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
