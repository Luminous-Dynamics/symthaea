// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Evidence Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Evidence {
    pub id: String,
    pub complaint_id: String,
    pub submitter: String,
    pub evidence_type: EvidenceType,
    pub title: String,
    pub description: String,
    pub content_hash: String,
    pub encrypted_content: Option<String>,
    pub submitted: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EvidenceType {
    Document,
    Testimony,
    Transaction,
    CrossReference,
    Media,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EvidenceVerification {
    pub id: String,
    pub evidence_id: String,
    pub verifier: String,
    pub status: VerificationStatus,
    pub notes: Option<String>,
    pub verified_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    Verified,
    Challenged,
    Rejected,
    Pending,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EvidenceDispute {
    pub id: String,
    pub evidence_id: String,
    pub disputant: String,
    pub reason: String,
    pub created_at: Timestamp,
    pub resolved: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Evidence(Evidence),
    EvidenceVerification(EvidenceVerification),
    EvidenceDispute(EvidenceDispute),
}

#[hdk_link_types]
pub enum LinkTypes {
    ComplaintToEvidence,
    SubmitterToEvidence,
    EvidenceToVerification,
    EvidenceToDispute,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Evidence(evidence) => validate_evidence(&evidence),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Evidence(evidence) => validate_evidence(&evidence),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_evidence(evidence: &Evidence) -> ExternResult<ValidateCallbackResult> {
    if !evidence.submitter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Submitter must be a valid DID".into(),
        ));
    }
    if evidence.content_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
