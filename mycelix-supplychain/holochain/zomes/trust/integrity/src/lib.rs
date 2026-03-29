// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Integrity Zome - Entry types for reputation and compliance
use hdi::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ReputationCategory { Reliability, Quality, Communication, Timeliness, Compliance }

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CertificationType { ISO9001, ISO14001, GDPR, SOC2, FairTrade, Organic, Custom(String) }

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReputationScore {
    pub agent: AgentPubKey,
    pub category: ReputationCategory,
    pub score: u32,
    pub total_reviews: u64,
    pub updated_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Review {
    pub subject: AgentPubKey,
    pub reviewer: AgentPubKey,
    pub po_hash: Option<ActionHash>,
    pub category: ReputationCategory,
    pub rating: u8,
    pub comment: Option<String>,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Certification {
    pub holder: AgentPubKey,
    pub cert_type: CertificationType,
    pub issuer: String,
    pub certificate_id: String,
    pub issued_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub verified_by: Option<AgentPubKey>,
    pub verified_at: Option<Timestamp>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Dispute {
    pub po_hash: ActionHash,
    pub claimant: AgentPubKey,
    pub respondent: AgentPubKey,
    pub description: String,
    pub evidence_hashes: Vec<ActionHash>,
    pub resolution: Option<String>,
    pub resolved_at: Option<Timestamp>,
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    ReputationScore(ReputationScore),
    Review(Review),
    Certification(Certification),
    Dispute(Dispute),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToReputation,
    AgentToReviews,
    AgentToCertifications,
    PoToDisputes,
    ReviewerToReviews,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::Review(r) => {
                    if r.rating > 5 { return Ok(ValidateCallbackResult::Invalid("Rating must be 0-5".into())); }
                    if r.subject == r.reviewer { return Ok(ValidateCallbackResult::Invalid("Cannot review yourself".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::Certification(c) => {
                    if c.certificate_id.is_empty() { return Ok(ValidateCallbackResult::Invalid("Certificate ID required".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::Dispute(d) => {
                    if d.description.is_empty() { return Ok(ValidateCallbackResult::Invalid("Dispute description required".into())); }
                    if d.claimant == d.respondent { return Ok(ValidateCallbackResult::Invalid("Cannot dispute with yourself".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
