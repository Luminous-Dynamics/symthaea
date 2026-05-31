// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Disputes Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PropertyDispute {
    pub id: String,
    pub property_id: String,
    pub dispute_type: DisputeType,
    pub claimant_did: String,
    pub respondent_did: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
    pub status: DisputeStatus,
    pub justice_case_id: Option<String>,
    pub filed: Timestamp,
    pub resolved: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeType {
    Boundary,
    Ownership,
    Encumbrance,
    Easement,
    Trespass,
    Damage,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    Filed,
    UnderReview,
    Mediation,
    Arbitration,
    Resolved,
    Dismissed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OwnershipClaim {
    pub id: String,
    pub property_id: String,
    pub claimant_did: String,
    pub claim_basis: ClaimBasis,
    pub supporting_documents: Vec<String>,
    pub status: ClaimStatus,
    pub filed: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimBasis {
    PriorOwnership,
    Inheritance,
    AdversePossession,
    FraudulentTransfer,
    DocumentaryEvidence,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimStatus {
    Pending,
    UnderInvestigation,
    Validated,
    Rejected,
    Superseded,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PropertyDispute(PropertyDispute),
    OwnershipClaim(OwnershipClaim),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    PropertyToDisputes,
    ClaimantToDisputes,
    PropertyToClaims,
    DisputeToJustice,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::PropertyDispute(dispute) => {
                    validate_create_property_dispute(EntryCreationAction::Create(action), dispute)
                }
                EntryTypes::OwnershipClaim(claim) => {
                    validate_create_ownership_claim(EntryCreationAction::Create(action), claim)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::PropertyDispute(dispute) => {
                    validate_update_property_dispute(action, dispute)
                }
                EntryTypes::OwnershipClaim(claim) => validate_update_ownership_claim(action, claim),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                    "Anchors cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::PropertyToDisputes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimantToDisputes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PropertyToClaims => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisputeToJustice => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_property_dispute(
    _action: EntryCreationAction,
    dispute: PropertyDispute,
) -> ExternResult<ValidateCallbackResult> {
    if !dispute.claimant_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Claimant must be a valid DID".into(),
        ));
    }
    if !dispute.respondent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_property_dispute(
    _action: Update,
    _dispute: PropertyDispute,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_ownership_claim(
    _action: EntryCreationAction,
    claim: OwnershipClaim,
) -> ExternResult<ValidateCallbackResult> {
    if !claim.claimant_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Claimant must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_ownership_claim(
    _action: Update,
    _claim: OwnershipClaim,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}
