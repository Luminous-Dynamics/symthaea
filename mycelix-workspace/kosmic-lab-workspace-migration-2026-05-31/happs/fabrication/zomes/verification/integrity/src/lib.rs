// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Verification Integrity Zome
//!
//! Defines entry types for design verification and safety claims,
//! integrating with the Knowledge hApp for epistemic classification.

use fabrication_common::*;
use hdi::prelude::*;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    DesignVerification(DesignVerification),
    #[entry_type(visibility = "public")]
    SafetyClaim(SafetyClaim),
    #[entry_type(visibility = "public")]
    VerificationRequest(VerificationRequest),
}

#[hdk_link_types]
pub enum LinkTypes {
    DesignToVerifications,
    DesignToClaims,
    VerifierToVerifications,
    OpenRequests,
    ClaimToKnowledge,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DesignVerification {
    pub design_hash: ActionHash,
    pub verification_type: VerificationType,
    pub result: VerificationResult,
    pub evidence: Vec<ActionHash>,
    pub verifier: AgentPubKey,
    pub verifier_credentials: Vec<String>,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SafetyClaim {
    pub design_hash: ActionHash,
    pub claim_type: SafetyClaimType,
    pub claim_text: String,
    pub epistemic: ClaimEpistemic,
    pub supporting_evidence: Vec<String>,
    pub knowledge_claim_hash: Option<ActionHash>,
    pub author: AgentPubKey,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerificationRequest {
    pub design_hash: ActionHash,
    pub requester: AgentPubKey,
    pub target_safety_class: SafetyClass,
    pub bounty: Option<u64>,
    pub deadline: Option<Timestamp>,
    pub status: RequestStatus,
    pub created_at: Timestamp,
}

#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::SafetyClaim(c) => {
                    if c.claim_text.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Claim text required".into(),
                        ));
                    }
                    if c.epistemic.empirical < 0.0 || c.epistemic.empirical > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Invalid epistemic score".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
