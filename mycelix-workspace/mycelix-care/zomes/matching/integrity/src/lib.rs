// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Matching Integrity Zome
//! Defines entry types and validation for intelligent care matching.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Status of a care match
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MatchStatus {
    Suggested,
    Accepted,
    Declined,
    Completed,
}

/// Factors contributing to a match score
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MatchFactors {
    /// How close the provider is to the requester (0.0 - 1.0)
    pub proximity_score: f32,
    /// How well the provider's skills align with the request (0.0 - 1.0)
    pub skill_alignment: f32,
    /// How compatible the schedules are (0.0 - 1.0)
    pub schedule_compatibility: f32,
    /// Trust/reputation score between the agents (0.0 - 1.0)
    pub trust_score: f32,
}

/// A match between a service offer and a service request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareMatch {
    /// Hash of the ServiceOffer
    pub offer_hash: ActionHash,
    /// Hash of the ServiceRequest
    pub request_hash: ActionHash,
    /// Provider agent
    pub provider: AgentPubKey,
    /// Requester agent
    pub requester: AgentPubKey,
    /// Overall match score (0.0 - 1.0)
    pub score: f32,
    /// Breakdown of contributing factors
    pub factors: MatchFactors,
    /// Current status of the match
    pub status: MatchStatus,
    /// When the match was suggested
    pub created_at: Timestamp,
    /// When the match status was last updated
    pub updated_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CareMatch(CareMatch),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Request to suggested matches
    RequestToMatch,
    /// Offer to suggested matches
    OfferToMatch,
    /// Agent to matches they are involved in (as provider)
    AgentProviderMatches,
    /// Agent to matches they are involved in (as requester)
    AgentRequesterMatches,
    /// All pending matches
    AllPendingMatches,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareMatch(care_match) => validate_create_match(action, care_match),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareMatch(care_match) => validate_update_match(care_match),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_match(
    _action: Create,
    care_match: CareMatch,
) -> ExternResult<ValidateCallbackResult> {
    if care_match.score < 0.0 || care_match.score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match score must be between 0.0 and 1.0".into(),
        ));
    }
    if care_match.factors.proximity_score < 0.0 || care_match.factors.proximity_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proximity score must be between 0.0 and 1.0".into(),
        ));
    }
    if care_match.factors.skill_alignment < 0.0 || care_match.factors.skill_alignment > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Skill alignment must be between 0.0 and 1.0".into(),
        ));
    }
    if care_match.factors.schedule_compatibility < 0.0
        || care_match.factors.schedule_compatibility > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule compatibility must be between 0.0 and 1.0".into(),
        ));
    }
    if care_match.factors.trust_score < 0.0 || care_match.factors.trust_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score must be between 0.0 and 1.0".into(),
        ));
    }
    if care_match.provider == care_match.requester {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider and requester cannot be the same agent".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_match(care_match: CareMatch) -> ExternResult<ValidateCallbackResult> {
    if care_match.score < 0.0 || care_match.score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match score must be between 0.0 and 1.0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
