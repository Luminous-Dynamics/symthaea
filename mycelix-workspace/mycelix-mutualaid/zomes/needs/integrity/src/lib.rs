// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Needs Integrity Zome
//!
//! This zome defines entry types and validation rules for needs matching
//! in the Mycelix Mutual Aid hApp. Supports needs, offers, matches, and fulfillments.

use hdi::prelude::*;
use mutualaid_common::*;

/// Entry types for the needs zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A need from a member
    #[entry_type(visibility = "public")]
    Need(Need),
    /// An offer from a member
    #[entry_type(visibility = "public")]
    Offer(Offer),
    /// A match between need and offer
    #[entry_type(visibility = "public")]
    Match(Match),
    /// Fulfillment record
    #[entry_type(visibility = "public")]
    Fulfillment(Fulfillment),
}

/// Link types for the needs zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from agent to their needs
    AgentToNeeds,
    /// Link from agent to their offers
    AgentToOffers,
    /// Link from category anchor to needs
    CategoryToNeeds,
    /// Link from category anchor to offers
    CategoryToOffers,
    /// Link from need to its matches
    NeedToMatches,
    /// Link from offer to its matches
    OfferToMatches,
    /// Link from match to fulfillment
    MatchToFulfillment,
    /// Link for all needs discovery
    AllNeeds,
    /// Link for all offers discovery
    AllOffers,
    /// Link for emergency needs
    EmergencyNeeds,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::Need(need) => validate_need(need),
        EntryTypes::Offer(offer) => validate_offer(offer),
        EntryTypes::Match(m) => validate_match(m),
        EntryTypes::Fulfillment(fulfillment) => validate_fulfillment(fulfillment),
    }
}

/// Validate a need
fn validate_need(need: Need) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if need.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Need ID cannot be empty".to_string(),
        ));
    }

    // Title must not be empty
    if need.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Need title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if need.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Need title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if need.description.len() > 3000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Need description cannot exceed 3000 characters".to_string(),
        ));
    }

    // Reciprocity offers limit
    if need.reciprocity_offers.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 10 reciprocity offers".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate an offer
fn validate_offer(offer: Offer) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if offer.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer ID cannot be empty".to_string(),
        ));
    }

    // Title must not be empty
    if offer.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if offer.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if offer.description.len() > 3000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer description cannot exceed 3000 characters".to_string(),
        ));
    }

    // Asking for limit
    if offer.asking_for.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 10 asking for items".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a match
fn validate_match(m: Match) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if m.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Match ID cannot be empty".to_string(),
        ));
    }

    // Requester and offerer must be different
    if m.requester == m.offerer {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester and offerer must be different".to_string(),
        ));
    }

    // Notes length limit
    if m.notes.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match notes cannot exceed 1000 characters".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a fulfillment
fn validate_fulfillment(fulfillment: Fulfillment) -> ExternResult<ValidateCallbackResult> {
    // Notes length limit
    if fulfillment.notes.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fulfillment notes cannot exceed 1000 characters".to_string(),
        ));
    }

    // Gratitude message length limit
    if let Some(ref msg) = fulfillment.gratitude_message {
        if msg.len() > 500 {
            return Ok(ValidateCallbackResult::Invalid(
                "Gratitude message cannot exceed 500 characters".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::AgentToNeeds => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CategoryToNeeds => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CategoryToOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::NeedToMatches => Ok(ValidateCallbackResult::Valid),
        LinkTypes::OfferToMatches => Ok(ValidateCallbackResult::Valid),
        LinkTypes::MatchToFulfillment => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllNeeds => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::EmergencyNeeds => Ok(ValidateCallbackResult::Valid),
    }
}
