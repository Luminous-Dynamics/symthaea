// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Timebank Integrity Zome
//!
//! This zome defines the entry types and validation rules for time banking
//! in the Mycelix Mutual Aid hApp. It implements the core principle:
//! 1 hour = 1 hour, regardless of service type.

use hdi::prelude::*;
use mutualaid_common::*;

/// Entry types for the timebank zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A service offer from a member
    #[entry_type(visibility = "public")]
    ServiceOffer(ServiceOffer),
    /// A service request from a member
    #[entry_type(visibility = "public")]
    ServiceRequest(ServiceRequest),
    /// A completed time exchange
    #[entry_type(visibility = "public")]
    TimeExchange(TimeExchange),
    /// Time credit record
    #[entry_type(visibility = "public")]
    TimeCredit(TimeCredit),
}

/// Link types for the timebank zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from agent to their service offers
    AgentToOffers,
    /// Link from agent to their service requests
    AgentToRequests,
    /// Link from agent to exchanges they participated in
    AgentToExchanges,
    /// Link from category anchor to offers
    CategoryToOffers,
    /// Link from category anchor to requests
    CategoryToRequests,
    /// Link from offer to exchange
    OfferToExchange,
    /// Link from request to exchange
    RequestToExchange,
    /// Link for all offers discovery
    AllOffers,
    /// Link for all requests discovery
    AllRequests,
    /// Link from agent to their time credits
    AgentToCredits,
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
        EntryTypes::ServiceOffer(offer) => validate_service_offer(offer),
        EntryTypes::ServiceRequest(request) => validate_service_request(request),
        EntryTypes::TimeExchange(exchange) => validate_time_exchange(exchange),
        EntryTypes::TimeCredit(credit) => validate_time_credit(credit),
    }
}

/// Validate a service offer
fn validate_service_offer(offer: ServiceOffer) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if offer.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer ID cannot be empty".to_string(),
        ));
    }

    // Title must not be empty
    if offer.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if offer.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if offer.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer description cannot exceed 5000 characters".to_string(),
        ));
    }

    // Minimum duration must be positive
    if offer.min_duration_hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum duration must be positive".to_string(),
        ));
    }

    // Max duration must be >= min if specified
    if let Some(max) = offer.max_duration_hours {
        if max < offer.min_duration_hours {
            return Ok(ValidateCallbackResult::Invalid(
                "Maximum duration cannot be less than minimum".to_string(),
            ));
        }
    }

    // Qualifications limit
    if offer.qualifications.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many qualifications (max 20)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a service request
fn validate_service_request(request: ServiceRequest) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if request.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request ID cannot be empty".to_string(),
        ));
    }

    // Title must not be empty
    if request.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if request.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if request.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request description cannot exceed 5000 characters".to_string(),
        ));
    }

    // Estimated hours must be positive
    if request.estimated_hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Estimated hours must be positive".to_string(),
        ));
    }

    // Estimated hours should be reasonable (max 168 = 1 week)
    if request.estimated_hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Estimated hours cannot exceed 168 (one week)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a time exchange
fn validate_time_exchange(exchange: TimeExchange) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if exchange.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Time exchange ID cannot be empty".to_string(),
        ));
    }

    // Hours must be positive
    if exchange.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange hours must be positive".to_string(),
        ));
    }

    // Hours should be reasonable (max 168 = 1 week)
    if exchange.hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange hours cannot exceed 168 (one week)".to_string(),
        ));
    }

    // Provider and recipient must be different
    if exchange.provider == exchange.recipient {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider and recipient must be different agents".to_string(),
        ));
    }

    // Description must not be empty
    if exchange.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange description cannot be empty".to_string(),
        ));
    }

    // Validate ratings if present
    if let Some(rating) = &exchange.provider_rating {
        if rating.score < 1 || rating.score > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating score must be between 1 and 5".to_string(),
            ));
        }
    }

    if let Some(rating) = &exchange.recipient_rating {
        if rating.score < 1 || rating.score > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating score must be between 1 and 5".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a time credit
fn validate_time_credit(credit: TimeCredit) -> ExternResult<ValidateCallbackResult> {
    // Hours must be positive
    if credit.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit hours must be positive".to_string(),
        ));
    }

    // Hours should be reasonable
    if credit.hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit hours cannot exceed 168 (one week)".to_string(),
        ));
    }

    // Earner and debtor must be different
    if credit.earner == credit.debtor {
        return Ok(ValidateCallbackResult::Invalid(
            "Earner and debtor must be different agents".to_string(),
        ));
    }

    // Description must not be empty
    if credit.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit description cannot be empty".to_string(),
        ));
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
        LinkTypes::AgentToOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToRequests => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToExchanges => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CategoryToOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CategoryToRequests => Ok(ValidateCallbackResult::Valid),
        LinkTypes::OfferToExchange => Ok(ValidateCallbackResult::Valid),
        LinkTypes::RequestToExchange => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllRequests => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToCredits => Ok(ValidateCallbackResult::Valid),
    }
}
