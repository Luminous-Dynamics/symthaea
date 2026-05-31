// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Timebank Integrity Zome
//! Defines entry types and validation for service offers, requests, exchanges, and time credits.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Categories of care services available in the timebank
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ServiceCategory {
    Childcare,
    Eldercare,
    PetCare,
    Cooking,
    Cleaning,
    Gardening,
    Tutoring,
    TechSupport,
    Transportation,
    Companionship,
    HealthSupport,
    HomeRepair,
    LegalAdvice,
    Counseling,
    ArtMusic,
    LanguageHelp,
    Administrative,
    Other(String),
}

impl ServiceCategory {
    /// Return a canonical string key for anchor-based discovery
    pub fn anchor_key(&self) -> String {
        match self {
            ServiceCategory::Childcare => "childcare".to_string(),
            ServiceCategory::Eldercare => "eldercare".to_string(),
            ServiceCategory::PetCare => "petcare".to_string(),
            ServiceCategory::Cooking => "cooking".to_string(),
            ServiceCategory::Cleaning => "cleaning".to_string(),
            ServiceCategory::Gardening => "gardening".to_string(),
            ServiceCategory::Tutoring => "tutoring".to_string(),
            ServiceCategory::TechSupport => "techsupport".to_string(),
            ServiceCategory::Transportation => "transportation".to_string(),
            ServiceCategory::Companionship => "companionship".to_string(),
            ServiceCategory::HealthSupport => "healthsupport".to_string(),
            ServiceCategory::HomeRepair => "homerepair".to_string(),
            ServiceCategory::LegalAdvice => "legaladvice".to_string(),
            ServiceCategory::Counseling => "counseling".to_string(),
            ServiceCategory::ArtMusic => "artmusic".to_string(),
            ServiceCategory::LanguageHelp => "languagehelp".to_string(),
            ServiceCategory::Administrative => "administrative".to_string(),
            ServiceCategory::Other(s) => format!("other_{}", s.to_lowercase().replace(' ', "_")),
        }
    }
}

/// Urgency level for service requests
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// A service offer posted by a provider
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceOffer {
    /// The agent offering the service
    pub provider: AgentPubKey,
    /// Category of service
    pub category: ServiceCategory,
    /// Short title for the offer
    pub title: String,
    /// Detailed description of what is offered
    pub description: String,
    /// Maximum hours available per week
    pub hours_available: f32,
    /// Availability description (e.g. "weekday mornings", "flexible")
    pub availability: String,
    /// Location or area served
    pub location: String,
    /// Skills or qualifications relevant to this offer
    pub skills_required: Vec<String>,
    /// Whether this offer is currently active
    pub active: bool,
    /// When the offer was created
    pub created_at: Timestamp,
    /// When the offer was last updated
    pub updated_at: Timestamp,
}

/// A service request posted by someone needing help
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceRequest {
    /// The agent requesting help
    pub requester: AgentPubKey,
    /// Category of service needed
    pub category: ServiceCategory,
    /// Short title for the request
    pub title: String,
    /// Detailed description of what is needed
    pub description: String,
    /// Estimated hours needed
    pub hours_needed: f32,
    /// Preferred schedule
    pub preferred_schedule: String,
    /// Location where service is needed
    pub location: String,
    /// How urgent the request is
    pub urgency: UrgencyLevel,
    /// Whether this request is still open
    pub open: bool,
    /// When the request was created
    pub created_at: Timestamp,
    /// When the request was last updated
    pub updated_at: Timestamp,
}

/// A completed exchange of service between two agents
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TimeExchange {
    /// The service offer that was fulfilled
    pub offer_id: ActionHash,
    /// The service request that was fulfilled
    pub request_id: ActionHash,
    /// The provider who gave the service
    pub provider: AgentPubKey,
    /// The recipient who received the service
    pub recipient: AgentPubKey,
    /// Number of hours exchanged
    pub hours: f32,
    /// Category of service exchanged
    pub category: ServiceCategory,
    /// When the exchange was completed
    pub completed_at: Timestamp,
    /// Provider's rating of the recipient (1-5)
    pub rating_provider: Option<u8>,
    /// Recipient's rating of the provider (1-5)
    pub rating_recipient: Option<u8>,
    /// Optional notes about the exchange
    pub notes: String,
}

/// Time credit balance for an agent
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TimeCredit {
    /// The agent this credit belongs to
    pub agent: AgentPubKey,
    /// Current balance in hours
    pub balance: f64,
    /// Total hours earned through providing services
    pub total_earned: f64,
    /// Total hours spent receiving services
    pub total_spent: f64,
    /// Last updated timestamp
    pub updated_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    ServiceOffer(ServiceOffer),
    ServiceRequest(ServiceRequest),
    TimeExchange(TimeExchange),
    TimeCredit(TimeCredit),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent to their service offers
    AgentToOffer,
    /// Agent to their service requests
    AgentToRequest,
    /// Category anchor to offers in that category
    CategoryToOffer,
    /// Category anchor to requests in that category
    CategoryToRequest,
    /// All active offers anchor
    AllActiveOffers,
    /// All open requests anchor
    AllOpenRequests,
    /// Agent to their completed exchanges
    AgentToExchange,
    /// Agent to their time credit record
    AgentToCredit,
    /// Offer to exchanges that fulfilled it
    OfferToExchange,
    /// Request to exchanges that fulfilled it
    RequestToExchange,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ServiceOffer(offer) => validate_create_offer(action, offer),
                EntryTypes::ServiceRequest(request) => validate_create_request(action, request),
                EntryTypes::TimeExchange(exchange) => validate_create_exchange(action, exchange),
                EntryTypes::TimeCredit(credit) => validate_create_credit(action, credit),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ServiceOffer(offer) => validate_update_offer(offer),
                EntryTypes::ServiceRequest(request) => validate_update_request(request),
                EntryTypes::TimeExchange(exchange) => validate_update_exchange(exchange),
                EntryTypes::TimeCredit(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AgentToOffer => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CategoryToOffer => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CategoryToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllActiveOffers => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllOpenRequests => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToExchange => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToCredit => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OfferToExchange => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RequestToExchange => Ok(ValidateCallbackResult::Valid),
        },
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

fn validate_create_offer(
    _action: Create,
    offer: ServiceOffer,
) -> ExternResult<ValidateCallbackResult> {
    if offer.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title cannot be empty".into(),
        ));
    }
    if offer.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title must be 256 characters or fewer".into(),
        ));
    }
    if offer.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer description cannot be empty".into(),
        ));
    }
    if offer.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer description must be 4096 characters or fewer".into(),
        ));
    }
    if offer.hours_available <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours available must be positive".into(),
        ));
    }
    if offer.hours_available > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours available cannot exceed 168 per week".into(),
        ));
    }
    if offer.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Location cannot be empty".into(),
        ));
    }
    if offer.location.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Location must be 512 characters or fewer".into(),
        ));
    }
    if offer.availability.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Availability must be 512 characters or fewer".into(),
        ));
    }
    if offer.skills_required.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot list more than 20 skills".into(),
        ));
    }
    for skill in &offer.skills_required {
        if skill.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each skill must be 128 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_request(
    _action: Create,
    request: ServiceRequest,
) -> ExternResult<ValidateCallbackResult> {
    if request.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title cannot be empty".into(),
        ));
    }
    if request.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title must be 256 characters or fewer".into(),
        ));
    }
    if request.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request description cannot be empty".into(),
        ));
    }
    if request.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request description must be 4096 characters or fewer".into(),
        ));
    }
    if request.hours_needed <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours needed must be positive".into(),
        ));
    }
    if request.hours_needed > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours needed cannot exceed 168".into(),
        ));
    }
    if request.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Location cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_exchange(
    _action: Create,
    exchange: TimeExchange,
) -> ExternResult<ValidateCallbackResult> {
    if exchange.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange hours must be positive".into(),
        ));
    }
    if exchange.hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange hours cannot exceed 168".into(),
        ));
    }
    if exchange.provider == exchange.recipient {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider and recipient cannot be the same agent".into(),
        ));
    }
    if let Some(rating) = exchange.rating_provider {
        if rating < 1 || rating > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Provider rating must be 1-5".into(),
            ));
        }
    }
    if let Some(rating) = exchange.rating_recipient {
        if rating < 1 || rating > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Recipient rating must be 1-5".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_credit(
    _action: Create,
    credit: TimeCredit,
) -> ExternResult<ValidateCallbackResult> {
    if credit.total_earned < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total earned cannot be negative".into(),
        ));
    }
    if credit.total_spent < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total spent cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_offer(offer: ServiceOffer) -> ExternResult<ValidateCallbackResult> {
    if offer.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title cannot be empty".into(),
        ));
    }
    if offer.hours_available < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours available cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_request(request: ServiceRequest) -> ExternResult<ValidateCallbackResult> {
    if request.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request title cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_exchange(exchange: TimeExchange) -> ExternResult<ValidateCallbackResult> {
    if let Some(rating) = exchange.rating_provider {
        if rating < 1 || rating > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Provider rating must be 1-5".into(),
            ));
        }
    }
    if let Some(rating) = exchange.rating_recipient {
        if rating < 1 || rating > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Recipient rating must be 1-5".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
