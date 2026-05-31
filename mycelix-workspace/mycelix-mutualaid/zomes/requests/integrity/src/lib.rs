// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Requests Integrity Zome - Aid requests and offers for mutual aid coordination
//!
//! This zome defines the data structures and validation rules for aid requests
//! and offers within the Mycelix mutual aid network.

use hdi::prelude::*;

/// Anchor entry type for string-based link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Anchor(pub String);

impl Anchor {
    pub fn new(value: impl Into<String>) -> Self {
        Anchor(value.into())
    }
}

/// Type of aid being requested
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestType {
    Financial,
    Housing,
    Food,
    Medical,
    Childcare,
    Transportation,
    Legal,
    Other(String),
}

/// Urgency level for aid requests
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Urgency {
    Critical,
    High,
    Medium,
    Low,
}

/// Status of an aid request
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Open,
    PartiallyFulfilled,
    Fulfilled,
    Cancelled,
}

/// Status of an aid offer
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferStatus {
    Pending,
    Accepted,
    Completed,
    Withdrawn,
}

/// An aid request from a community member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AidRequest {
    /// Unique identifier for this request
    pub id: String,
    /// DID of the person requesting aid
    pub requester_did: String,
    /// Type of aid being requested
    pub request_type: RequestType,
    /// Detailed description of the need
    pub description: String,
    /// Urgency level
    pub urgency: Urgency,
    /// Optional location (for local aid)
    pub location: Option<String>,
    /// Amount needed (if applicable, in smallest currency unit)
    pub amount_needed: Option<u64>,
    /// Amount already fulfilled
    pub fulfilled_amount: u64,
    /// Current status of the request
    pub status: RequestStatus,
    /// Timestamp when request was created
    pub created_at: Timestamp,
    /// Timestamp when request was last updated
    pub updated_at: Timestamp,
}

/// An offer to fulfill an aid request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AidOffer {
    /// Unique identifier for this offer
    pub id: String,
    /// Reference to the aid request being fulfilled
    pub request_id: String,
    /// DID of the person offering aid
    pub offerer_did: String,
    /// Amount being offered (if applicable)
    pub amount: Option<u64>,
    /// Message from the offerer
    pub message: String,
    /// Current status of the offer
    pub status: OfferStatus,
    /// Timestamp when offer was created
    pub created_at: Timestamp,
    /// Timestamp when offer was last updated
    pub updated_at: Timestamp,
}

/// All entry types for this zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    AidRequest(AidRequest),
    #[entry_type(visibility = "public")]
    AidOffer(AidOffer),
}

/// Link types for connecting entries
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all requests
    AnchorToRequest,
    /// Anchor to requests by type
    TypeToRequest,
    /// Anchor to requests by status
    StatusToRequest,
    /// Anchor to requests by urgency
    UrgencyToRequest,
    /// Request to its offers
    RequestToOffer,
    /// Requester DID to their requests
    RequesterToRequest,
    /// Offerer DID to their offers
    OffererToOffer,
}

/// Validation errors for requests zome
#[derive(Debug)]
pub enum RequestsError {
    InvalidDid(String),
    InvalidId(String),
    NegativeAmount,
    FulfilledExceedsNeeded,
    EmptyDescription,
    EmptyRequestId,
}

impl std::fmt::Display for RequestsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDid(s) => write!(f, "Invalid DID format: {}", s),
            Self::InvalidId(s) => write!(f, "Invalid ID format: {}", s),
            Self::NegativeAmount => write!(f, "Amount cannot be negative"),
            Self::FulfilledExceedsNeeded => write!(f, "Fulfilled amount exceeds needed amount"),
            Self::EmptyDescription => write!(f, "Description cannot be empty"),
            Self::EmptyRequestId => write!(f, "Request ID cannot be empty"),
        }
    }
}

/// Validate that a DID has a valid format
fn validate_did(did: &str) -> ExternResult<()> {
    if did.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            RequestsError::InvalidDid("DID cannot be empty".to_string()).to_string()
        )));
    }
    // Basic DID format check: did:method:identifier
    if !did.starts_with("did:") || did.split(':').count() < 3 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            RequestsError::InvalidDid(format!("Invalid DID format: {}", did)).to_string()
        )));
    }
    Ok(())
}

/// Validate that an ID is non-empty
fn validate_id(id: &str, field_name: &str) -> ExternResult<()> {
    if id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            RequestsError::InvalidId(format!("{} cannot be empty", field_name)).to_string()
        )));
    }
    Ok(())
}

/// Validate an AidRequest entry
fn validate_aid_request(request: &AidRequest) -> ExternResult<ValidateCallbackResult> {
    // Validate requester DID
    validate_did(&request.requester_did)?;

    // Validate ID
    validate_id(&request.id, "Request ID")?;

    // Validate description is not empty
    if request.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            RequestsError::EmptyDescription.to_string(),
        ));
    }

    // Validate fulfilled amount doesn't exceed needed amount
    if let Some(needed) = request.amount_needed {
        if request.fulfilled_amount > needed {
            return Ok(ValidateCallbackResult::Invalid(
                RequestsError::FulfilledExceedsNeeded.to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate an AidOffer entry
fn validate_aid_offer(offer: &AidOffer) -> ExternResult<ValidateCallbackResult> {
    // Validate offerer DID
    validate_did(&offer.offerer_did)?;

    // Validate IDs
    validate_id(&offer.id, "Offer ID")?;
    validate_id(&offer.request_id, "Request ID")?;

    Ok(ValidateCallbackResult::Valid)
}

/// Genesis self-check callback
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::AidRequest(request) => validate_aid_request(&request),
                    EntryTypes::AidOffer(offer) => validate_aid_offer(&offer),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToRequest
            | LinkTypes::TypeToRequest
            | LinkTypes::StatusToRequest
            | LinkTypes::UrgencyToRequest
            | LinkTypes::RequestToOffer
            | LinkTypes::RequesterToRequest
            | LinkTypes::OffererToOffer => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToRequest
            | LinkTypes::TypeToRequest
            | LinkTypes::StatusToRequest
            | LinkTypes::UrgencyToRequest
            | LinkTypes::RequestToOffer
            | LinkTypes::RequesterToRequest
            | LinkTypes::OffererToOffer => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
