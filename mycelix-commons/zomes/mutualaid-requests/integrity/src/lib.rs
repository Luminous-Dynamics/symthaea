// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Requests Integrity Zome - Aid requests and offers for mutual aid coordination
//!
//! This zome defines the data structures and validation rules for aid requests
//! and offers within the Mycelix mutual aid network.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
    if did.trim().is_empty() {
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
    if id.trim().is_empty() {
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

    // DID length limit
    if request.requester_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester DID exceeds 256 character limit".into(),
        ));
    }

    // Validate ID
    validate_id(&request.id, "Request ID")?;

    // ID length limit
    if request.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID exceeds 256 character limit".into(),
        ));
    }

    // Validate description is not empty
    if request.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            RequestsError::EmptyDescription.to_string(),
        ));
    }

    // Description length limit
    if request.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description exceeds 4096 character limit".into(),
        ));
    }

    // Location length limit
    if let Some(ref loc) = request.location {
        if loc.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Location exceeds 256 character limit".into(),
            ));
        }
    }

    // RequestType::Other variant length limit
    if let RequestType::Other(ref s) = request.request_type {
        if s.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom request type cannot be empty".into(),
            ));
        }
        if s.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom request type exceeds 128 character limit".into(),
            ));
        }
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

    // DID length limit
    if offer.offerer_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offerer DID exceeds 256 character limit".into(),
        ));
    }

    // Validate IDs
    validate_id(&offer.id, "Offer ID")?;
    validate_id(&offer.request_id, "Request ID")?;

    // ID length limits
    if offer.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer ID exceeds 256 character limit".into(),
        ));
    }
    if offer.request_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID exceeds 256 character limit".into(),
        ));
    }

    // Validate offer amount is non-zero when present
    if let Some(amount) = offer.amount {
        if amount == 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Offer amount must be greater than zero".into(),
            ));
        }
    }

    // Validate message is not excessively long
    if offer.message.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer message must be 4096 characters or fewer".into(),
        ));
    }

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
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::AnchorToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AnchorToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToRequest => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToRequest link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::StatusToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "StatusToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::UrgencyToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "UrgencyToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RequestToOffer => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequestToOffer link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RequesterToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequesterToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OffererToOffer => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OffererToOffer link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Factory functions for valid test data

    fn valid_did() -> String {
        "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK".to_string()
    }

    fn valid_did_2() -> String {
        "did:web:example.com:users:alice".to_string()
    }

    fn valid_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    fn valid_aid_request() -> AidRequest {
        AidRequest {
            id: "req-123".to_string(),
            requester_did: valid_did(),
            request_type: RequestType::Financial,
            description: "Need help with rent payment".to_string(),
            urgency: Urgency::High,
            location: Some("Portland, OR".to_string()),
            amount_needed: Some(1000_00), // $1000.00
            fulfilled_amount: 0,
            status: RequestStatus::Open,
            created_at: valid_timestamp(),
            updated_at: valid_timestamp(),
        }
    }

    fn valid_aid_offer() -> AidOffer {
        AidOffer {
            id: "offer-456".to_string(),
            request_id: "req-123".to_string(),
            offerer_did: valid_did_2(),
            amount: Some(500_00), // $500.00
            message: "Happy to help!".to_string(),
            status: OfferStatus::Pending,
            created_at: valid_timestamp(),
            updated_at: valid_timestamp(),
        }
    }

    // DID Validation Tests

    #[test]
    fn test_validate_did_empty() {
        let result = validate_did("");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("DID cannot be empty"));
    }

    #[test]
    fn test_validate_did_no_prefix() {
        let result = validate_did("key:z6Mkh123");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Invalid DID format"));
    }

    #[test]
    fn test_validate_did_only_prefix() {
        let result = validate_did("did:");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Invalid DID format"));
    }

    #[test]
    fn test_validate_did_two_parts() {
        let result = validate_did("did:method");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Invalid DID format"));
    }

    #[test]
    fn test_validate_did_valid_key() {
        let result = validate_did("did:key:abc");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_did_valid_web() {
        let result = validate_did("did:web:example.com");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_did_valid_long_identifier() {
        let result = validate_did(&valid_did());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_did_multiple_colons() {
        let result = validate_did("did:web:example.com:users:alice:profile");
        assert!(result.is_ok());
    }

    // ID Validation Tests

    #[test]
    fn test_validate_id_empty() {
        let result = validate_id("", "Test ID");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Test ID cannot be empty"));
    }

    #[test]
    fn test_validate_id_valid() {
        let result = validate_id("req-123", "Request ID");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_id_with_special_chars() {
        let result = validate_id("req-123_abc-XYZ", "Request ID");
        assert!(result.is_ok());
    }

    // AidRequest Validation Tests

    #[test]
    fn test_validate_aid_request_valid() {
        let request = valid_aid_request();
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_aid_request_invalid_requester_did_empty() {
        let mut request = valid_aid_request();
        request.requester_did = "".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_request_invalid_requester_did_format() {
        let mut request = valid_aid_request();
        request.requester_did = "not-a-did".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_request_invalid_id_empty() {
        let mut request = valid_aid_request();
        request.id = "".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_request_empty_description() {
        let mut request = valid_aid_request();
        request.description = "".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(RequestsError::EmptyDescription.to_string())
        );
    }

    #[test]
    fn test_validate_aid_request_whitespace_only_description() {
        let mut request = valid_aid_request();
        request.description = "   \n\t  ".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(RequestsError::EmptyDescription.to_string())
        );
    }

    #[test]
    fn test_validate_aid_request_fulfilled_exceeds_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(1000);
        request.fulfilled_amount = 1500;
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(RequestsError::FulfilledExceedsNeeded.to_string())
        );
    }

    #[test]
    fn test_validate_aid_request_fulfilled_equals_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(1000);
        request.fulfilled_amount = 1000;
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_aid_request_fulfilled_less_than_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(1000);
        request.fulfilled_amount = 500;
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_aid_request_no_amount_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = None;
        request.fulfilled_amount = 500; // Should not cause error
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_aid_request_zero_amounts() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(0);
        request.fulfilled_amount = 0;
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // AidOffer Validation Tests

    #[test]
    fn test_validate_aid_offer_valid() {
        let offer = valid_aid_offer();
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_aid_offer_invalid_offerer_did_empty() {
        let mut offer = valid_aid_offer();
        offer.offerer_did = "".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_offer_invalid_offerer_did_format() {
        let mut offer = valid_aid_offer();
        offer.offerer_did = "invalid-did-format".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_offer_empty_offer_id() {
        let mut offer = valid_aid_offer();
        offer.id = "".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_offer_empty_request_id() {
        let mut offer = valid_aid_offer();
        offer.request_id = "".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_aid_offer_no_amount() {
        let mut offer = valid_aid_offer();
        offer.amount = None;
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_aid_offer_zero_amount() {
        let mut offer = valid_aid_offer();
        offer.amount = Some(0);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        // Zero amount is now rejected as part of hardening
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // Enum Serde Tests

    #[test]
    fn test_request_type_serde_financial() {
        let rt = RequestType::Financial;
        let json = serde_json::to_string(&rt).unwrap();
        assert_eq!(json, r#""financial""#);
        let deserialized: RequestType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, rt);
    }

    #[test]
    fn test_request_type_serde_housing() {
        let rt = RequestType::Housing;
        let json = serde_json::to_string(&rt).unwrap();
        assert_eq!(json, r#""housing""#);
        let deserialized: RequestType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, rt);
    }

    #[test]
    fn test_request_type_serde_other() {
        let rt = RequestType::Other("Custom need".to_string());
        let json = serde_json::to_string(&rt).unwrap();
        assert!(json.contains("other"));
        let deserialized: RequestType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, rt);
    }

    #[test]
    fn test_urgency_serde_critical() {
        let urgency = Urgency::Critical;
        let json = serde_json::to_string(&urgency).unwrap();
        assert_eq!(json, r#""critical""#);
        let deserialized: Urgency = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, urgency);
    }

    #[test]
    fn test_urgency_serde_high() {
        let urgency = Urgency::High;
        let json = serde_json::to_string(&urgency).unwrap();
        assert_eq!(json, r#""high""#);
        let deserialized: Urgency = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, urgency);
    }

    #[test]
    fn test_urgency_serde_medium() {
        let urgency = Urgency::Medium;
        let json = serde_json::to_string(&urgency).unwrap();
        assert_eq!(json, r#""medium""#);
        let deserialized: Urgency = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, urgency);
    }

    #[test]
    fn test_urgency_serde_low() {
        let urgency = Urgency::Low;
        let json = serde_json::to_string(&urgency).unwrap();
        assert_eq!(json, r#""low""#);
        let deserialized: Urgency = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, urgency);
    }

    #[test]
    fn test_request_status_serde_open() {
        let status = RequestStatus::Open;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""open""#);
        let deserialized: RequestStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_request_status_serde_partially_fulfilled() {
        let status = RequestStatus::PartiallyFulfilled;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""partially_fulfilled""#);
        let deserialized: RequestStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_request_status_serde_fulfilled() {
        let status = RequestStatus::Fulfilled;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""fulfilled""#);
        let deserialized: RequestStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_request_status_serde_cancelled() {
        let status = RequestStatus::Cancelled;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""cancelled""#);
        let deserialized: RequestStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_offer_status_serde_pending() {
        let status = OfferStatus::Pending;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""pending""#);
        let deserialized: OfferStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_offer_status_serde_accepted() {
        let status = OfferStatus::Accepted;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""accepted""#);
        let deserialized: OfferStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_offer_status_serde_completed() {
        let status = OfferStatus::Completed;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""completed""#);
        let deserialized: OfferStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_offer_status_serde_withdrawn() {
        let status = OfferStatus::Withdrawn;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, r#""withdrawn""#);
        let deserialized: OfferStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    // Anchor Tests

    #[test]
    fn test_anchor_new_string() {
        let anchor = Anchor::new("test-anchor");
        assert_eq!(anchor.0, "test-anchor");
    }

    #[test]
    fn test_anchor_new_owned_string() {
        let anchor = Anchor::new("owned".to_string());
        assert_eq!(anchor.0, "owned");
    }

    #[test]
    fn test_anchor_clone() {
        let anchor1 = Anchor::new("clone-test");
        let anchor2 = anchor1.clone();
        assert_eq!(anchor1, anchor2);
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor::new("same");
        let anchor2 = Anchor::new("same");
        assert_eq!(anchor1, anchor2);
    }

    #[test]
    fn test_anchor_inequality() {
        let anchor1 = Anchor::new("different1");
        let anchor2 = Anchor::new("different2");
        assert_ne!(anchor1, anchor2);
    }

    // RequestsError Display Tests

    #[test]
    fn test_error_display_invalid_did() {
        let error = RequestsError::InvalidDid("bad-did".to_string());
        assert_eq!(error.to_string(), "Invalid DID format: bad-did");
    }

    #[test]
    fn test_error_display_invalid_id() {
        let error = RequestsError::InvalidId("bad-id".to_string());
        assert_eq!(error.to_string(), "Invalid ID format: bad-id");
    }

    #[test]
    fn test_error_display_negative_amount() {
        let error = RequestsError::NegativeAmount;
        assert_eq!(error.to_string(), "Amount cannot be negative");
    }

    #[test]
    fn test_error_display_fulfilled_exceeds_needed() {
        let error = RequestsError::FulfilledExceedsNeeded;
        assert_eq!(error.to_string(), "Fulfilled amount exceeds needed amount");
    }

    #[test]
    fn test_error_display_empty_description() {
        let error = RequestsError::EmptyDescription;
        assert_eq!(error.to_string(), "Description cannot be empty");
    }

    #[test]
    fn test_error_display_empty_request_id() {
        let error = RequestsError::EmptyRequestId;
        assert_eq!(error.to_string(), "Request ID cannot be empty");
    }

    // Edge Case Tests

    #[test]
    fn test_aid_request_with_all_request_types() {
        let types = vec![
            RequestType::Financial,
            RequestType::Housing,
            RequestType::Food,
            RequestType::Medical,
            RequestType::Childcare,
            RequestType::Transportation,
            RequestType::Legal,
            RequestType::Other("Custom".to_string()),
        ];

        for request_type in types {
            let mut request = valid_aid_request();
            request.request_type = request_type;
            let result = validate_aid_request(&request);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_aid_request_with_all_urgencies() {
        let urgencies = vec![
            Urgency::Critical,
            Urgency::High,
            Urgency::Medium,
            Urgency::Low,
        ];

        for urgency in urgencies {
            let mut request = valid_aid_request();
            request.urgency = urgency;
            let result = validate_aid_request(&request);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_aid_request_with_all_statuses() {
        let statuses = vec![
            RequestStatus::Open,
            RequestStatus::PartiallyFulfilled,
            RequestStatus::Fulfilled,
            RequestStatus::Cancelled,
        ];

        for status in statuses {
            let mut request = valid_aid_request();
            request.status = status;
            let result = validate_aid_request(&request);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_aid_offer_with_all_statuses() {
        let statuses = vec![
            OfferStatus::Pending,
            OfferStatus::Accepted,
            OfferStatus::Completed,
            OfferStatus::Withdrawn,
        ];

        for status in statuses {
            let mut offer = valid_aid_offer();
            offer.status = status;
            let result = validate_aid_offer(&offer);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_aid_request_no_location() {
        let mut request = valid_aid_request();
        request.location = None;
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_large_amounts() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(u64::MAX);
        request.fulfilled_amount = u64::MAX - 1;
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_offer_empty_message() {
        let mut offer = valid_aid_offer();
        offer.message = "".to_string();
        let result = validate_aid_offer(&offer);
        // Message is allowed to be empty
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_very_long_description_rejected() {
        let mut request = valid_aid_request();
        request.description = "a".repeat(10000);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_aid_request_unicode_description() {
        let mut request = valid_aid_request();
        request.description = "需要帮助 🙏 Помощь необходима".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ── Link tag validation tests ──────────────────────────────────────

    fn check_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::TypeToRequest => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToRequest link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            _ => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_anchor_to_request_tag_at_limit() {
        let result = check_link_tag(LinkTypes::AnchorToRequest, vec![0u8; 256]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_anchor_to_request_tag_too_long() {
        let result = check_link_tag(LinkTypes::AnchorToRequest, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_type_to_request_tag_at_limit() {
        let result = check_link_tag(LinkTypes::TypeToRequest, vec![0u8; 512]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_type_to_request_tag_too_long() {
        let result = check_link_tag(LinkTypes::TypeToRequest, vec![0u8; 513]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_status_to_request_tag_at_limit() {
        let result = check_link_tag(LinkTypes::StatusToRequest, vec![0u8; 256]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_status_to_request_tag_too_long() {
        let result = check_link_tag(LinkTypes::StatusToRequest, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_urgency_to_request_tag_at_limit() {
        let result = check_link_tag(LinkTypes::UrgencyToRequest, vec![0u8; 256]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_urgency_to_request_tag_too_long() {
        let result = check_link_tag(LinkTypes::UrgencyToRequest, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_request_to_offer_tag_at_limit() {
        let result = check_link_tag(LinkTypes::RequestToOffer, vec![0u8; 256]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_request_to_offer_tag_too_long() {
        let result = check_link_tag(LinkTypes::RequestToOffer, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_requester_to_request_tag_at_limit() {
        let result = check_link_tag(LinkTypes::RequesterToRequest, vec![0u8; 256]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_requester_to_request_tag_too_long() {
        let result = check_link_tag(LinkTypes::RequesterToRequest, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_offerer_to_offer_tag_at_limit() {
        let result = check_link_tag(LinkTypes::OffererToOffer, vec![0u8; 256]);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_offerer_to_offer_tag_too_long() {
        let result = check_link_tag(LinkTypes::OffererToOffer, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ========================================================================
    // HARDENING: Byzantine & Edge Case Tests
    // ========================================================================

    // ── Fulfilled amount edge cases ─────────────────────────────────────

    #[test]
    fn test_aid_request_fulfilled_just_below_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(1000);
        request.fulfilled_amount = 999;
        let result = validate_aid_request(&request);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_fulfilled_just_above_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(1000);
        request.fulfilled_amount = 1001;
        let result = validate_aid_request(&request);
        assert_eq!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(RequestsError::FulfilledExceedsNeeded.to_string())
        );
    }

    #[test]
    fn test_aid_request_fulfilled_max_u64_no_needed() {
        let mut request = valid_aid_request();
        request.amount_needed = None;
        request.fulfilled_amount = u64::MAX;
        let result = validate_aid_request(&request);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_fulfilled_0_needed_0() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(0);
        request.fulfilled_amount = 0;
        let result = validate_aid_request(&request);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_fulfilled_1_needed_0_rejected() {
        let mut request = valid_aid_request();
        request.amount_needed = Some(0);
        request.fulfilled_amount = 1;
        let result = validate_aid_request(&request);
        assert_eq!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(RequestsError::FulfilledExceedsNeeded.to_string())
        );
    }

    // ── Offer amount edge cases ─────────────────────────────────────────

    #[test]
    fn test_aid_offer_zero_amount_rejected() {
        let mut offer = valid_aid_offer();
        offer.amount = Some(0);
        let result = validate_aid_offer(&offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
        if let Ok(ValidateCallbackResult::Invalid(msg)) = result {
            assert!(
                msg.contains("Offer amount must be greater than zero"),
                "Got: {}",
                msg
            );
        }
    }

    #[test]
    fn test_aid_offer_amount_1_ok() {
        let mut offer = valid_aid_offer();
        offer.amount = Some(1);
        let result = validate_aid_offer(&offer);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_offer_max_amount_ok() {
        let mut offer = valid_aid_offer();
        offer.amount = Some(u64::MAX);
        let result = validate_aid_offer(&offer);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_offer_no_amount_ok() {
        let mut offer = valid_aid_offer();
        offer.amount = None;
        let result = validate_aid_offer(&offer);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ── Offer message length edge cases ─────────────────────────────────

    #[test]
    fn test_aid_offer_message_at_4096_ok() {
        let mut offer = valid_aid_offer();
        offer.message = "m".repeat(4096);
        let result = validate_aid_offer(&offer);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_offer_message_4097_rejected() {
        let mut offer = valid_aid_offer();
        offer.message = "m".repeat(4097);
        let result = validate_aid_offer(&offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
        if let Ok(ValidateCallbackResult::Invalid(msg)) = result {
            assert!(msg.contains("4096 characters"), "Got: {}", msg);
        }
    }

    // ── DID edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_aid_request_whitespace_did_rejected() {
        let mut request = valid_aid_request();
        request.requester_did = "   ".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_aid_offer_whitespace_did_rejected() {
        let mut offer = valid_aid_offer();
        offer.offerer_did = "  \t  ".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    #[test]
    fn test_aid_request_did_with_spaces_in_middle() {
        // "did:my celix:alice" — spaces in method part
        let mut request = valid_aid_request();
        request.requester_did = "did:my celix:alice".to_string();
        // Basic format check passes (has did: prefix, 3+ parts)
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
    }

    // ── Description edge cases ──────────────────────────────────────────

    #[test]
    fn test_aid_request_description_single_char_ok() {
        let mut request = valid_aid_request();
        request.description = "x".to_string();
        let result = validate_aid_request(&request);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_description_tab_only_rejected() {
        let mut request = valid_aid_request();
        request.description = "\t".to_string();
        let result = validate_aid_request(&request);
        assert_eq!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(RequestsError::EmptyDescription.to_string())
        );
    }

    // ── ID edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_aid_request_whitespace_id_rejected() {
        let mut request = valid_aid_request();
        request.id = "   ".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_aid_offer_whitespace_offer_id_rejected() {
        let mut offer = valid_aid_offer();
        offer.id = " \t ".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    #[test]
    fn test_aid_offer_whitespace_request_id_rejected() {
        let mut offer = valid_aid_offer();
        offer.request_id = "  ".to_string();
        let result = validate_aid_offer(&offer);
        assert!(result.is_err());
    }

    // ========================================================================
    // HARDENING: String length limit boundary tests
    // ========================================================================

    // ── AidRequest ID (max 64) ─────────────────────────────────────────

    #[test]
    fn test_validate_request_id_at_limit() {
        let mut request = valid_aid_request();
        request.id = "a".repeat(64);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_request_id_too_long() {
        let mut request = valid_aid_request();
        request.id = "a".repeat(257);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidRequest requester_did (max 256) ─────────────────────────────

    #[test]
    fn test_validate_request_requester_did_at_limit() {
        let mut request = valid_aid_request();
        // Build a valid DID that's exactly 256 chars: "did:key:" + padding
        let prefix = "did:key:";
        let padding = "x".repeat(256 - prefix.len());
        request.requester_did = format!("{}{}", prefix, padding);
        assert_eq!(request.requester_did.len(), 256);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_request_requester_did_too_long() {
        let mut request = valid_aid_request();
        let prefix = "did:key:";
        let padding = "x".repeat(257 - prefix.len());
        request.requester_did = format!("{}{}", prefix, padding);
        assert_eq!(request.requester_did.len(), 257);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidRequest description (max 4096) ──────────────────────────────

    #[test]
    fn test_validate_request_description_at_limit() {
        let mut request = valid_aid_request();
        request.description = "d".repeat(4096);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_request_description_too_long() {
        let mut request = valid_aid_request();
        request.description = "d".repeat(4097);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidRequest location (max 256) ──────────────────────────────────

    #[test]
    fn test_validate_request_location_at_limit() {
        let mut request = valid_aid_request();
        request.location = Some("L".repeat(256));
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_request_location_too_long() {
        let mut request = valid_aid_request();
        request.location = Some("L".repeat(257));
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidRequest RequestType::Other (max 128) ────────────────────────

    #[test]
    fn test_validate_request_custom_type_at_limit() {
        let mut request = valid_aid_request();
        request.request_type = RequestType::Other("c".repeat(128));
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_request_custom_type_too_long() {
        let mut request = valid_aid_request();
        request.request_type = RequestType::Other("c".repeat(129));
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_validate_request_custom_type_empty() {
        let mut request = valid_aid_request();
        request.request_type = RequestType::Other("".to_string());
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidOffer ID (max 64) ───────────────────────────────────────────

    #[test]
    fn test_validate_offer_id_at_limit() {
        let mut offer = valid_aid_offer();
        offer.id = "a".repeat(64);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_offer_id_too_long() {
        let mut offer = valid_aid_offer();
        offer.id = "a".repeat(257);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidOffer request_id (max 64) ───────────────────────────────────

    #[test]
    fn test_validate_offer_request_id_at_limit() {
        let mut offer = valid_aid_offer();
        offer.request_id = "r".repeat(64);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_offer_request_id_too_long() {
        let mut offer = valid_aid_offer();
        offer.request_id = "r".repeat(257);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ── AidOffer offerer_did (max 256) ─────────────────────────────────

    #[test]
    fn test_validate_offer_offerer_did_at_limit() {
        let mut offer = valid_aid_offer();
        let prefix = "did:key:";
        let padding = "x".repeat(256 - prefix.len());
        offer.offerer_did = format!("{}{}", prefix, padding);
        assert_eq!(offer.offerer_did.len(), 256);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_offer_offerer_did_too_long() {
        let mut offer = valid_aid_offer();
        let prefix = "did:key:";
        let padding = "x".repeat(257 - prefix.len());
        offer.offerer_did = format!("{}{}", prefix, padding);
        assert_eq!(offer.offerer_did.len(), 257);
        let result = validate_aid_offer(&offer);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }
}
