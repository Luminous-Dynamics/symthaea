//! Requests Integrity Zome - Aid requests and offers for mutual aid coordination
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
            RequestsError::EmptyDescription.to_string()
        ));
    }

    // Validate fulfilled amount doesn't exceed needed amount
    if let Some(needed) = request.amount_needed {
        if request.fulfilled_amount > needed {
            return Ok(ValidateCallbackResult::Invalid(
                RequestsError::FulfilledExceedsNeeded.to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use hdi::prelude::*;

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
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
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
    fn test_aid_request_very_long_description() {
        let mut request = valid_aid_request();
        request.description = "a".repeat(10000);
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_aid_request_unicode_description() {
        let mut request = valid_aid_request();
        request.description = "需要帮助 🙏 Помощь необходима".to_string();
        let result = validate_aid_request(&request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }
}
