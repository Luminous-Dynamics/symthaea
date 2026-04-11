// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Requests Coordinator Zome - Aid request and offer management
//!
//! This zome provides the coordination logic for creating, updating, and
//! querying aid requests and offers in the mutual aid network.

use hdk::prelude::*;
use mutualaid_requests_integrity::{
    AidOffer, AidRequest, Anchor as RequestsAnchor, EntryTypes, LinkTypes, OfferStatus,
    RequestStatus, RequestType, Urgency,
};
use mycelix_bridge_common::{
    gate_civic, civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};

/// Input for creating a new aid request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateRequestInput {
    pub requester_did: String,
    pub request_type: RequestType,
    pub description: String,
    pub urgency: Urgency,
    pub location: Option<String>,
    pub amount_needed: Option<u64>,
}

/// Input for updating a request's status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateRequestStatusInput {
    pub request_hash: ActionHash,
    pub status: RequestStatus,
    pub fulfilled_amount: Option<u64>,
}

/// Input for creating a new aid offer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateOfferInput {
    pub request_hash: ActionHash,
    pub request_id: String,
    pub offerer_did: String,
    pub amount: Option<u64>,
    pub message: String,
}

/// Input for updating an offer's status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateOfferStatusInput {
    pub offer_hash: ActionHash,
    pub status: OfferStatus,
}

/// Input for querying requests
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryRequestsInput {
    pub request_type: Option<RequestType>,
    pub status: Option<RequestStatus>,
    pub urgency: Option<Urgency>,
    pub requester_did: Option<String>,
}

/// Output containing a request with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestWithHash {
    pub hash: ActionHash,
    pub request: AidRequest,
}

/// Output containing an offer with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OfferWithHash {
    pub hash: ActionHash,
    pub offer: AidOffer,
}

/// Generate a unique ID based on timestamp and agent
fn generate_id(prefix: &str) -> ExternResult<String> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;
    // Create unique ID from timestamp and agent pubkey truncated hash
    let agent_str = format!("{:?}", agent);
    let short_hash = &agent_str[agent_str.len().saturating_sub(8)..];
    Ok(format!(
        "{}_{:x}_{}",
        prefix,
        now.as_micros() as u64,
        short_hash
    ))
}

/// Get or create the main requests anchor
fn get_requests_anchor() -> ExternResult<EntryHash> {
    let anchor = RequestsAnchor::new("all_requests");
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a type-specific anchor
fn get_type_anchor(request_type: &RequestType) -> ExternResult<EntryHash> {
    let type_str = match request_type {
        RequestType::Financial => "financial",
        RequestType::Housing => "housing",
        RequestType::Food => "food",
        RequestType::Medical => "medical",
        RequestType::Childcare => "childcare",
        RequestType::Transportation => "transportation",
        RequestType::Legal => "legal",
        RequestType::Other(s) => s.as_str(),
    };
    let anchor = RequestsAnchor::new(format!("type_{}", type_str));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a status-specific anchor
fn get_status_anchor(status: &RequestStatus) -> ExternResult<EntryHash> {
    let status_str = match status {
        RequestStatus::Open => "open",
        RequestStatus::PartiallyFulfilled => "partially_fulfilled",
        RequestStatus::Fulfilled => "fulfilled",
        RequestStatus::Cancelled => "cancelled",
    };
    let anchor = RequestsAnchor::new(format!("status_{}", status_str));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create an urgency-specific anchor
fn get_urgency_anchor(urgency: &Urgency) -> ExternResult<EntryHash> {
    let urgency_str = match urgency {
        Urgency::Critical => "critical",
        Urgency::High => "high",
        Urgency::Medium => "medium",
        Urgency::Low => "low",
    };
    let anchor = RequestsAnchor::new(format!("urgency_{}", urgency_str));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a requester-specific anchor
fn get_requester_anchor(did: &str) -> ExternResult<EntryHash> {
    let anchor = RequestsAnchor::new(format!("requester_{}", did));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create an offerer-specific anchor
fn get_offerer_anchor(did: &str) -> ExternResult<EntryHash> {
    let anchor = RequestsAnchor::new(format!("offerer_{}", did));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Create a new aid request
#[hdk_extern]
pub fn create_request(input: CreateRequestInput) -> ExternResult<RequestWithHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_request")?;

    // Validate description is not empty or whitespace-only
    if input.description.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request description cannot be empty or whitespace-only".into()
        )));
    }

    // Validate requester DID is not empty or whitespace-only
    if input.requester_did.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Requester DID cannot be empty or whitespace-only".into()
        )));
    }

    // Validate amount_needed is non-zero when provided
    if let Some(amount) = input.amount_needed {
        if amount == 0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Request amount must be greater than zero".into()
            )));
        }
    }

    let now = sys_time()?;
    let id = generate_id("req")?;

    let request = AidRequest {
        id: id.clone(),
        requester_did: input.requester_did.clone(),
        request_type: input.request_type.clone(),
        description: input.description,
        urgency: input.urgency.clone(),
        location: input.location,
        amount_needed: input.amount_needed,
        fulfilled_amount: 0,
        status: RequestStatus::Open,
        created_at: now,
        updated_at: now,
    };

    // Create the entry
    let action_hash = create_entry(EntryTypes::AidRequest(request.clone()))?;

    // Link from main anchor
    let requests_anchor = get_requests_anchor()?;
    create_link(
        requests_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToRequest,
        id.as_bytes().to_vec(),
    )?;

    // Link from type anchor
    let type_anchor = get_type_anchor(&input.request_type)?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::TypeToRequest,
        (),
    )?;

    // Link from status anchor (Open)
    let status_anchor = get_status_anchor(&RequestStatus::Open)?;
    create_link(
        status_anchor,
        action_hash.clone(),
        LinkTypes::StatusToRequest,
        (),
    )?;

    // Link from urgency anchor
    let urgency_anchor = get_urgency_anchor(&input.urgency)?;
    create_link(
        urgency_anchor,
        action_hash.clone(),
        LinkTypes::UrgencyToRequest,
        (),
    )?;

    // Link from requester anchor
    let requester_anchor = get_requester_anchor(&input.requester_did)?;
    create_link(
        requester_anchor,
        action_hash.clone(),
        LinkTypes::RequesterToRequest,
        (),
    )?;

    Ok(RequestWithHash {
        hash: action_hash,
        request,
    })
}

/// Get a request by its action hash
#[hdk_extern]
pub fn get_request(action_hash: ActionHash) -> ExternResult<Option<RequestWithHash>> {
    match get(action_hash.clone(), GetOptions::default())? {
        Some(record) => {
            let request: AidRequest = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("No AidRequest found".to_string()))
                })?;
            Ok(Some(RequestWithHash {
                hash: action_hash,
                request,
            }))
        }
        None => Ok(None),
    }
}

/// Update a request's status
#[hdk_extern]
pub fn update_request_status(input: UpdateRequestStatusInput) -> ExternResult<RequestWithHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_request_status")?;
    // Get the current request
    let record = get(input.request_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Request not found".to_string())))?;

    let mut request: AidRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No AidRequest found".to_string())))?;

    let old_status = request.status.clone();

    // Update the request
    request.status = input.status.clone();
    if let Some(amount) = input.fulfilled_amount {
        request.fulfilled_amount = amount;
    }
    request.updated_at = sys_time()?;

    // Update the entry
    let new_hash = update_entry(
        input.request_hash.clone(),
        EntryTypes::AidRequest(request.clone()),
    )?;

    // Update status links if status changed
    if old_status != input.status {
        // Get and delete old status links
        let old_status_anchor = get_status_anchor(&old_status)?;
        let old_links = get_links(
            LinkQuery::try_new(old_status_anchor, LinkTypes::StatusToRequest)?,
            GetStrategy::default(),
        )?;
        for link in old_links {
            if link.target == input.request_hash.clone().into() {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }

        // Create new status link
        let new_status_anchor = get_status_anchor(&input.status)?;
        create_link(
            new_status_anchor,
            new_hash.clone(),
            LinkTypes::StatusToRequest,
            (),
        )?;
    }

    Ok(RequestWithHash {
        hash: new_hash,
        request,
    })
}

/// Get all open requests
#[hdk_extern]
pub fn get_open_requests(_: ()) -> ExternResult<Vec<RequestWithHash>> {
    let status_anchor = get_status_anchor(&RequestStatus::Open)?;
    let links = get_links(
        LinkQuery::try_new(status_anchor, LinkTypes::StatusToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(request_with_hash) = get_request(action_hash)? {
                requests.push(request_with_hash);
            }
        }
    }

    Ok(requests)
}

/// Get requests by type
#[hdk_extern]
pub fn get_requests_by_type(request_type: RequestType) -> ExternResult<Vec<RequestWithHash>> {
    let type_anchor = get_type_anchor(&request_type)?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::TypeToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(request_with_hash) = get_request(action_hash)? {
                requests.push(request_with_hash);
            }
        }
    }

    Ok(requests)
}

/// Get requests by urgency
#[hdk_extern]
pub fn get_requests_by_urgency(urgency: Urgency) -> ExternResult<Vec<RequestWithHash>> {
    let urgency_anchor = get_urgency_anchor(&urgency)?;
    let links = get_links(
        LinkQuery::try_new(urgency_anchor, LinkTypes::UrgencyToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(request_with_hash) = get_request(action_hash)? {
                requests.push(request_with_hash);
            }
        }
    }

    Ok(requests)
}

/// Get requests by requester DID
#[hdk_extern]
pub fn get_requests_by_requester(requester_did: String) -> ExternResult<Vec<RequestWithHash>> {
    let requester_anchor = get_requester_anchor(&requester_did)?;
    let links = get_links(
        LinkQuery::try_new(requester_anchor, LinkTypes::RequesterToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(request_with_hash) = get_request(action_hash)? {
                requests.push(request_with_hash);
            }
        }
    }

    Ok(requests)
}

/// Create a new aid offer
#[hdk_extern]
pub fn create_offer(input: CreateOfferInput) -> ExternResult<OfferWithHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_offer")?;

    // Validate message is not empty or whitespace-only
    if input.message.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Offer message cannot be empty or whitespace-only".into()
        )));
    }

    // Validate offerer DID is not empty or whitespace-only
    if input.offerer_did.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Offerer DID cannot be empty or whitespace-only".into()
        )));
    }

    // Validate amount is non-zero when provided
    if let Some(amount) = input.amount {
        if amount == 0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Offer amount must be greater than zero".into()
            )));
        }
    }

    let now = sys_time()?;
    let id = generate_id("off")?;

    let offer = AidOffer {
        id: id.clone(),
        request_id: input.request_id,
        offerer_did: input.offerer_did.clone(),
        amount: input.amount,
        message: input.message,
        status: OfferStatus::Pending,
        created_at: now,
        updated_at: now,
    };

    // Create the entry
    let action_hash = create_entry(EntryTypes::AidOffer(offer.clone()))?;

    // Link from request to offer
    create_link(
        input.request_hash,
        action_hash.clone(),
        LinkTypes::RequestToOffer,
        id.as_bytes().to_vec(),
    )?;

    // Link from offerer anchor
    let offerer_anchor = get_offerer_anchor(&input.offerer_did)?;
    create_link(
        offerer_anchor,
        action_hash.clone(),
        LinkTypes::OffererToOffer,
        (),
    )?;

    Ok(OfferWithHash {
        hash: action_hash,
        offer,
    })
}

/// Get an offer by its action hash
#[hdk_extern]
pub fn get_offer(action_hash: ActionHash) -> ExternResult<Option<OfferWithHash>> {
    match get(action_hash.clone(), GetOptions::default())? {
        Some(record) => {
            let offer: AidOffer = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("No AidOffer found".to_string()))
                })?;
            Ok(Some(OfferWithHash {
                hash: action_hash,
                offer,
            }))
        }
        None => Ok(None),
    }
}

/// Update an offer's status
#[hdk_extern]
pub fn update_offer_status(input: UpdateOfferStatusInput) -> ExternResult<OfferWithHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_offer_status")?;
    // Get the current offer
    let record = get(input.offer_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Offer not found".to_string())))?;

    let mut offer: AidOffer = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No AidOffer found".to_string())))?;

    // Update the offer
    offer.status = input.status;
    offer.updated_at = sys_time()?;

    // Update the entry
    let new_hash = update_entry(input.offer_hash, EntryTypes::AidOffer(offer.clone()))?;

    Ok(OfferWithHash {
        hash: new_hash,
        offer,
    })
}

/// Get all offers for a request
#[hdk_extern]
pub fn get_offers_for_request(request_hash: ActionHash) -> ExternResult<Vec<OfferWithHash>> {
    let links = get_links(
        LinkQuery::try_new(request_hash, LinkTypes::RequestToOffer)?,
        GetStrategy::default(),
    )?;

    let mut offers = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(offer_with_hash) = get_offer(action_hash)? {
                offers.push(offer_with_hash);
            }
        }
    }

    Ok(offers)
}

/// Get all offers by an offerer
#[hdk_extern]
pub fn get_offers_by_offerer(offerer_did: String) -> ExternResult<Vec<OfferWithHash>> {
    let offerer_anchor = get_offerer_anchor(&offerer_did)?;
    let links = get_links(
        LinkQuery::try_new(offerer_anchor, LinkTypes::OffererToOffer)?,
        GetStrategy::default(),
    )?;

    let mut offers = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(offer_with_hash) = get_offer(action_hash)? {
                offers.push(offer_with_hash);
            }
        }
    }

    Ok(offers)
}

/// Cancel an aid request (only by the requester)
#[hdk_extern]
pub fn cancel_request(request_hash: ActionHash) -> ExternResult<RequestWithHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "cancel_request")?;
    update_request_status(UpdateRequestStatusInput {
        request_hash,
        status: RequestStatus::Cancelled,
        fulfilled_amount: None,
    })
}

/// Withdraw an aid offer (only by the offerer)
#[hdk_extern]
pub fn withdraw_offer(offer_hash: ActionHash) -> ExternResult<OfferWithHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "withdraw_offer")?;
    update_offer_status(UpdateOfferStatusInput {
        offer_hash,
        status: OfferStatus::Withdrawn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Input struct serde roundtrip tests ─────────────────────────────

    #[test]
    fn create_request_input_serde_roundtrip() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:alice".to_string(),
            request_type: RequestType::Medical,
            description: "Need help with prescription costs".to_string(),
            urgency: Urgency::High,
            location: Some("Downtown area".to_string()),
            amount_needed: Some(200),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.requester_did, "did:mycelix:alice");
        assert_eq!(decoded.request_type, RequestType::Medical);
        assert_eq!(decoded.urgency, Urgency::High);
        assert_eq!(decoded.location, Some("Downtown area".to_string()));
        assert_eq!(decoded.amount_needed, Some(200));
    }

    #[test]
    fn create_request_input_no_optionals() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:bob".to_string(),
            request_type: RequestType::Food,
            description: "Need groceries this week".to_string(),
            urgency: Urgency::Medium,
            location: None,
            amount_needed: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location, None);
        assert_eq!(decoded.amount_needed, None);
    }

    #[test]
    fn create_request_input_other_type() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:charlie".to_string(),
            request_type: RequestType::Other("Emotional Support".to_string()),
            description: "Going through a rough time".to_string(),
            urgency: Urgency::Low,
            location: None,
            amount_needed: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.request_type,
            RequestType::Other("Emotional Support".to_string())
        );
    }

    #[test]
    fn update_request_status_input_serde_roundtrip() {
        let request_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let input = UpdateRequestStatusInput {
            request_hash: request_hash.clone(),
            status: RequestStatus::PartiallyFulfilled,
            fulfilled_amount: Some(100),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.request_hash, request_hash);
        assert_eq!(decoded.status, RequestStatus::PartiallyFulfilled);
        assert_eq!(decoded.fulfilled_amount, Some(100));
    }

    #[test]
    fn update_request_status_input_no_amount() {
        let request_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let input = UpdateRequestStatusInput {
            request_hash: request_hash.clone(),
            status: RequestStatus::Cancelled,
            fulfilled_amount: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, RequestStatus::Cancelled);
        assert_eq!(decoded.fulfilled_amount, None);
    }

    #[test]
    fn create_offer_input_serde_roundtrip() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash: request_hash.clone(),
            request_id: "req_12345".to_string(),
            offerer_did: "did:mycelix:generous".to_string(),
            amount: Some(150),
            message: "Happy to help with this".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.request_hash, request_hash);
        assert_eq!(decoded.request_id, "req_12345");
        assert_eq!(decoded.offerer_did, "did:mycelix:generous");
        assert_eq!(decoded.amount, Some(150));
        assert_eq!(decoded.message, "Happy to help with this");
    }

    #[test]
    fn create_offer_input_no_amount() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash: request_hash.clone(),
            request_id: "req_67890".to_string(),
            offerer_did: "did:mycelix:helper".to_string(),
            amount: None,
            message: "I can provide the service directly".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, None);
    }

    #[test]
    fn update_offer_status_input_serde_roundtrip() {
        let offer_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = UpdateOfferStatusInput {
            offer_hash: offer_hash.clone(),
            status: OfferStatus::Accepted,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOfferStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.offer_hash, offer_hash);
        assert_eq!(decoded.status, OfferStatus::Accepted);
    }

    #[test]
    fn query_requests_input_serde_roundtrip() {
        let input = QueryRequestsInput {
            request_type: Some(RequestType::Housing),
            status: Some(RequestStatus::Open),
            urgency: Some(Urgency::Critical),
            requester_did: Some("did:mycelix:searcher".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: QueryRequestsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.request_type, Some(RequestType::Housing));
        assert_eq!(decoded.status, Some(RequestStatus::Open));
        assert_eq!(decoded.urgency, Some(Urgency::Critical));
        assert_eq!(
            decoded.requester_did,
            Some("did:mycelix:searcher".to_string())
        );
    }

    #[test]
    fn query_requests_input_all_none() {
        let input = QueryRequestsInput {
            request_type: None,
            status: None,
            urgency: None,
            requester_did: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: QueryRequestsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.request_type, None);
        assert_eq!(decoded.status, None);
        assert_eq!(decoded.urgency, None);
        assert_eq!(decoded.requester_did, None);
    }

    // ── Integrity enum serde roundtrip tests ──────────────────────────

    #[test]
    fn request_type_all_variants_serde() {
        let variants = vec![
            RequestType::Financial,
            RequestType::Housing,
            RequestType::Food,
            RequestType::Medical,
            RequestType::Childcare,
            RequestType::Transportation,
            RequestType::Legal,
            RequestType::Other("Custom Need".to_string()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: RequestType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn urgency_all_variants_serde() {
        let variants = vec![
            Urgency::Critical,
            Urgency::High,
            Urgency::Medium,
            Urgency::Low,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: Urgency = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn request_status_all_variants_serde() {
        let variants = vec![
            RequestStatus::Open,
            RequestStatus::PartiallyFulfilled,
            RequestStatus::Fulfilled,
            RequestStatus::Cancelled,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn offer_status_all_variants_serde() {
        let variants = vec![
            OfferStatus::Pending,
            OfferStatus::Accepted,
            OfferStatus::Completed,
            OfferStatus::Withdrawn,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: OfferStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    // ── AidRequest serde roundtrip ───────────────────────────────────────

    #[test]
    fn aid_request_full_serde_roundtrip() {
        let request = AidRequest {
            id: "req_001".to_string(),
            requester_did: "did:mycelix:alice".to_string(),
            request_type: RequestType::Housing,
            description: "Need temporary housing for 2 weeks".to_string(),
            urgency: Urgency::Critical,
            location: Some("Portland, OR".to_string()),
            amount_needed: Some(3000),
            fulfilled_amount: 500,
            status: RequestStatus::PartiallyFulfilled,
            created_at: Timestamp::from_micros(1000000),
            updated_at: Timestamp::from_micros(2000000),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: AidRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn aid_request_no_optionals_serde_roundtrip() {
        let request = AidRequest {
            id: "req_002".to_string(),
            requester_did: "did:mycelix:bob".to_string(),
            request_type: RequestType::Food,
            description: "Groceries needed".to_string(),
            urgency: Urgency::Low,
            location: None,
            amount_needed: None,
            fulfilled_amount: 0,
            status: RequestStatus::Open,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: AidRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location, None);
        assert_eq!(decoded.amount_needed, None);
        assert_eq!(decoded.fulfilled_amount, 0);
    }

    // ── AidOffer serde roundtrip ─────────────────────────────────────────

    #[test]
    fn aid_offer_full_serde_roundtrip() {
        let offer = AidOffer {
            id: "off_001".to_string(),
            request_id: "req_001".to_string(),
            offerer_did: "did:mycelix:donor".to_string(),
            amount: Some(250),
            message: "I can cover part of it".to_string(),
            status: OfferStatus::Accepted,
            created_at: Timestamp::from_micros(1500000),
            updated_at: Timestamp::from_micros(2500000),
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: AidOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, offer);
    }

    #[test]
    fn aid_offer_no_amount_serde_roundtrip() {
        let offer = AidOffer {
            id: "off_002".to_string(),
            request_id: "req_003".to_string(),
            offerer_did: "did:mycelix:volunteer".to_string(),
            amount: None,
            message: "I can provide the service directly".to_string(),
            status: OfferStatus::Pending,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: AidOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, None);
        assert_eq!(decoded.status, OfferStatus::Pending);
    }

    // ── RequestWithHash serde roundtrip ──────────────────────────────────

    #[test]
    fn request_with_hash_serde_roundtrip() {
        let rwh = RequestWithHash {
            hash: ActionHash::from_raw_36(vec![0xDD; 36]),
            request: AidRequest {
                id: "req_rwh".to_string(),
                requester_did: "did:mycelix:test".to_string(),
                request_type: RequestType::Medical,
                description: "Prescription costs".to_string(),
                urgency: Urgency::High,
                location: None,
                amount_needed: Some(500),
                fulfilled_amount: 0,
                status: RequestStatus::Open,
                created_at: Timestamp::from_micros(0),
                updated_at: Timestamp::from_micros(0),
            },
        };
        let json = serde_json::to_string(&rwh).unwrap();
        let decoded: RequestWithHash = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hash, rwh.hash);
        assert_eq!(decoded.request.id, "req_rwh");
    }

    // ── OfferWithHash serde roundtrip ────────────────────────────────────

    #[test]
    fn offer_with_hash_serde_roundtrip() {
        let owh = OfferWithHash {
            hash: ActionHash::from_raw_36(vec![0xEE; 36]),
            offer: AidOffer {
                id: "off_owh".to_string(),
                request_id: "req_001".to_string(),
                offerer_did: "did:mycelix:helper".to_string(),
                amount: Some(100),
                message: "Small contribution".to_string(),
                status: OfferStatus::Completed,
                created_at: Timestamp::from_micros(0),
                updated_at: Timestamp::from_micros(0),
            },
        };
        let json = serde_json::to_string(&owh).unwrap();
        let decoded: OfferWithHash = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hash, owh.hash);
        assert_eq!(decoded.offer.status, OfferStatus::Completed);
    }

    // ── Status transition patterns ───────────────────────────────────────

    #[test]
    fn request_status_open_to_partially_fulfilled_serde() {
        let input = UpdateRequestStatusInput {
            request_hash: ActionHash::from_raw_36(vec![0xAA; 36]),
            status: RequestStatus::PartiallyFulfilled,
            fulfilled_amount: Some(200),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, RequestStatus::PartiallyFulfilled);
        assert_eq!(decoded.fulfilled_amount, Some(200));
    }

    #[test]
    fn request_status_partially_fulfilled_to_fulfilled_serde() {
        let input = UpdateRequestStatusInput {
            request_hash: ActionHash::from_raw_36(vec![0xAA; 36]),
            status: RequestStatus::Fulfilled,
            fulfilled_amount: Some(1000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, RequestStatus::Fulfilled);
        assert_eq!(decoded.fulfilled_amount, Some(1000));
    }

    #[test]
    fn request_status_open_to_cancelled_serde() {
        let input = UpdateRequestStatusInput {
            request_hash: ActionHash::from_raw_36(vec![0xAA; 36]),
            status: RequestStatus::Cancelled,
            fulfilled_amount: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, RequestStatus::Cancelled);
        assert!(decoded.fulfilled_amount.is_none());
    }

    // ── Offer status transitions ─────────────────────────────────────────

    #[test]
    fn offer_status_pending_to_accepted_serde() {
        let input = UpdateOfferStatusInput {
            offer_hash: ActionHash::from_raw_36(vec![0xCC; 36]),
            status: OfferStatus::Accepted,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOfferStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, OfferStatus::Accepted);
    }

    #[test]
    fn offer_status_accepted_to_completed_serde() {
        let input = UpdateOfferStatusInput {
            offer_hash: ActionHash::from_raw_36(vec![0xCC; 36]),
            status: OfferStatus::Completed,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOfferStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, OfferStatus::Completed);
    }

    #[test]
    fn offer_status_pending_to_withdrawn_serde() {
        let input = UpdateOfferStatusInput {
            offer_hash: ActionHash::from_raw_36(vec![0xCC; 36]),
            status: OfferStatus::Withdrawn,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOfferStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, OfferStatus::Withdrawn);
    }

    // ── Urgency levels in CreateRequestInput ─────────────────────────────

    #[test]
    fn create_request_input_all_urgency_levels() {
        for urgency in [
            Urgency::Critical,
            Urgency::High,
            Urgency::Medium,
            Urgency::Low,
        ] {
            let input = CreateRequestInput {
                requester_did: "did:mycelix:test".to_string(),
                request_type: RequestType::Financial,
                description: "Urgency test".to_string(),
                urgency: urgency.clone(),
                location: None,
                amount_needed: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.urgency, urgency);
        }
    }

    // ── All RequestType variants in CreateRequestInput ────────────────────

    #[test]
    fn create_request_input_all_request_types() {
        let types = vec![
            RequestType::Financial,
            RequestType::Housing,
            RequestType::Food,
            RequestType::Medical,
            RequestType::Childcare,
            RequestType::Transportation,
            RequestType::Legal,
            RequestType::Other("Tutoring".to_string()),
        ];
        for rt in types {
            let input = CreateRequestInput {
                requester_did: "did:mycelix:types".to_string(),
                request_type: rt.clone(),
                description: "Type test".to_string(),
                urgency: Urgency::Medium,
                location: None,
                amount_needed: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.request_type, rt);
        }
    }

    // ── Fulfilled amount boundary values ─────────────────────────────────

    #[test]
    fn update_request_status_zero_fulfilled_amount() {
        let input = UpdateRequestStatusInput {
            request_hash: ActionHash::from_raw_36(vec![0xAA; 36]),
            status: RequestStatus::Open,
            fulfilled_amount: Some(0),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.fulfilled_amount, Some(0));
    }

    #[test]
    fn update_request_status_large_fulfilled_amount() {
        let input = UpdateRequestStatusInput {
            request_hash: ActionHash::from_raw_36(vec![0xAA; 36]),
            status: RequestStatus::Fulfilled,
            fulfilled_amount: Some(u64::MAX),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRequestStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.fulfilled_amount, Some(u64::MAX));
    }

    // ── QueryRequestsInput combinations ──────────────────────────────────

    #[test]
    fn query_requests_input_single_filter() {
        let input = QueryRequestsInput {
            request_type: Some(RequestType::Legal),
            status: None,
            urgency: None,
            requester_did: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: QueryRequestsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.request_type, Some(RequestType::Legal));
        assert!(decoded.status.is_none());
        assert!(decoded.urgency.is_none());
        assert!(decoded.requester_did.is_none());
    }

    #[test]
    fn query_requests_input_status_and_urgency_only() {
        let input = QueryRequestsInput {
            request_type: None,
            status: Some(RequestStatus::Fulfilled),
            urgency: Some(Urgency::Low),
            requester_did: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: QueryRequestsInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.request_type.is_none());
        assert_eq!(decoded.status, Some(RequestStatus::Fulfilled));
        assert_eq!(decoded.urgency, Some(Urgency::Low));
    }

    // ── Validation edge case tests ──────────────────────────────────

    #[test]
    fn create_request_input_whitespace_description_detected() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:alice".to_string(),
            request_type: RequestType::Financial,
            description: "   \t\n  ".to_string(),
            urgency: Urgency::Medium,
            location: None,
            amount_needed: Some(100),
        };
        assert!(
            input.description.trim().is_empty(),
            "Whitespace-only description should be detected"
        );
    }

    #[test]
    fn create_request_input_empty_description_detected() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:alice".to_string(),
            request_type: RequestType::Food,
            description: "".to_string(),
            urgency: Urgency::High,
            location: None,
            amount_needed: None,
        };
        assert!(
            input.description.trim().is_empty(),
            "Empty description should be detected"
        );
    }

    #[test]
    fn create_request_input_whitespace_requester_did_detected() {
        let input = CreateRequestInput {
            requester_did: "  \t ".to_string(),
            request_type: RequestType::Medical,
            description: "Valid description".to_string(),
            urgency: Urgency::Critical,
            location: None,
            amount_needed: None,
        };
        assert!(
            input.requester_did.trim().is_empty(),
            "Whitespace-only DID should be detected"
        );
    }

    #[test]
    fn create_request_input_zero_amount_detected() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:alice".to_string(),
            request_type: RequestType::Financial,
            description: "Need funds".to_string(),
            urgency: Urgency::High,
            location: None,
            amount_needed: Some(0),
        };
        assert_eq!(
            input.amount_needed,
            Some(0),
            "Zero amount should be detected"
        );
    }

    #[test]
    fn create_request_input_valid_accepted() {
        let input = CreateRequestInput {
            requester_did: "did:mycelix:alice".to_string(),
            request_type: RequestType::Financial,
            description: "Need help with rent".to_string(),
            urgency: Urgency::High,
            location: Some("Downtown".to_string()),
            amount_needed: Some(500),
        };
        assert!(
            !input.description.trim().is_empty(),
            "Valid description should pass"
        );
        assert!(
            !input.requester_did.trim().is_empty(),
            "Valid DID should pass"
        );
        assert!(input.amount_needed.unwrap() > 0, "Valid amount should pass");
    }

    #[test]
    fn create_request_input_none_amount_accepted() {
        // None amount_needed is valid (non-financial requests)
        let input = CreateRequestInput {
            requester_did: "did:mycelix:bob".to_string(),
            request_type: RequestType::Childcare,
            description: "Need childcare help".to_string(),
            urgency: Urgency::Medium,
            location: None,
            amount_needed: None,
        };
        assert!(
            input.amount_needed.is_none(),
            "None amount should be accepted for non-financial requests"
        );
    }

    #[test]
    fn create_offer_input_whitespace_message_detected() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash,
            request_id: "req_1".to_string(),
            offerer_did: "did:mycelix:helper".to_string(),
            amount: Some(100),
            message: "   \t\n  ".to_string(),
        };
        assert!(
            input.message.trim().is_empty(),
            "Whitespace-only message should be detected"
        );
    }

    #[test]
    fn create_offer_input_empty_message_detected() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash,
            request_id: "req_2".to_string(),
            offerer_did: "did:mycelix:helper".to_string(),
            amount: None,
            message: "".to_string(),
        };
        assert!(
            input.message.trim().is_empty(),
            "Empty message should be detected"
        );
    }

    #[test]
    fn create_offer_input_zero_amount_detected() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash,
            request_id: "req_3".to_string(),
            offerer_did: "did:mycelix:helper".to_string(),
            amount: Some(0),
            message: "I can help".to_string(),
        };
        assert_eq!(
            input.amount,
            Some(0),
            "Zero offer amount should be detected"
        );
    }

    #[test]
    fn create_offer_input_whitespace_offerer_did_detected() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash,
            request_id: "req_4".to_string(),
            offerer_did: "   ".to_string(),
            amount: Some(50),
            message: "Offering help".to_string(),
        };
        assert!(
            input.offerer_did.trim().is_empty(),
            "Whitespace-only offerer DID should be detected"
        );
    }

    #[test]
    fn create_offer_input_valid_accepted() {
        let request_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = CreateOfferInput {
            request_hash,
            request_id: "req_5".to_string(),
            offerer_did: "did:mycelix:generous".to_string(),
            amount: Some(200),
            message: "Happy to help!".to_string(),
        };
        assert!(
            !input.message.trim().is_empty(),
            "Valid message should pass"
        );
        assert!(
            !input.offerer_did.trim().is_empty(),
            "Valid DID should pass"
        );
        assert!(input.amount.unwrap() > 0, "Valid amount should pass");
    }
}
