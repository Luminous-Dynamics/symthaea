// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Requests Coordinator Zome - Aid request and offer management
//!
//! This zome provides the coordination logic for creating, updating, and
//! querying aid requests and offers in the mutual aid network.

use hdk::prelude::*;
use requests_integrity::{
    AidOffer, AidRequest, Anchor as RequestsAnchor, EntryTypes, LinkTypes, OfferStatus,
    RequestStatus, RequestType, Urgency,
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
    update_request_status(UpdateRequestStatusInput {
        request_hash,
        status: RequestStatus::Cancelled,
        fulfilled_amount: None,
    })
}

/// Withdraw an aid offer (only by the offerer)
#[hdk_extern]
pub fn withdraw_offer(offer_hash: ActionHash) -> ExternResult<OfferWithHash> {
    update_offer_status(UpdateOfferStatusInput {
        offer_hash,
        status: OfferStatus::Withdrawn,
    })
}
