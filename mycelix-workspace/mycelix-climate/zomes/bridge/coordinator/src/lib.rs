// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bridge Coordinator Zome
//!
//! Provides functions for cross-hApp climate verification and marketplace integration.
//! Uses HDK 0.6.0-dev.1 with LinkQuery::try_new() pattern.

use bridge_integrity::*;
use hdk::prelude::*;

/// Get or create an anchor for the given string
fn get_or_create_anchor(anchor_text: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_text.to_string());
    let entry_hash = hash_entry(&anchor)?;

    // Check if anchor exists
    if get(entry_hash.clone(), GetOptions::default())?.is_none() {
        create_entry(&EntryTypes::Anchor(anchor))?;
    }

    Ok(entry_hash)
}

// ============================================================================
// Climate Query Management
// ============================================================================

/// Input for creating a climate query
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateQueryInput {
    pub purpose: QueryPurpose,
    pub requester_did: String,
    pub target_id: String,
    pub parameters: Option<String>,
}

/// Create a new climate verification query
#[hdk_extern]
pub fn create_climate_query(input: CreateQueryInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let created_at = now.as_micros() / 1_000_000;
    let query_id = format!("query:{}:{}", input.requester_did, now.as_micros());

    let query = ClimateQuery {
        query_id: query_id.clone(),
        purpose: input.purpose,
        requester_did: input.requester_did.clone(),
        target_id: input.target_id,
        parameters: input.parameters,
        status: QueryStatus::Pending,
        created_at,
    };

    let action_hash = create_entry(&EntryTypes::ClimateQuery(query))?;

    // Link from all queries anchor
    let all_anchor = get_or_create_anchor("all_queries")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToQueries,
        (),
    )?;

    // Link from requester
    let requester_anchor = get_or_create_anchor(&format!("requester:{}", input.requester_did))?;
    create_link(
        requester_anchor,
        action_hash.clone(),
        LinkTypes::RequesterToQueries,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get created query".into()
    )))
}

/// Get a specific query by action hash
#[hdk_extern]
pub fn get_climate_query(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all pending queries
#[hdk_extern]
pub fn get_pending_queries(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor("all_queries")?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::AnchorToQueries)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(q) = record
                .entry()
                .to_app_option::<ClimateQuery>()
                .ok()
                .flatten()
            {
                if q.status == QueryStatus::Pending {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

/// Get queries by requester
#[hdk_extern]
pub fn get_queries_by_requester(requester_did: String) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("requester:{}", requester_did))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::RequesterToQueries)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Input for updating query status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateQueryStatusInput {
    pub query_action_hash: ActionHash,
    pub new_status: QueryStatus,
}

/// Update the status of a query
#[hdk_extern]
pub fn update_query_status(input: UpdateQueryStatusInput) -> ExternResult<Record> {
    let record = get(input.query_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let query: ClimateQuery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid query entry".into()
        )))?;

    let updated = ClimateQuery {
        status: input.new_status,
        ..query
    };

    let action_hash = update_entry(input.query_action_hash, &EntryTypes::ClimateQuery(updated))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get updated query".into()
    )))
}

// ============================================================================
// Climate Result Management
// ============================================================================

/// Input for submitting a query result
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitResultInput {
    pub query_action_hash: ActionHash,
    pub query_id: String,
    pub result: VerificationResult,
    pub data: Option<String>,
    pub responder_did: String,
    pub signature: Option<String>,
}

/// Submit a result for a climate query
#[hdk_extern]
pub fn submit_climate_result(input: SubmitResultInput) -> ExternResult<Record> {
    // Verify the query exists and update its status
    let query_record = get(input.query_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let query: ClimateQuery = query_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid query entry".into()
        )))?;

    // Verify query IDs match
    if query.query_id != input.query_id {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query ID mismatch".into()
        )));
    }

    let now = sys_time()?;
    let responded_at = now.as_micros() / 1_000_000;

    let result = ClimateResult {
        query_id: input.query_id,
        result: input.result,
        data: input.data,
        responder_did: input.responder_did,
        responded_at,
        signature: input.signature,
    };

    let result_action_hash = create_entry(&EntryTypes::ClimateResult(result))?;

    // Link query to result
    create_link(
        input.query_action_hash.clone(),
        result_action_hash.clone(),
        LinkTypes::QueryToResult,
        (),
    )?;

    // Update query status to Completed
    let updated_query = ClimateQuery {
        status: QueryStatus::Completed,
        ..query
    };
    let _ = update_entry(
        input.query_action_hash,
        &EntryTypes::ClimateQuery(updated_query),
    )?;

    get(result_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get created result".into()
    )))
}

/// Get the result for a query
#[hdk_extern]
pub fn get_query_result(query_action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let query = LinkQuery::try_new(query_action_hash, LinkTypes::QueryToResult)?;
    let links = get_links(query, GetStrategy::default())?;

    if let Some(link) = links.first() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

// ============================================================================
// Marketplace Listing Management
// ============================================================================

/// Input for creating a marketplace listing
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateListingInput {
    pub credit_id: String,
    pub credit_action_hash: String,
    pub project_id: String,
    pub seller_did: String,
    pub price_per_tonne: u64,
    pub currency: String,
    pub min_purchase: f64,
    pub available_tonnes: f64,
    pub expires_at: i64,
}

/// Create a new marketplace listing for carbon credits
#[hdk_extern]
pub fn create_marketplace_listing(input: CreateListingInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let created_at = now.as_micros() / 1_000_000;
    let listing_id = format!("listing:{}:{}", input.credit_id, now.as_micros());

    let listing = MarketplaceListing {
        listing_id: listing_id.clone(),
        credit_id: input.credit_id.clone(),
        credit_action_hash: input.credit_action_hash,
        project_id: input.project_id,
        seller_did: input.seller_did.clone(),
        price_per_tonne: input.price_per_tonne,
        currency: input.currency,
        min_purchase: input.min_purchase,
        available_tonnes: input.available_tonnes,
        expires_at: input.expires_at,
        is_active: true,
        created_at,
    };

    let action_hash = create_entry(&EntryTypes::MarketplaceListing(listing))?;

    // Link from all listings anchor
    let all_anchor = get_or_create_anchor("all_listings")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToListings,
        (),
    )?;

    // Link from seller
    let seller_anchor = get_or_create_anchor(&format!("seller:{}", input.seller_did))?;
    create_link(
        seller_anchor,
        action_hash.clone(),
        LinkTypes::SellerToListings,
        (),
    )?;

    // Link from credit
    let credit_anchor = get_or_create_anchor(&format!("credit_listing:{}", input.credit_id))?;
    create_link(
        credit_anchor,
        action_hash.clone(),
        LinkTypes::CreditToListings,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get created listing".into()
    )))
}

/// Get a specific listing by action hash
#[hdk_extern]
pub fn get_marketplace_listing(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all active listings
#[hdk_extern]
pub fn get_active_listings(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor("all_listings")?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::AnchorToListings)?;
    let links = get_links(query, GetStrategy::default())?;

    let now = sys_time()?;
    let current_time = now.as_micros() / 1_000_000;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(listing) = record
                .entry()
                .to_app_option::<MarketplaceListing>()
                .ok()
                .flatten()
            {
                if listing.is_active && listing.expires_at > current_time {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

/// Get listings by seller
#[hdk_extern]
pub fn get_listings_by_seller(seller_did: String) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("seller:{}", seller_did))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::SellerToListings)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get listing for a specific credit
#[hdk_extern]
pub fn get_listing_for_credit(credit_id: String) -> ExternResult<Option<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("credit_listing:{}", credit_id))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::CreditToListings)?;
    let links = get_links(query, GetStrategy::default())?;

    if let Some(link) = links.last() {
        // Get most recent listing
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Input for updating a listing
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateListingInput {
    pub listing_action_hash: ActionHash,
    pub price_per_tonne: Option<u64>,
    pub available_tonnes: Option<f64>,
    pub is_active: Option<bool>,
}

/// Update a marketplace listing
#[hdk_extern]
pub fn update_marketplace_listing(input: UpdateListingInput) -> ExternResult<Record> {
    let record = get(input.listing_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Listing not found".into())),
    )?;

    let listing: MarketplaceListing = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid listing entry".into()
        )))?;

    let updated = MarketplaceListing {
        price_per_tonne: input.price_per_tonne.unwrap_or(listing.price_per_tonne),
        available_tonnes: input.available_tonnes.unwrap_or(listing.available_tonnes),
        is_active: input.is_active.unwrap_or(listing.is_active),
        ..listing
    };

    let action_hash = update_entry(
        input.listing_action_hash,
        &EntryTypes::MarketplaceListing(updated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get updated listing".into()
    )))
}

/// Deactivate a listing
#[hdk_extern]
pub fn deactivate_listing(listing_action_hash: ActionHash) -> ExternResult<Record> {
    update_marketplace_listing(UpdateListingInput {
        listing_action_hash,
        price_per_tonne: None,
        available_tonnes: None,
        is_active: Some(false),
    })
}

// ============================================================================
// Cross-hApp Verification Functions
// ============================================================================

/// Input for verifying a credit via cross-hApp query
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyCreditInput {
    pub credit_id: String,
    pub requester_did: String,
}

/// Create a credit verification query
#[hdk_extern]
pub fn verify_credit(input: VerifyCreditInput) -> ExternResult<Record> {
    create_climate_query(CreateQueryInput {
        purpose: QueryPurpose::CreditVerification,
        requester_did: input.requester_did,
        target_id: input.credit_id,
        parameters: None,
    })
}

/// Input for auditing a footprint via cross-hApp query
#[derive(Serialize, Deserialize, Debug)]
pub struct AuditFootprintInput {
    pub footprint_action_hash: String,
    pub requester_did: String,
}

/// Create a footprint audit query
#[hdk_extern]
pub fn audit_footprint(input: AuditFootprintInput) -> ExternResult<Record> {
    create_climate_query(CreateQueryInput {
        purpose: QueryPurpose::FootprintAudit,
        requester_did: input.requester_did,
        target_id: input.footprint_action_hash,
        parameters: None,
    })
}

/// Input for project due diligence via cross-hApp query
#[derive(Serialize, Deserialize, Debug)]
pub struct ProjectDueDiligenceInput {
    pub project_id: String,
    pub requester_did: String,
    pub additional_checks: Option<String>,
}

/// Create a project due diligence query
#[hdk_extern]
pub fn project_due_diligence(input: ProjectDueDiligenceInput) -> ExternResult<Record> {
    create_climate_query(CreateQueryInput {
        purpose: QueryPurpose::ProjectDueDiligence,
        requester_did: input.requester_did,
        target_id: input.project_id,
        parameters: input.additional_checks,
    })
}

// ============================================================================
// Statistics
// ============================================================================

/// Summary of bridge activity
#[derive(Serialize, Deserialize, Debug)]
pub struct BridgeSummary {
    pub total_queries: u64,
    pub pending_queries: u64,
    pub completed_queries: u64,
    pub verification_success_rate: f64,
    pub total_listings: u64,
    pub active_listings: u64,
    pub total_listed_tonnes: f64,
}

/// Get a summary of bridge activity
#[hdk_extern]
pub fn get_bridge_summary(_: ()) -> ExternResult<BridgeSummary> {
    let now = sys_time()?;
    let current_time = now.as_micros() / 1_000_000;

    // Get all queries
    let query_anchor = get_or_create_anchor("all_queries")?;
    let query_links = get_links(
        LinkQuery::try_new(query_anchor, LinkTypes::AnchorToQueries)?,
        GetStrategy::default(),
    )?;

    let mut total_queries = 0u64;
    let mut pending_queries = 0u64;
    let mut completed_queries = 0u64;
    let mut successful_verifications = 0u64;

    for link in query_links {
        if let Ok(action_hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(q) = record
                    .entry()
                    .to_app_option::<ClimateQuery>()
                    .ok()
                    .flatten()
                {
                    total_queries += 1;
                    match q.status {
                        QueryStatus::Pending => pending_queries += 1,
                        QueryStatus::Completed => {
                            completed_queries += 1;
                            // Check if result was successful
                            if let Some(result_record) = get_query_result(action_hash)? {
                                if let Some(r) = result_record
                                    .entry()
                                    .to_app_option::<ClimateResult>()
                                    .ok()
                                    .flatten()
                                {
                                    if r.result == VerificationResult::Verified {
                                        successful_verifications += 1;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Get all listings
    let listing_anchor = get_or_create_anchor("all_listings")?;
    let listing_links = get_links(
        LinkQuery::try_new(listing_anchor, LinkTypes::AnchorToListings)?,
        GetStrategy::default(),
    )?;

    let mut total_listings = 0u64;
    let mut active_listings = 0u64;
    let mut total_listed_tonnes = 0.0f64;

    for link in listing_links {
        if let Ok(action_hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(l) = record
                    .entry()
                    .to_app_option::<MarketplaceListing>()
                    .ok()
                    .flatten()
                {
                    total_listings += 1;
                    if l.is_active && l.expires_at > current_time {
                        active_listings += 1;
                        total_listed_tonnes += l.available_tonnes;
                    }
                }
            }
        }
    }

    let verification_success_rate = if completed_queries > 0 {
        (successful_verifications as f64) / (completed_queries as f64)
    } else {
        0.0
    };

    Ok(BridgeSummary {
        total_queries,
        pending_queries,
        completed_queries,
        verification_success_rate,
        total_listings,
        active_listings,
        total_listed_tonnes,
    })
}
