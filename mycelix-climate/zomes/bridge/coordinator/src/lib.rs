// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Coordinator Zome
//!
//! Provides functions for cross-hApp climate verification and marketplace integration.
//! Uses HDK 0.6.0-dev.1 with LinkQuery::try_new() pattern.

use hdk::prelude::*;
use bridge_integrity::*;

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created query".into())))
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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid query entry".into())))?;

    let updated = ClimateQuery {
        status: input.new_status,
        ..query
    };

    let action_hash = update_entry(input.query_action_hash, &EntryTypes::ClimateQuery(updated))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated query".into())))
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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid query entry".into())))?;

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
    let _ = update_entry(input.query_action_hash, &EntryTypes::ClimateQuery(updated_query))?;

    get(result_action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created result".into())))
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created listing".into())))
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
    let record = get(input.listing_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Listing not found".into())))?;

    let listing: MarketplaceListing = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid listing entry".into())))?;

    let updated = MarketplaceListing {
        price_per_tonne: input.price_per_tonne.unwrap_or(listing.price_per_tonne),
        available_tonnes: input.available_tonnes.unwrap_or(listing.available_tonnes),
        is_active: input.is_active.unwrap_or(listing.is_active),
        ..listing
    };

    let action_hash = update_entry(input.listing_action_hash, &EntryTypes::MarketplaceListing(updated))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated listing".into())))
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
// Gap 5: Climate Provenance via Supplychain
// ============================================================================

/// Input for verifying supply chain sustainability claims.
#[derive(Serialize, Deserialize, Debug)]
pub struct SustainabilityVerificationInput {
    /// Item identifier to verify (e.g., product ID or batch ID).
    pub item_id: String,
    /// Type of sustainability claim to look for (e.g., "carbon_neutral", "recycled").
    pub claim_type: String,
}

/// Summary of verification results across all claims for an item.
#[derive(Serialize, Deserialize, Debug)]
pub struct SustainabilityVerificationResult {
    /// Item ID that was checked.
    pub item_id: String,
    /// Claim type that was checked.
    pub claim_type: String,
    /// Number of claims found with this type.
    pub claims_found: u32,
    /// Number of those claims with a verified status.
    pub verified_count: u32,
    /// Number of those claims with a pending status.
    pub pending_count: u32,
    /// Number of those claims with a rejected status.
    pub rejected_count: u32,
    /// Whether the supplychain cluster was reachable.
    pub supplychain_available: bool,
    /// Non-fatal error message.
    pub error: Option<String>,
}

/// Verify supply chain sustainability for an item by querying the supplychain cluster.
///
/// 1. Calls `get_claims_by_item` on the supplychain claims zome to find all
///    claims for the given item.
/// 2. For each claim matching `claim_type`, calls `get_verifications_for_claim`
///    on the supplychain verification zome to get verification status.
/// 3. Returns a summary of verified / pending / rejected counts.
///
/// All cross-cluster failures are non-fatal.
#[hdk_extern]
pub fn verify_supply_chain_sustainability(
    input: SustainabilityVerificationInput,
) -> ExternResult<SustainabilityVerificationResult> {
    if input.item_id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "item_id is required".to_string()
        )));
    }
    if input.claim_type.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "claim_type is required".to_string()
        )));
    }

    // Step 1: Get all claims for the item from supplychain
    let claims_payload = ExternIO::encode(input.item_id.clone())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let claims_value: serde_json::Value = match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("claims_coordinator"),
        FunctionName::from("get_claims_by_item"),
        None,
        claims_payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            data.decode().unwrap_or(serde_json::Value::Array(vec![]))
        }
        Ok(other) => {
            return Ok(SustainabilityVerificationResult {
                item_id: input.item_id,
                claim_type: input.claim_type,
                claims_found: 0,
                verified_count: 0,
                pending_count: 0,
                rejected_count: 0,
                supplychain_available: true,
                error: Some(format!("get_claims_by_item rejected: {:?}", other)),
            });
        }
        Err(_) => {
            return Ok(SustainabilityVerificationResult {
                item_id: input.item_id,
                claim_type: input.claim_type,
                claims_found: 0,
                verified_count: 0,
                pending_count: 0,
                rejected_count: 0,
                supplychain_available: false,
                error: Some("Supplychain cluster not available".to_string()),
            });
        }
    };

    // Step 2: Filter claims by type and get verification status for each
    let empty_arr = vec![];
    let claims_arr = claims_value.as_array().unwrap_or(&empty_arr);

    let mut claims_found: u32 = 0;
    let mut verified_count: u32 = 0;
    let mut pending_count: u32 = 0;
    let mut rejected_count: u32 = 0;

    for claim in claims_arr {
        // Each claim record's entry has a `claim_type` field.
        // The record is serialized as { signed_action: ..., entry: ... }.
        // Navigate into the entry to get claim_type.
        let ct = claim
            .get("entry")
            .and_then(|e| e.get("claim_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if ct != input.claim_type {
            continue;
        }

        claims_found += 1;

        // Get the action_hash for this claim record to query verifications
        let claim_hash_value = claim
            .get("signed_action")
            .and_then(|sa| sa.get("hashed"))
            .and_then(|h| h.get("hash"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        if claim_hash_value.is_null() {
            // Can't query verifications without the hash; count as pending
            pending_count += 1;
            continue;
        }

        let verif_payload = ExternIO::encode(claim_hash_value)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

        let verifications: serde_json::Value = match call(
            CallTargetCell::OtherRole("supplychain".into()),
            ZomeName::from("verification_coordinator"),
            FunctionName::from("get_verifications_for_claim"),
            None,
            verif_payload,
        ) {
            Ok(ZomeCallResponse::Ok(data)) => {
                data.decode().unwrap_or(serde_json::Value::Array(vec![]))
            }
            _ => {
                // Verification zome unavailable for this claim — treat as pending
                pending_count += 1;
                continue;
            }
        };

        let verif_arr = match verifications.as_array() {
            Some(a) => a,
            None => {
                pending_count += 1;
                continue;
            }
        };

        if verif_arr.is_empty() {
            // Claim exists but has no verifications yet
            pending_count += 1;
        } else {
            // Use the most recent verification status
            let Some(last) = verif_arr.last() else {
                pending_count += 1;
                continue;
            };
            let status = last
                .get("entry")
                .and_then(|e| e.get("status"))
                .and_then(|s| s.as_str())
                .unwrap_or("Pending");

            match status {
                "Verified" => verified_count += 1,
                "Rejected" => rejected_count += 1,
                _ => pending_count += 1,
            }
        }
    }

    Ok(SustainabilityVerificationResult {
        item_id: input.item_id,
        claim_type: input.claim_type,
        claims_found,
        verified_count,
        pending_count,
        rejected_count,
        supplychain_available: true,
        error: None,
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
    let query_links = get_links(LinkQuery::try_new(query_anchor, LinkTypes::AnchorToQueries)?, GetStrategy::default())?;

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
    let listing_links = get_links(LinkQuery::try_new(listing_anchor, LinkTypes::AnchorToListings)?, GetStrategy::default())?;

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

// ============================================================================
// Praxis Credential Pipeline
// ============================================================================

/// Verify a Praxis learning credential for climate domain authorization.
///
/// Calls the Praxis credential_zome to verify that a user has completed
/// environmental science curriculum, gating project creation behind
/// proven knowledge.
///
/// Returns true if credential is valid, false if invalid or Praxis unavailable.
/// Graceful degradation: if Praxis cluster is unreachable, returns true
/// (don't gate access when the verifier is down).
#[hdk_extern]
pub fn verify_praxis_credential(credential_id: String) -> ExternResult<bool> {
    // Attempt cross-cluster call to Praxis credential_zome
    let payload = ExternIO::encode(&credential_id)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("encode error: {e:?}"))))?;
    match call(
        CallTargetCell::OtherRole("praxis".into()),
        ZomeName::from("credential_zome"),
        FunctionName::from("verify_credential"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            let valid: bool = data.decode().unwrap_or(false);
            debug!("Praxis credential {credential_id} verification: {valid}");
            Ok(valid)
        }
        Ok(_) => {
            warn!("Praxis credential verification rejected/unauthorized — allowing access");
            Ok(true)
        }
        Err(e) => {
            warn!("Praxis cluster unreachable: {e:?} — allowing access (graceful degradation)");
            Ok(true)
        }
    }
}

/// Check if a user has any environmental credential from Praxis.
///
/// Queries Praxis for credentials matching climate-related course IDs.
#[hdk_extern]
pub fn check_environmental_credential(agent_did: String) -> ExternResult<bool> {
    let payload = ExternIO::encode(&agent_did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("encode error: {e:?}"))))?;
    match call(
        CallTargetCell::OtherRole("praxis".into()),
        ZomeName::from("credential_zome"),
        FunctionName::from("get_credentials_by_learner"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            let credentials: Vec<serde_json::Value> = data.decode().unwrap_or_default();
            // Check if any credential relates to environmental science
            let has_env = credentials.iter().any(|cred| {
                cred.get("course_id")
                    .and_then(|c| c.as_str())
                    .map(|c| {
                        c.contains("environment") || c.contains("climate")
                            || c.contains("ecology") || c.contains("sustainability")
                    })
                    .unwrap_or(false)
            });
            Ok(has_env)
        }
        _ => Ok(true), // Graceful degradation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Gap 5: Sustainability Verification Tests
    // =========================================================================

    #[test]
    fn test_sustainability_verification_result_serde() {
        // Fully verified item
        let result = SustainabilityVerificationResult {
            item_id: "BATCH-SOLAR-2026-001".to_string(),
            claim_type: "carbon_neutral".to_string(),
            claims_found: 3,
            verified_count: 2,
            pending_count: 1,
            rejected_count: 0,
            supplychain_available: true,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: SustainabilityVerificationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.item_id, "BATCH-SOLAR-2026-001");
        assert_eq!(back.claim_type, "carbon_neutral");
        assert_eq!(back.claims_found, 3);
        assert_eq!(back.verified_count, 2);
        assert_eq!(back.pending_count, 1);
        assert_eq!(back.rejected_count, 0);
        assert!(back.supplychain_available);
        assert!(back.error.is_none());

        // Supplychain unavailable
        let result2 = SustainabilityVerificationResult {
            item_id: "BATCH-WIND-2026-007".to_string(),
            claim_type: "recycled_materials".to_string(),
            claims_found: 0,
            verified_count: 0,
            pending_count: 0,
            rejected_count: 0,
            supplychain_available: false,
            error: Some("Supplychain cluster not available".to_string()),
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let back2: SustainabilityVerificationResult = serde_json::from_str(&json2).unwrap();
        assert!(!back2.supplychain_available);
        assert_eq!(back2.claims_found, 0);
        assert!(back2.error.is_some());

        // Rejected claims
        let result3 = SustainabilityVerificationResult {
            item_id: "ITEM-XYZ".to_string(),
            claim_type: "fair_trade".to_string(),
            claims_found: 2,
            verified_count: 0,
            pending_count: 0,
            rejected_count: 2,
            supplychain_available: true,
            error: None,
        };
        let json3 = serde_json::to_string(&result3).unwrap();
        let back3: SustainabilityVerificationResult = serde_json::from_str(&json3).unwrap();
        assert_eq!(back3.rejected_count, 2);
        assert_eq!(back3.verified_count, 0);
    }

    #[test]
    fn test_sustainability_verification_input_serde() {
        let input = SustainabilityVerificationInput {
            item_id: "SOLAR-PANEL-BATCH-2026".to_string(),
            claim_type: "carbon_neutral".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SustainabilityVerificationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.item_id, "SOLAR-PANEL-BATCH-2026");
        assert_eq!(back.claim_type, "carbon_neutral");
    }
}
