// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use listings_integrity::*;
use mycelix_common::{bridge, error_handling, link_queries, time};

/// Create a new listing
///
/// This function:
/// 1. Sanitizes all user inputs for security
/// 2. Creates the listing entry on the DHT
/// 3. Creates links for discovery (agent, category, status, all)
/// 4. Returns the listing with its action hash
#[hdk_extern]
pub fn create_listing(input: CreateListingInput) -> ExternResult<ListingOutput> {
    let agent_info = agent_info()?;

    // Sanitize inputs to prevent XSS and injection attacks
    let sanitized_title = security::sanitize_user_input(&input.title);
    let sanitized_description = security::sanitize_user_input(&input.description);

    // Validate and sanitize IPFS CIDs
    let mut sanitized_cids = Vec::new();
    for cid in &input.photos_ipfs_cids {
        match security::sanitize_ipfs_cid(cid) {
            Ok(clean_cid) => sanitized_cids.push(clean_cid),
            Err(e) => {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    format!("Invalid IPFS CID: {}", e)
                )));
            }
        }
    }

    // Validate price and quantity
    security::validate_price(input.price_cents).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid price: {}",
            e
        )))
    })?;

    security::validate_quantity(input.quantity_available).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid quantity: {}",
            e
        )))
    })?;

    // Build listing with Epistemic Charter classification
    let listing = Listing {
        title: sanitized_title,
        description: sanitized_description,
        price_cents: input.price_cents,
        category: input.category,
        photos_ipfs_cids: sanitized_cids,
        quantity_available: input.quantity_available,
        status: ListingStatus::Active,
        epistemic: EpistemicClassification {
            // Seller's testimonial claim
            empirical: EmpiricalLevel::E1Testimonial,
            // Personal claim (seller only)
            normative: NormativeLevel::N0Personal,
            // Temporal - prune when sold
            materiality: MaterialityLevel::M1Temporal,
        },
        created_at: time::now()?,
        updated_at: time::now()?,
    };

    // Create entry on DHT
    let action_hash = create_entry(&EntryTypes::Listing(listing.clone()))?;
    let entry_hash = hash_entry(&listing)?;

    // Create discovery links

    // 1. Agent -> Listing (seller's listings)
    // Clone to avoid move since we use agent_initial_pubkey again later
    let agent_path = agent_info.agent_initial_pubkey.clone();
    create_link(
        agent_path.clone(), // Clone here since we use agent_path again for monitoring
        entry_hash.clone(),
        LinkTypes::AgentToListings,
        (),
    )?;

    // 2. Category -> Listing (browse by category)
    let category_path = Path::from(format!("listings.category.{:?}", listing.category));
    // Note: HDK 0.6.0 - Path.ensure() removed, paths auto-created
    create_link(
        category_path.path_entry_hash()?,
        entry_hash.clone(),
        LinkTypes::CategoryToListings,
        (),
    )?;

    // 3. Status -> Listing (filter by status)
    let status_path = Path::from(format!("listings.status.{:?}", listing.status));
    // Note: HDK 0.6.0 - Path.ensure() removed, paths auto-created
    create_link(
        status_path.path_entry_hash()?,
        entry_hash.clone(),
        LinkTypes::StatusToListings,
        (),
    )?;

    // 4. All Listings anchor
    let all_path = Path::from("all_listings");
    // Note: HDK 0.6.0 - Path.ensure() removed, paths auto-created
    create_link(
        all_path.path_entry_hash()?,
        entry_hash.clone(),
        LinkTypes::AllListings,
        (),
    )?;

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::ListingCreated,
        1.0,
        Some(agent_path.clone()),
        Some(format!("category:{:?},price:{}", listing.category, listing.price_cents)),
    )?;

    Ok(ListingOutput {
        listing_hash: action_hash.into(),
        listing,
        seller_agent_id: agent_info.agent_initial_pubkey.into(),
    })
}

/// Get a specific listing by hash
#[hdk_extern]
pub fn get_listing(listing_hash: ActionHash) -> ExternResult<Option<ListingOutput>> {
    let record = get(listing_hash.clone(), GetOptions::default())?;

    match record {
        Some(record) => {
            // Use shared utility for deserialization
            let listing: Listing = error_handling::deserialize_entry(&record)?;

            Ok(Some(ListingOutput {
                listing_hash: listing_hash.into(),
                listing,
                seller_agent_id: record.action().author().clone().into(),
            }))
        }
        None => Ok(None),
    }
}

/// Get all listings in the marketplace
#[hdk_extern]
pub fn get_all_listings(_: ()) -> ExternResult<ListingsResponse> {
    let path = Path::from("all_listings");
    // Use shared utility for get_links
    let links = link_queries::get_links_local(path.path_entry_hash()?, LinkTypes::AllListings)?;

    let mut listings = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(listing_output) = get_listing(action_hash)? {
                // Filter out deleted/inactive if needed
                if listing_output.listing.status != ListingStatus::Deleted {
                    listings.push(listing_output);
                }
            }
        }
    }

    Ok(ListingsResponse { listings })
}

/// Get listings by seller
#[hdk_extern]
pub fn get_listings_by_seller(agent_id: AgentPubKey) -> ExternResult<ListingsResponse> {
    // Use shared utility for get_links
    let links = link_queries::get_links_local(agent_id, LinkTypes::AgentToListings)?;

    let mut listings = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(listing_output) = get_listing(action_hash)? {
                listings.push(listing_output);
            }
        }
    }

    Ok(ListingsResponse { listings })
}

/// Get my listings (current user)
#[hdk_extern]
pub fn get_my_listings(_: ()) -> ExternResult<ListingsResponse> {
    let agent_info = agent_info()?;
    get_listings_by_seller(agent_info.agent_initial_pubkey)
}

/// Get listings by category
#[hdk_extern]
pub fn get_listings_by_category(category: ListingCategory) -> ExternResult<ListingsResponse> {
    let path = Path::from(format!("listings.category.{:?}", category));
    // Use shared utility for get_links
    let links = link_queries::get_links_local(path.path_entry_hash()?, LinkTypes::CategoryToListings)?;

    let mut listings = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(listing_output) = get_listing(action_hash)? {
                listings.push(listing_output);
            }
        }
    }

    Ok(ListingsResponse { listings })
}

/// Update a listing
#[hdk_extern]
pub fn update_listing(input: UpdateListingInput) -> ExternResult<ListingOutput> {
    // Get the original listing
    let original_record = get(input.listing_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Listing not found".into()
        )))?;

    // Use shared utility for deserialization
    let mut listing: Listing = error_handling::deserialize_entry(&original_record)?;

    // Verify ownership
    let agent_info = agent_info()?;
    if original_record.action().author() != &agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the seller can update their listing".into()
        )));
    }

    // Apply updates
    if let Some(title) = input.title {
        listing.title = title;
    }
    if let Some(description) = input.description {
        listing.description = description;
    }
    if let Some(price_cents) = input.price_cents {
        listing.price_cents = price_cents;
    }
    if let Some(category) = input.category {
        listing.category = category;
    }
    if let Some(photos_ipfs_cids) = input.photos_ipfs_cids {
        listing.photos_ipfs_cids = photos_ipfs_cids;
    }
    if let Some(quantity_available) = input.quantity_available {
        listing.quantity_available = quantity_available;
    }
    if let Some(status) = input.status {
        listing.status = status;
    }

    listing.updated_at = time::now()?;

    // Create update
    let action_hash = update_entry(input.listing_hash, &EntryTypes::Listing(listing.clone()))?;

    Ok(ListingOutput {
        listing_hash: action_hash.into(),
        listing,
        seller_agent_id: agent_info.agent_initial_pubkey.into(),
    })
}

/// Delete a listing (soft delete by setting status to Deleted)
#[hdk_extern]
pub fn delete_listing(listing_hash: ActionHash) -> ExternResult<()> {
    update_listing(UpdateListingInput {
        listing_hash,
        title: None,
        description: None,
        price_cents: None,
        category: None,
        photos_ipfs_cids: None,
        quantity_available: None,
        status: Some(ListingStatus::Deleted),
    })?;

    Ok(())
}

/// Search listings by text query (simple implementation)
#[hdk_extern]
pub fn search_listings(query: String) -> ExternResult<ListingsResponse> {
    let all = get_all_listings(())?;
    let query_lower = query.to_lowercase();

    let filtered: Vec<ListingOutput> = all
        .listings
        .into_iter()
        .filter(|output| {
            output.listing.title.to_lowercase().contains(&query_lower)
                || output.listing.description.to_lowercase().contains(&query_lower)
        })
        .collect();

    Ok(ListingsResponse {
        listings: filtered,
    })
}

/// Get seller verification and trust information from Bridge
///
/// This queries the Bridge zome to get cross-app identity verification
/// and reputation data for a seller. Useful for buyer trust decisions.
#[hdk_extern]
pub fn get_seller_verification(seller: AgentPubKey) -> ExternResult<SellerVerificationResponse> {
    // Query Bridge for identity info
    let identity_result = bridge::query_identity(&seller);

    // Get DID for reputation lookup
    let did = bridge::agent_to_did(&seller);

    // Query Bridge for cross-app reputation
    let reputation_result = bridge::get_cross_app_reputation(&did);

    // Build response based on Bridge availability
    let (identity_verified, verification_level, identity_error) = match identity_result {
        bridge::BridgeResult::Success(identity) => {
            (identity.verified, Some(identity.verification_level), None)
        }
        bridge::BridgeResult::Unavailable => {
            (false, None, Some("Bridge identity service unavailable".to_string()))
        }
        bridge::BridgeResult::Error(e) => {
            (false, None, Some(e))
        }
    };

    let (cross_app_score, app_count, total_transactions, reputation_error) = match reputation_result {
        bridge::BridgeResult::Success(rep) => {
            (Some(rep.score), Some(rep.app_count), Some(rep.total_transactions), None)
        }
        bridge::BridgeResult::Unavailable => {
            (None, None, None, Some("Bridge reputation service unavailable".to_string()))
        }
        bridge::BridgeResult::Error(e) => {
            (None, None, None, Some(e))
        }
    };

    Ok(SellerVerificationResponse {
        seller: seller.clone(),
        did,
        identity_verified,
        verification_level,
        cross_app_reputation: cross_app_score,
        app_count,
        total_transactions,
        bridge_available: identity_error.is_none() || reputation_error.is_none(),
        errors: vec![identity_error, reputation_error]
            .into_iter()
            .flatten()
            .collect(),
    })
}

/// Get listing with enhanced seller trust info
///
/// This returns the listing along with cross-app seller verification
/// and reputation data from Bridge.
#[hdk_extern]
pub fn get_listing_with_trust(listing_hash: ActionHash) -> ExternResult<ListingWithTrustOutput> {
    // Get the listing
    let listing_output = get_listing(listing_hash.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Listing not found".into()
        )))?;

    // Get seller verification info
    let seller_verification = get_seller_verification(listing_output.seller_agent_id.clone())?;

    Ok(ListingWithTrustOutput {
        listing_hash: listing_output.listing_hash,
        listing: listing_output.listing,
        seller_agent_id: listing_output.seller_agent_id,
        seller_trust: SellerTrustInfo {
            identity_verified: seller_verification.identity_verified,
            verification_level: seller_verification.verification_level,
            cross_app_reputation: seller_verification.cross_app_reputation,
            app_count: seller_verification.app_count,
            bridge_available: seller_verification.bridge_available,
        },
    })
}

/// Verify seller meets minimum trust requirements before transaction
///
/// This is a helper function that can be called before creating a transaction
/// to ensure the seller has adequate cross-app reputation.
#[hdk_extern]
pub fn verify_seller_trust(input: VerifySellerTrustInput) -> ExternResult<SellerTrustCheckResult> {
    let did = bridge::agent_to_did(&input.seller);

    // Get cross-app reputation
    let reputation_result = bridge::get_cross_app_reputation(&did);

    match reputation_result {
        bridge::BridgeResult::Success(rep) => {
            let meets_threshold = rep.score >= input.min_reputation_score;
            let has_enough_transactions = rep.total_transactions >= input.min_transactions as u64;

            Ok(SellerTrustCheckResult {
                seller: input.seller,
                passes: meets_threshold && has_enough_transactions,
                score: rep.score,
                required_score: input.min_reputation_score,
                transaction_count: rep.total_transactions,
                required_transactions: input.min_transactions as u64,
                bridge_available: true,
                reason: if meets_threshold && has_enough_transactions {
                    None
                } else if !meets_threshold {
                    Some(format!(
                        "Reputation score {:.2} below minimum {:.2}",
                        rep.score, input.min_reputation_score
                    ))
                } else {
                    Some(format!(
                        "Transaction count {} below minimum {}",
                        rep.total_transactions, input.min_transactions
                    ))
                },
            })
        }
        bridge::BridgeResult::Unavailable => {
            // Bridge unavailable - pass by default (rely on local reputation)
            Ok(SellerTrustCheckResult {
                seller: input.seller,
                passes: true, // Allow transaction when Bridge unavailable
                score: 0.5,   // Default neutral score
                required_score: input.min_reputation_score,
                transaction_count: 0,
                required_transactions: input.min_transactions as u64,
                bridge_available: false,
                reason: Some("Bridge unavailable - using local reputation only".to_string()),
            })
        }
        bridge::BridgeResult::Error(e) => {
            Ok(SellerTrustCheckResult {
                seller: input.seller,
                passes: true, // Don't block on Bridge errors
                score: 0.5,
                required_score: input.min_reputation_score,
                transaction_count: 0,
                required_transactions: input.min_transactions as u64,
                bridge_available: false,
                reason: Some(format!("Bridge error: {}", e)),
            })
        }
    }
}

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateListingInput {
    pub title: String,
    pub description: String,
    pub price_cents: u64,
    pub category: ListingCategory,
    pub photos_ipfs_cids: Vec<String>,
    pub quantity_available: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateListingInput {
    pub listing_hash: ActionHash,
    pub title: Option<String>,
    pub description: Option<String>,
    pub price_cents: Option<u64>,
    pub category: Option<ListingCategory>,
    pub photos_ipfs_cids: Option<Vec<String>>,
    pub quantity_available: Option<u32>,
    pub status: Option<ListingStatus>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListingOutput {
    pub listing_hash: ActionHash,
    pub listing: Listing,
    pub seller_agent_id: AgentPubKey,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListingsResponse {
    pub listings: Vec<ListingOutput>,
}

/// Response for seller verification query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SellerVerificationResponse {
    /// The seller being verified
    pub seller: AgentPubKey,
    /// Decentralized identifier
    pub did: String,
    /// Whether identity is verified through Bridge
    pub identity_verified: bool,
    /// Verification level (0-3)
    pub verification_level: Option<u8>,
    /// Cross-app reputation score
    pub cross_app_reputation: Option<f64>,
    /// Number of apps contributing to reputation
    pub app_count: Option<u32>,
    /// Total transactions across all apps
    pub total_transactions: Option<u64>,
    /// Whether Bridge services are available
    pub bridge_available: bool,
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Seller trust info included with listing
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SellerTrustInfo {
    /// Whether seller identity is verified
    pub identity_verified: bool,
    /// Verification level
    pub verification_level: Option<u8>,
    /// Cross-app reputation score
    pub cross_app_reputation: Option<f64>,
    /// Number of apps contributing
    pub app_count: Option<u32>,
    /// Whether Bridge is available
    pub bridge_available: bool,
}

/// Listing output with seller trust information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListingWithTrustOutput {
    pub listing_hash: ActionHash,
    pub listing: Listing,
    pub seller_agent_id: AgentPubKey,
    pub seller_trust: SellerTrustInfo,
}

/// Input for seller trust verification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VerifySellerTrustInput {
    /// The seller to verify
    pub seller: AgentPubKey,
    /// Minimum required reputation score (0.0 - 1.0)
    pub min_reputation_score: f64,
    /// Minimum required transaction count
    pub min_transactions: u32,
}

/// Result of seller trust verification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SellerTrustCheckResult {
    /// The seller being checked
    pub seller: AgentPubKey,
    /// Whether seller passes all trust checks
    pub passes: bool,
    /// Actual reputation score
    pub score: f64,
    /// Required minimum score
    pub required_score: f64,
    /// Actual transaction count
    pub transaction_count: u64,
    /// Required minimum transactions
    pub required_transactions: u64,
    /// Whether Bridge was available for check
    pub bridge_available: bool,
    /// Reason for failure (if any)
    pub reason: Option<String>,
}


// ===== Tests =====
#[cfg(test)]
mod tests;
