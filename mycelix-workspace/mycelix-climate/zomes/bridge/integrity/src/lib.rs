// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Integrity Zome
//!
//! Defines entry types and validation for cross-hApp climate verification.
//! Uses HDI 0.7.0-dev.1 with FlatOp validation pattern.

use hdi::prelude::*;
use mycelix_bridge_entry_types::CrossClusterNotification;

/// Anchor entry for creating deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Anchor(pub String);

/// Purpose of a climate query
#[hdk_entry_helper]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QueryPurpose {
    /// Verify the authenticity of a carbon credit
    CreditVerification,
    /// Audit a carbon footprint
    FootprintAudit,
    /// Due diligence on a climate project
    ProjectDueDiligence,
}

/// Status of a query
#[hdk_entry_helper]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QueryStatus {
    /// Query submitted, awaiting processing
    Pending,
    /// Query is being processed
    Processing,
    /// Query completed with results
    Completed,
    /// Query failed
    Failed,
}

/// Result type for climate queries
#[hdk_entry_helper]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VerificationResult {
    /// Verification passed
    Verified,
    /// Verification failed
    Failed,
    /// Inconclusive - needs more information
    Inconclusive,
    /// Target not found
    NotFound,
}

/// A cross-hApp query for climate verification
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClimateQuery {
    /// Unique query identifier
    pub query_id: String,
    /// Purpose of the query
    pub purpose: QueryPurpose,
    /// DID of the requester
    pub requester_did: String,
    /// Target ID (credit ID, footprint ID, or project ID)
    pub target_id: String,
    /// Optional additional parameters as JSON
    pub parameters: Option<String>,
    /// Query status
    pub status: QueryStatus,
    /// Timestamp when query was created
    pub created_at: i64,
}

/// Result of a climate query
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClimateResult {
    /// ID of the associated query
    pub query_id: String,
    /// Verification result
    pub result: VerificationResult,
    /// Detailed result data as JSON
    pub data: Option<String>,
    /// DID of the responder
    pub responder_did: String,
    /// Timestamp when result was created
    pub responded_at: i64,
    /// Digital signature of the result
    pub signature: Option<String>,
}

/// Cross-hApp marketplace listing for carbon credits
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MarketplaceListing {
    /// Listing ID
    pub listing_id: String,
    /// Credit ID being listed
    pub credit_id: String,
    /// Credit action hash from the carbon zome
    pub credit_action_hash: String,
    /// Project ID the credit belongs to
    pub project_id: String,
    /// Seller DID
    pub seller_did: String,
    /// Price per tonne in base currency units
    pub price_per_tonne: u64,
    /// Currency code (e.g., "USD", "EUR")
    pub currency: String,
    /// Minimum purchase in tonnes
    pub min_purchase: f64,
    /// Available tonnes for sale
    pub available_tonnes: f64,
    /// Listing expiry (Unix timestamp)
    pub expires_at: i64,
    /// Whether listing is active
    pub is_active: bool,
    /// Timestamp when listing was created
    pub created_at: i64,
}

/// Entry types for the bridge zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    ClimateQuery(ClimateQuery),
    #[entry_type(visibility = "public")]
    ClimateResult(ClimateResult),
    #[entry_type(visibility = "public")]
    MarketplaceListing(MarketplaceListing),
    Notification(CrossClusterNotification),
}

/// Link types for the bridge zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all queries
    AnchorToQueries,
    /// Query to its result
    QueryToResult,
    /// Requester to their queries
    RequesterToQueries,
    /// Anchor to marketplace listings
    AnchorToListings,
    /// Seller to their listings
    SellerToListings,
    /// Credit to listings
    CreditToListings,
    AgentToNotification,
    AllNotifications,
    NotificationSubscription,
}

/// Validate DIDs have proper format
fn validate_did(did: &str) -> ExternResult<ValidateCallbackResult> {
    if did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("DID cannot be empty".to_string()));
    }
    if !did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:' prefix".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate a ClimateQuery entry
fn validate_climate_query(query: &ClimateQuery) -> ExternResult<ValidateCallbackResult> {
    // Validate query ID
    if query.query_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query ID cannot be empty".to_string(),
        ));
    }

    // Validate requester DID
    let did_result = validate_did(&query.requester_did)?;
    if let ValidateCallbackResult::Invalid(_) = did_result {
        return Ok(did_result);
    }

    // Validate target ID
    if query.target_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Target ID cannot be empty".to_string(),
        ));
    }

    // Validate parameters JSON if present
    if let Some(ref params) = query.parameters {
        if !params.is_empty() {
            if serde_json::from_str::<serde_json::Value>(params).is_err() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Parameters must be valid JSON".to_string(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a ClimateResult entry
fn validate_climate_result(result: &ClimateResult) -> ExternResult<ValidateCallbackResult> {
    // Validate query ID
    if result.query_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query ID cannot be empty".to_string(),
        ));
    }

    // Validate responder DID
    let did_result = validate_did(&result.responder_did)?;
    if let ValidateCallbackResult::Invalid(_) = did_result {
        return Ok(did_result);
    }

    // Validate data JSON if present
    if let Some(ref data) = result.data {
        if !data.is_empty() {
            if serde_json::from_str::<serde_json::Value>(data).is_err() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Data must be valid JSON".to_string(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a MarketplaceListing entry
fn validate_marketplace_listing(listing: &MarketplaceListing) -> ExternResult<ValidateCallbackResult> {
    // Validate listing ID
    if listing.listing_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Listing ID cannot be empty".to_string(),
        ));
    }

    // Validate credit ID
    if listing.credit_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit ID cannot be empty".to_string(),
        ));
    }

    // Validate credit action hash
    if listing.credit_action_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit action hash cannot be empty".to_string(),
        ));
    }

    // Validate seller DID
    let did_result = validate_did(&listing.seller_did)?;
    if let ValidateCallbackResult::Invalid(_) = did_result {
        return Ok(did_result);
    }

    // Validate price
    if listing.price_per_tonne == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Price must be greater than zero".to_string(),
        ));
    }

    // Validate currency
    if listing.currency.is_empty() || listing.currency.len() != 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency must be a 3-letter code".to_string(),
        ));
    }

    // Validate min purchase
    if listing.min_purchase <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum purchase must be positive".to_string(),
        ));
    }

    // Validate available tonnes
    if listing.available_tonnes <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Available tonnes must be positive".to_string(),
        ));
    }

    // Min purchase cannot exceed available
    if listing.min_purchase > listing.available_tonnes {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum purchase cannot exceed available tonnes".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ClimateQuery(query) => validate_climate_query(&query),
                EntryTypes::ClimateResult(result) => validate_climate_result(&result),
                EntryTypes::MarketplaceListing(listing) => validate_marketplace_listing(&listing),
                EntryTypes::Notification(n) => {
                    mycelix_bridge_entry_types::validate_notification(&n)
                        .map(|()| ValidateCallbackResult::Valid)
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))
                }
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                    "Anchors cannot be updated".to_string(),
                )),
                EntryTypes::ClimateQuery(query) => validate_climate_query(&query),
                EntryTypes::ClimateResult(_) => Ok(ValidateCallbackResult::Invalid(
                    "Climate results cannot be updated once submitted".to_string(),
                )),
                EntryTypes::MarketplaceListing(listing) => validate_marketplace_listing(&listing),
                EntryTypes::Notification(_) => Ok(ValidateCallbackResult::Invalid(
                    "Notifications cannot be updated".to_string(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToQueries
            | LinkTypes::QueryToResult
            | LinkTypes::RequesterToQueries
            | LinkTypes::AnchorToListings
            | LinkTypes::SellerToListings
            | LinkTypes::CreditToListings
            | LinkTypes::AgentToNotification
            | LinkTypes::AllNotifications
            | LinkTypes::NotificationSubscription => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { link_type, .. } => match link_type {
            LinkTypes::QueryToResult => Ok(ValidateCallbackResult::Invalid(
                "Query-to-result links cannot be deleted".to_string(),
            )),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
