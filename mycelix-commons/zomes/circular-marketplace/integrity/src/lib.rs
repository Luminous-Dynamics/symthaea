// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Circular Marketplace Integrity Zome
//! Entry types for secondary material listings, orders, and circular economy trading

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// SECONDARY MATERIAL LISTING
// ============================================================================

/// Quality grade for secondary materials
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaterialQualityGrade {
    /// Top quality, suitable for direct agricultural use or premium markets
    Premium,
    /// Standard quality, meets basic requirements
    Standard,
    /// Lower grade, suitable for industrial or bulk applications
    Industrial,
}

/// Type of secondary material available
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SecondaryMaterialType {
    /// Finished compost
    Compost,
    /// Worm castings
    Vermicompost,
    /// Anaerobic digestion output
    Digestate,
    /// Biochar from pyrolysis
    Biochar,
    /// Recycled plastic pellets/flake
    RecycledPlastic,
    /// Recycled metal
    RecycledMetal,
    /// Recycled glass cullet
    RecycledGlass,
    /// Recycled paper/cardboard fiber
    RecycledFiber,
    /// Mulch or woodchips
    Mulch,
}

/// Status of a listing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ListingStatus {
    /// Available for purchase
    Available,
    /// Reserved for a buyer (pending confirmation)
    Reserved,
    /// Sold and delivered
    Sold,
    /// Listing withdrawn by seller
    Withdrawn,
}

/// Optional nutrient profile for compost/organic listings
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NutrientInfo {
    pub nitrogen_pct: f64,
    pub phosphorus_pct: f64,
    pub potassium_pct: f64,
}

/// A listing of secondary material for sale or distribution
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SecondaryMaterialListing {
    /// Link to the facility that produced this material
    pub facility_hash: ActionHash,
    /// Optional link to the compost batch that produced it
    pub batch_hash: Option<ActionHash>,
    /// Type of material
    pub material_type: SecondaryMaterialType,
    /// Available quantity (kg)
    pub quantity_kg: f64,
    /// Quality assessment
    pub quality_grade: MaterialQualityGrade,
    /// Nutrient profile (for compost/organic types)
    pub nutrient_info: Option<NutrientInfo>,
    /// Price per kg (in smallest currency unit, 0 = free/donation)
    pub price_per_kg: f64,
    /// GPS latitude of material location
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// When the material becomes available (Unix microseconds)
    pub available_from: u64,
    /// Current listing status
    pub status: ListingStatus,
    /// Seller DID
    pub seller_did: String,
}

// ============================================================================
// ORDER
// ============================================================================

/// Status of an order
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OrderStatus {
    /// Order placed, awaiting seller confirmation
    Placed,
    /// Seller confirmed, preparing for delivery
    Confirmed,
    /// Material in transit
    InTransit,
    /// Delivered to buyer
    Delivered,
    /// Order cancelled
    Cancelled,
}

/// An order for secondary material
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SecondaryMaterialOrder {
    /// The listing being ordered from
    pub listing_hash: ActionHash,
    /// Buyer agent
    pub buyer: AgentPubKey,
    /// Quantity ordered (kg)
    pub quantity_kg: f64,
    /// Delivery GPS latitude
    pub delivery_lat: f64,
    /// Delivery GPS longitude
    pub delivery_lon: f64,
    /// Optional link to transport route for delivery
    pub delivery_route_hash: Option<ActionHash>,
    /// Current order status
    pub status: OrderStatus,
    /// When the order was placed (Unix microseconds)
    pub placed_at: u64,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    SecondaryMaterialListing(SecondaryMaterialListing),
    SecondaryMaterialOrder(SecondaryMaterialOrder),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All listings (global anchor)
    AllListings,
    /// Material type anchor to listings of that type
    TypeToListings,
    /// Listing to its orders
    ListingToOrders,
    /// Buyer agent to their orders
    BuyerToOrders,
    /// Facility to its listings
    FacilityToListings,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SecondaryMaterialListing(listing) => {
                    validate_create_listing(action, listing)
                }
                EntryTypes::SecondaryMaterialOrder(order) => {
                    validate_create_order(action, order)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SecondaryMaterialListing(_) | EntryTypes::SecondaryMaterialOrder(_) => {
                    let original = must_get_action(original_action_hash)?;
                    Ok(check_author_match(
                        original.action().author(),
                        &action.author,
                        "update",
                    ))
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            match link_type {
                LinkTypes::AllListings
                | LinkTypes::TypeToListings
                | LinkTypes::ListingToOrders
                | LinkTypes::BuyerToOrders
                | LinkTypes::FacilityToListings => {
                    if tag.0.len() > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            format!("{:?} link tag too long (max 512 bytes)", link_type),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
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
            let original_action = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original_action.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_listing(
    _action: Create,
    listing: SecondaryMaterialListing,
) -> ExternResult<ValidateCallbackResult> {
    if !listing.quantity_kg.is_finite() || listing.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a positive finite number".into(),
        ));
    }
    if !listing.price_per_kg.is_finite() || listing.price_per_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Price must be a non-negative finite number".into(),
        ));
    }
    if !listing.location_lat.is_finite()
        || listing.location_lat < -90.0
        || listing.location_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !listing.location_lon.is_finite()
        || listing.location_lon < -180.0
        || listing.location_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if listing.seller_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Seller DID cannot be empty".into(),
        ));
    }
    // Validate nutrient info if present
    if let Some(ref nutrients) = listing.nutrient_info {
        if !nutrients.nitrogen_pct.is_finite()
            || !nutrients.phosphorus_pct.is_finite()
            || !nutrients.potassium_pct.is_finite()
        {
            return Ok(ValidateCallbackResult::Invalid(
                "Nutrient values must be finite".into(),
            ));
        }
        if nutrients.nitrogen_pct < 0.0
            || nutrients.phosphorus_pct < 0.0
            || nutrients.potassium_pct < 0.0
        {
            return Ok(ValidateCallbackResult::Invalid(
                "Nutrient values must be non-negative".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_order(
    _action: Create,
    order: SecondaryMaterialOrder,
) -> ExternResult<ValidateCallbackResult> {
    if !order.quantity_kg.is_finite() || order.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Order quantity must be a positive finite number".into(),
        ));
    }
    if !order.delivery_lat.is_finite()
        || order.delivery_lat < -90.0
        || order.delivery_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Delivery latitude must be between -90 and 90".into(),
        ));
    }
    if !order.delivery_lon.is_finite()
        || order.delivery_lon < -180.0
        || order.delivery_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Delivery longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_listing() -> SecondaryMaterialListing {
        SecondaryMaterialListing {
            facility_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            batch_hash: Some(ActionHash::from_raw_36(vec![1u8; 36])),
            material_type: SecondaryMaterialType::Compost,
            quantity_kg: 500.0,
            quality_grade: MaterialQualityGrade::Standard,
            nutrient_info: Some(NutrientInfo {
                nitrogen_pct: 1.8,
                phosphorus_pct: 0.7,
                potassium_pct: 1.2,
            }),
            price_per_kg: 0.05,
            location_lat: 32.9500,
            location_lon: -96.7300,
            available_from: 1711100000_000000,
            status: ListingStatus::Available,
            seller_did: "did:example:composting_facility".to_string(),
        }
    }

    fn make_valid_order() -> SecondaryMaterialOrder {
        SecondaryMaterialOrder {
            listing_hash: ActionHash::from_raw_36(vec![2u8; 36]),
            buyer: AgentPubKey::from_raw_36(vec![3u8; 36]),
            quantity_kg: 100.0,
            delivery_lat: 32.9600,
            delivery_lon: -96.7200,
            delivery_route_hash: None,
            status: OrderStatus::Placed,
            placed_at: 1711200000_000000,
        }
    }

    #[test]
    fn test_valid_listing() {
        let listing = make_valid_listing();
        assert!(listing.quantity_kg > 0.0);
        assert!(listing.price_per_kg >= 0.0);
        assert!((-90.0..=90.0).contains(&listing.location_lat));
    }

    #[test]
    fn test_listing_zero_price_allowed() {
        let mut listing = make_valid_listing();
        listing.price_per_kg = 0.0;
        assert!(listing.price_per_kg >= 0.0, "Free/donation listings should be valid");
    }

    #[test]
    fn test_listing_negative_price_invalid() {
        let mut listing = make_valid_listing();
        listing.price_per_kg = -1.0;
        assert!(listing.price_per_kg < 0.0);
    }

    #[test]
    fn test_listing_quantity_must_be_positive() {
        let mut listing = make_valid_listing();
        listing.quantity_kg = 0.0;
        assert!(listing.quantity_kg <= 0.0);
    }

    #[test]
    fn test_valid_order() {
        let order = make_valid_order();
        assert!(order.quantity_kg > 0.0);
        assert!((-90.0..=90.0).contains(&order.delivery_lat));
    }

    #[test]
    fn test_order_quantity_must_be_positive() {
        let mut order = make_valid_order();
        order.quantity_kg = 0.0;
        assert!(order.quantity_kg <= 0.0);
    }

    #[test]
    fn test_material_types_exhaustive() {
        let types = vec![
            SecondaryMaterialType::Compost,
            SecondaryMaterialType::Vermicompost,
            SecondaryMaterialType::Digestate,
            SecondaryMaterialType::Biochar,
            SecondaryMaterialType::RecycledPlastic,
            SecondaryMaterialType::RecycledMetal,
            SecondaryMaterialType::RecycledGlass,
            SecondaryMaterialType::RecycledFiber,
            SecondaryMaterialType::Mulch,
        ];
        assert_eq!(types.len(), 9);
    }

    #[test]
    fn test_listing_status_lifecycle() {
        let statuses = vec![
            ListingStatus::Available,
            ListingStatus::Reserved,
            ListingStatus::Sold,
            ListingStatus::Withdrawn,
        ];
        assert_eq!(statuses.len(), 4);
    }

    #[test]
    fn test_order_status_lifecycle() {
        let statuses = vec![
            OrderStatus::Placed,
            OrderStatus::Confirmed,
            OrderStatus::InTransit,
            OrderStatus::Delivered,
            OrderStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_nutrient_info_valid() {
        let n = NutrientInfo {
            nitrogen_pct: 1.8,
            phosphorus_pct: 0.7,
            potassium_pct: 1.2,
        };
        assert!(n.nitrogen_pct >= 0.0);
        assert!(n.phosphorus_pct >= 0.0);
        assert!(n.potassium_pct >= 0.0);
    }

    #[test]
    fn test_nan_quantity_rejected() {
        let mut listing = make_valid_listing();
        listing.quantity_kg = f64::NAN;
        assert!(!listing.quantity_kg.is_finite());
    }

    #[test]
    fn test_inf_price_rejected() {
        let mut listing = make_valid_listing();
        listing.price_per_kg = f64::INFINITY;
        assert!(!listing.price_per_kg.is_finite());
    }
}
