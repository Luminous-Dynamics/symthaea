// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Circular Marketplace Coordinator Zome
//! Secondary material listings, demand matching, and order management

use hdk::prelude::*;
use mycelix_bridge_common::civic_requirement_basic;
use mycelix_zome_helpers::records_from_links;
use circular_marketplace_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}


// ============================================================================
// LISTINGS
// ============================================================================

/// List a secondary material for sale or distribution
#[hdk_extern]
pub fn list_secondary_material(listing: SecondaryMaterialListing) -> ExternResult<Record> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "list_secondary_material")?;

    if !listing.quantity_kg.is_finite() || listing.quantity_kg <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Quantity must be a positive finite number".into()
        )));
    }
    if !listing.price_per_kg.is_finite() || listing.price_per_kg < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Price must be a non-negative finite number".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::SecondaryMaterialListing(listing.clone()))?;

    // Link to all listings anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_listings".to_string())))?;
    create_link(
        anchor_hash("all_listings")?,
        action_hash.clone(),
        LinkTypes::AllListings,
        (),
    )?;

    // Link to material type anchor
    let type_anchor = format!("material_type:{:?}", listing.material_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::TypeToListings,
        (),
    )?;

    // Link facility to listing
    create_link(
        listing.facility_hash.clone(),
        action_hash.clone(),
        LinkTypes::FacilityToListings,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "secondary_material_listed",
        "source_zome": "circular_marketplace",
        "payload": {
            "material_type": format!("{:?}", listing.material_type),
            "quantity_kg": listing.quantity_kg,
            "quality_grade": format!("{:?}", listing.quality_grade),
            "price_per_kg": listing.price_per_kg,
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get a listing by hash
#[hdk_extern]
pub fn get_listing(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all listings
#[hdk_extern]
pub fn get_all_listings(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_listings")?, LinkTypes::AllListings)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get listings by material type
#[hdk_extern]
pub fn get_listings_by_type(material_type: SecondaryMaterialType) -> ExternResult<Vec<Record>> {
    let type_anchor = format!("material_type:{:?}", material_type);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::TypeToListings)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get listings from a facility
#[hdk_extern]
pub fn get_facility_listings(facility_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(facility_hash, LinkTypes::FacilityToListings)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// DEMAND MATCHING
// ============================================================================

/// Input for finding matching listings
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DemandMatchInput {
    /// What type of material is needed
    pub material_type: SecondaryMaterialType,
    /// Minimum quantity needed (kg)
    pub min_quantity_kg: f64,
    /// Minimum acceptable quality
    pub min_quality: MaterialQualityGrade,
    /// Buyer GPS latitude (for proximity scoring)
    pub buyer_lat: f64,
    /// Buyer GPS longitude
    pub buyer_lon: f64,
    /// Maximum distance in km (0 = no limit)
    pub max_distance_km: f64,
}

/// A matched listing with score
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListingMatch {
    pub listing_hash: ActionHash,
    pub material_type: String,
    pub quantity_kg: f64,
    pub quality_grade: String,
    pub price_per_kg: f64,
    pub distance_km: f64,
    pub match_score: f32,
}

/// Find listings matching a demand specification
#[hdk_extern]
pub fn find_matching_listings(input: DemandMatchInput) -> ExternResult<Vec<ListingMatch>> {
    let type_anchor = format!("material_type:{:?}", input.material_type);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::TypeToListings)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    let mut matches: Vec<ListingMatch> = Vec::new();

    for record in records {
        let listing: SecondaryMaterialListing = match record.entry().to_app_option() {
            Ok(Some(l)) => l,
            _ => continue,
        };

        // Skip non-available listings
        if listing.status != ListingStatus::Available {
            continue;
        }

        // Quantity filter
        if listing.quantity_kg < input.min_quantity_kg {
            continue;
        }

        // Quality filter
        if !quality_meets_minimum(&listing.quality_grade, &input.min_quality) {
            continue;
        }

        // Distance calculation
        let distance = haversine_km(
            input.buyer_lat,
            input.buyer_lon,
            listing.location_lat,
            listing.location_lon,
        );

        // Distance filter
        if input.max_distance_km > 0.0 && distance > input.max_distance_km {
            continue;
        }

        // Compute match score (lower price + closer = higher score)
        let proximity_score = if input.max_distance_km > 0.0 {
            (1.0 - distance / input.max_distance_km).max(0.0) as f32
        } else {
            (1.0 - distance / 200.0).max(0.0) as f32 // default 200km max
        };

        let quality_score = match listing.quality_grade {
            MaterialQualityGrade::Premium => 1.0_f32,
            MaterialQualityGrade::Standard => 0.7,
            MaterialQualityGrade::Industrial => 0.4,
        };

        // 50% proximity + 30% quality + 20% quantity surplus
        let quantity_surplus = ((listing.quantity_kg - input.min_quantity_kg)
            / input.min_quantity_kg)
            .min(1.0) as f32;
        let match_score = 0.50 * proximity_score + 0.30 * quality_score + 0.20 * quantity_surplus;

        matches.push(ListingMatch {
            listing_hash: record.action_address().clone(),
            material_type: format!("{:?}", listing.material_type),
            quantity_kg: listing.quantity_kg,
            quality_grade: format!("{:?}", listing.quality_grade),
            price_per_kg: listing.price_per_kg,
            distance_km: distance,
            match_score,
        });
    }

    // Sort by score descending
    matches.sort_by(|a, b| b.match_score.partial_cmp(&a.match_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(matches)
}

/// Check if a quality grade meets the minimum requirement
fn quality_meets_minimum(actual: &MaterialQualityGrade, minimum: &MaterialQualityGrade) -> bool {
    let grade_to_num = |g: &MaterialQualityGrade| -> u8 {
        match g {
            MaterialQualityGrade::Premium => 2,
            MaterialQualityGrade::Standard => 1,
            MaterialQualityGrade::Industrial => 0,
        }
    };
    grade_to_num(actual) >= grade_to_num(minimum)
}

/// Haversine distance between two GPS points in kilometers
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

// ============================================================================
// ORDERS
// ============================================================================

/// Place an order for secondary material
#[hdk_extern]
pub fn place_order(order: SecondaryMaterialOrder) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "place_order")?;

    if !order.quantity_kg.is_finite() || order.quantity_kg <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Order quantity must be a positive finite number".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::SecondaryMaterialOrder(order.clone()))?;

    // Link listing to order
    create_link(
        order.listing_hash.clone(),
        action_hash.clone(),
        LinkTypes::ListingToOrders,
        (),
    )?;

    // Link buyer to order
    create_link(
        order.buyer.clone(),
        action_hash.clone(),
        LinkTypes::BuyerToOrders,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "order_placed",
        "source_zome": "circular_marketplace",
        "payload": {
            "quantity_kg": order.quantity_kg,
            "status": format!("{:?}", order.status),
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get an order by hash
#[hdk_extern]
pub fn get_order(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get orders for a listing
#[hdk_extern]
pub fn get_listing_orders(listing_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(listing_hash, LinkTypes::ListingToOrders)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CROSS-ZOME: FOOD PRODUCTION INTEGRATION
// ============================================================================

/// Result of querying food-production for plots that could use compost.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompostDemandMatch {
    /// Food-production plot action hash
    pub plot_hash: String,
    /// Listing that could supply this plot
    pub listing_hash: ActionHash,
    /// Material type (should be Compost/Vermicompost/Digestate)
    pub material_type: String,
    /// Available quantity (kg)
    pub available_kg: f64,
}

/// Find food-production plots that could benefit from available compost listings.
///
/// Calls `food_production::get_all_plots` via `CallTargetCell::Local` to discover
/// plots, then matches them against available Compost/Vermicompost listings.
/// This enables the "back to farm" closed loop.
#[hdk_extern]
pub fn find_plots_needing_compost(_: ()) -> ExternResult<Vec<CompostDemandMatch>> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "find_plots_needing_compost")?;

    // Query food-production for all plots via cross-zome call
    let plots_response = call(
        CallTargetCell::Local,
        ZomeName::from("food_production"),
        FunctionName::from("get_all_plots"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {}", e))))?,
    );

    let plot_hashes: Vec<String> = match plots_response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            // Decode as Vec<Record> and extract action hashes
            let records: Vec<serde_json::Value> = extern_io
                .decode()
                .unwrap_or_default();
            records
                .iter()
                .filter_map(|r| {
                    r.get("signed_action")
                        .and_then(|sa| sa.get("hashed"))
                        .and_then(|h| h.get("hash"))
                        .and_then(|h| h.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        }
        _ => {
            // food_production zome not available — return empty
            return Ok(vec![]);
        }
    };

    if plot_hashes.is_empty() {
        return Ok(vec![]);
    }

    // Get all compost-type listings
    let compost_types = [
        SecondaryMaterialType::Compost,
        SecondaryMaterialType::Vermicompost,
        SecondaryMaterialType::Digestate,
    ];

    let mut matches: Vec<CompostDemandMatch> = Vec::new();

    for mat_type in &compost_types {
        let type_anchor = format!("material_type:{:?}", mat_type);
        let links = get_links(
            LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::TypeToListings)?,
            GetStrategy::default(),
        )?;
        let records = records_from_links(links)?;

        for record in &records {
            let listing: SecondaryMaterialListing = match record.entry().to_app_option() {
                Ok(Some(l)) => l,
                _ => continue,
            };

            if listing.status != ListingStatus::Available {
                continue;
            }

            // Match each plot with available listings
            for plot_hash in &plot_hashes {
                matches.push(CompostDemandMatch {
                    plot_hash: plot_hash.clone(),
                    listing_hash: record.action_address().clone(),
                    material_type: format!("{:?}", mat_type),
                    available_kg: listing.quantity_kg,
                });
            }
        }
    }

    Ok(matches)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_meets_minimum() {
        assert!(quality_meets_minimum(
            &MaterialQualityGrade::Premium,
            &MaterialQualityGrade::Standard,
        ));
        assert!(quality_meets_minimum(
            &MaterialQualityGrade::Standard,
            &MaterialQualityGrade::Standard,
        ));
        assert!(!quality_meets_minimum(
            &MaterialQualityGrade::Industrial,
            &MaterialQualityGrade::Standard,
        ));
        assert!(quality_meets_minimum(
            &MaterialQualityGrade::Industrial,
            &MaterialQualityGrade::Industrial,
        ));
        assert!(quality_meets_minimum(
            &MaterialQualityGrade::Premium,
            &MaterialQualityGrade::Industrial,
        ));
    }

    #[test]
    fn test_haversine_same_point() {
        let d = haversine_km(32.95, -96.73, 32.95, -96.73);
        assert!(d < 0.001);
    }

    #[test]
    fn test_haversine_short_distance() {
        // ~1.5 km apart
        let d = haversine_km(32.9500, -96.7300, 32.9600, -96.7200);
        assert!(d > 0.5 && d < 5.0, "Expected ~1.5km, got {}", d);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_quality() -> impl Strategy<Value = MaterialQualityGrade> {
        prop_oneof![
            Just(MaterialQualityGrade::Premium),
            Just(MaterialQualityGrade::Standard),
            Just(MaterialQualityGrade::Industrial),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// Quality filtering is transitive: if A meets B and B meets C, then A meets C.
        #[test]
        fn quality_filter_transitive(
            a in arb_quality(),
            b in arb_quality(),
            c in arb_quality(),
        ) {
            if quality_meets_minimum(&a, &b) && quality_meets_minimum(&b, &c) {
                prop_assert!(quality_meets_minimum(&a, &c),
                    "Transitivity violated: {:?} meets {:?} and {:?} meets {:?} but {:?} doesn't meet {:?}",
                    a, b, b, c, a, c);
            }
        }

        /// Quality filtering is reflexive: every grade meets itself.
        #[test]
        fn quality_filter_reflexive(grade in arb_quality()) {
            prop_assert!(quality_meets_minimum(&grade, &grade),
                "{:?} should meet itself", grade);
        }

        /// Premium meets all minimums.
        #[test]
        fn premium_meets_all(minimum in arb_quality()) {
            prop_assert!(quality_meets_minimum(&MaterialQualityGrade::Premium, &minimum),
                "Premium should meet {:?}", minimum);
        }

        /// Haversine distance is non-negative and symmetric.
        #[test]
        fn haversine_non_negative_symmetric(
            lat1 in -90.0..=90.0_f64,
            lon1 in -180.0..=180.0_f64,
            lat2 in -90.0..=90.0_f64,
            lon2 in -180.0..=180.0_f64,
        ) {
            let d1 = haversine_km(lat1, lon1, lat2, lon2);
            let d2 = haversine_km(lat2, lon2, lat1, lon1);
            prop_assert!(d1 >= 0.0, "Distance must be non-negative: {}", d1);
            prop_assert!(d1.is_finite(), "Distance must be finite");
            prop_assert!((d1 - d2).abs() < 0.001,
                "Distance must be symmetric: {} vs {}", d1, d2);
        }

        /// Same point has zero distance.
        #[test]
        fn haversine_same_point_zero(
            lat in -90.0..=90.0_f64,
            lon in -180.0..=180.0_f64,
        ) {
            let d = haversine_km(lat, lon, lat, lon);
            prop_assert!(d < 0.001, "Same point should have ~0 distance: {}", d);
        }
    }
}
