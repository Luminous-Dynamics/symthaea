//! Food Distribution Coordinator Zome
//! Business logic for markets, listings, and order fulfillment.

use food_distribution_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};

// ============================================================================
// BRIDGE SIGNAL (for cross-domain UI notification)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub source_zome: String,
    pub payload: String,
}

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// MARKET MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn create_market(market: Market) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "create_market")?;
    let action_hash = create_entry(&EntryTypes::Market(market.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_markets".to_string())))?;
    create_link(
        anchor_hash("all_markets")?,
        action_hash.clone(),
        LinkTypes::AllMarkets,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created market".into()
    )))
}

#[hdk_extern]
pub fn get_all_markets(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_markets")?, LinkTypes::AllMarkets)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// LISTINGS
// ============================================================================

#[hdk_extern]
pub fn list_product(listing: Listing) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "list_product")?;
    let _market = get(listing.market_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Market not found".into())
    ))?;

    let action_hash = create_entry(&EntryTypes::Listing(listing.clone()))?;
    create_link(
        listing.market_hash,
        action_hash.clone(),
        LinkTypes::MarketToListing,
        (),
    )?;
    create_link(
        listing.producer,
        action_hash.clone(),
        LinkTypes::ProducerToListing,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created listing".into()
    )))
}

#[hdk_extern]
pub fn get_market_listings(market_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(market_hash, LinkTypes::MarketToListing)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn get_producer_listings(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::ProducerToListing)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// ORDERS
// ============================================================================

#[hdk_extern]
pub fn place_order(order: Order) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "place_order")?;
    let listing_record = get(order.listing_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Listing not found".into())),
    )?;

    let action_hash = create_entry(&EntryTypes::Order(order.clone()))?;
    create_link(
        order.listing_hash.clone(),
        action_hash.clone(),
        LinkTypes::ListingToOrder,
        (),
    )?;
    create_link(
        order.buyer,
        action_hash.clone(),
        LinkTypes::BuyerToOrder,
        (),
    )?;

    // Cross-domain: if the listing's market is a FoodBank, notify mutualaid
    let listing: Option<Listing> = listing_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    if let Some(listing) = listing {
        if let Some(market_record) = get(listing.market_hash, GetOptions::default())? {
            let market: Option<Market> = market_record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
            if let Some(market) = market {
                if market.market_type == MarketType::FoodBank {
                    // Best-effort: check mutualaid for matching needs
                    let _ = check_mutualaid_matching_needs();

                    // Emit bridge signal so UI can show the food bank order
                    let _ = emit_signal(&BridgeEventSignal {
                        event_type: "food_bank_order_placed".to_string(),
                        source_zome: "food_distribution".to_string(),
                        payload: format!(
                            r#"{{"order_hash":"{}","market_id":"{}","quantity_kg":{}}}"#,
                            action_hash, market.id, order.quantity_kg,
                        ),
                    });
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created order".into()
    )))
}

/// Best-effort cross-domain call to mutualaid_needs to check for matching needs.
///
/// Uses `call(CallTargetCell::Local, ...)` since mutualaid_needs is in the same
/// Commons cluster DNA. Failures are silently ignored -- the primary order
/// operation must not fail because the mutualaid zome is unavailable.
fn check_mutualaid_matching_needs() -> Option<Vec<Record>> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("mutualaid_needs"),
        FunctionName::from("get_all_needs"),
        None,
        (),
    );
    match response {
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io.decode::<Vec<Record>>().ok(),
        _ => None,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AllergenSearchInput {
    pub market_hash: ActionHash,
    pub exclude_allergens: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateOrderStatusInput {
    pub order_hash: ActionHash,
    pub new_status: OrderStatus,
}

#[hdk_extern]
pub fn fulfill_order(input: UpdateOrderStatusInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "fulfill_order")?;
    let record = get(input.order_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Order not found".into())))?;
    let mut order: Order = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid order entry".into()
        )))?;

    order.status = OrderStatus::Fulfilled;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::Order(order))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated order".into()
    )))
}

#[hdk_extern]
pub fn cancel_order(order_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "cancel_order")?;
    let record = get(order_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Order not found".into())))?;
    let mut order: Order = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid order entry".into()
        )))?;

    order.status = OrderStatus::Cancelled;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::Order(order))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated order".into()
    )))
}

#[hdk_extern]
pub fn get_my_orders(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::BuyerToOrder)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn search_allergen_safe(input: AllergenSearchInput) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(input.market_hash, LinkTypes::MarketToListing)?,
        GetStrategy::default(),
    )?;
    let all_records = records_from_links(links)?;
    let exclude_lower: Vec<String> = input
        .exclude_allergens
        .iter()
        .map(|a| a.to_lowercase())
        .collect();
    let mut safe = Vec::new();
    for record in all_records {
        let listing: Option<Listing> = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
        if let Some(listing) = listing {
            let has_excluded = listing
                .allergen_flags
                .iter()
                .any(|f| exclude_lower.contains(&f.to_lowercase()));
            if !has_excluded {
                safe.push(record);
            }
        }
    }
    Ok(safe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_order_status_input_serde_fulfilled() {
        let input = UpdateOrderStatusInput {
            order_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: OrderStatus::Fulfilled,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOrderStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, OrderStatus::Fulfilled);
    }

    #[test]
    fn update_order_status_input_serde_cancelled() {
        let input = UpdateOrderStatusInput {
            order_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: OrderStatus::Cancelled,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOrderStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, OrderStatus::Cancelled);
    }

    #[test]
    fn bridge_event_signal_food_bank_serde() {
        let signal = BridgeEventSignal {
            event_type: "food_bank_order_placed".to_string(),
            source_zome: "food_distribution".to_string(),
            payload: r#"{"order_hash":"abc","market_id":"m1","quantity_kg":10.0}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type, "food_bank_order_placed");
        assert!(decoded.payload.contains("market_id"));
    }

    #[test]
    fn order_status_all_variants_serialize() {
        let statuses = vec![
            OrderStatus::Pending,
            OrderStatus::Confirmed,
            OrderStatus::Fulfilled,
            OrderStatus::Cancelled,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: OrderStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn market_type_all_variants_serialize() {
        let types = vec![
            MarketType::Farmers,
            MarketType::CSA,
            MarketType::FoodBank,
            MarketType::CoOp,
        ];
        for mt in types {
            let json = serde_json::to_string(&mt).unwrap();
            let decoded: MarketType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mt);
        }
    }

    // ========================================================================
    // ListingStatus enum serde roundtrip
    // ========================================================================

    #[test]
    fn listing_status_all_variants_serde_roundtrip() {
        let variants = vec![
            ListingStatus::Available,
            ListingStatus::Reserved,
            ListingStatus::Sold,
            ListingStatus::Expired,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ListingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Market struct serde roundtrip
    // ========================================================================

    #[test]
    fn market_serde_roundtrip() {
        let market = Market {
            id: "mkt-7".to_string(),
            name: "Richardson Farmers Market".to_string(),
            location_lat: 32.95,
            location_lon: -96.73,
            market_type: MarketType::Farmers,
            steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
            schedule: "Saturdays 8am-1pm".to_string(),
        };
        let json = serde_json::to_string(&market).unwrap();
        let decoded: Market = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "mkt-7");
        assert_eq!(decoded.name, "Richardson Farmers Market");
        assert_eq!(decoded.location_lat, 32.95);
        assert_eq!(decoded.location_lon, -96.73);
        assert_eq!(decoded.market_type, MarketType::Farmers);
        assert_eq!(decoded.schedule, "Saturdays 8am-1pm");
    }

    #[test]
    fn market_serde_all_types() {
        for mt in [
            MarketType::Farmers,
            MarketType::CSA,
            MarketType::FoodBank,
            MarketType::CoOp,
        ] {
            let market = Market {
                id: "mkt-types".to_string(),
                name: "Test Market".to_string(),
                location_lat: 0.0,
                location_lon: 0.0,
                market_type: mt.clone(),
                steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
                schedule: "Daily".to_string(),
            };
            let json = serde_json::to_string(&market).unwrap();
            let decoded: Market = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.market_type, mt);
        }
    }

    // ========================================================================
    // Listing struct serde roundtrip
    // ========================================================================

    #[test]
    fn listing_serde_roundtrip_available() {
        let listing = Listing {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            product_name: "Heirloom Tomatoes".to_string(),
            quantity_kg: 15.0,
            price_per_kg: 5.50,
            available_from: 1700000000,
            status: ListingStatus::Available,
            allergen_flags: vec![],
            organic: false,
            cultural_markers: vec![],
        };
        let json = serde_json::to_string(&listing).unwrap();
        let decoded: Listing = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.product_name, "Heirloom Tomatoes");
        assert_eq!(decoded.quantity_kg, 15.0);
        assert_eq!(decoded.price_per_kg, 5.50);
        assert_eq!(decoded.available_from, 1700000000);
        assert_eq!(decoded.status, ListingStatus::Available);
    }

    #[test]
    fn listing_serde_all_statuses() {
        for status in [
            ListingStatus::Available,
            ListingStatus::Reserved,
            ListingStatus::Sold,
            ListingStatus::Expired,
        ] {
            let listing = Listing {
                market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
                product_name: "Basil".to_string(),
                quantity_kg: 2.0,
                price_per_kg: 8.00,
                available_from: 1700000000,
                status: status.clone(),
                allergen_flags: vec![],
                organic: false,
                cultural_markers: vec![],
            };
            let json = serde_json::to_string(&listing).unwrap();
            let decoded: Listing = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    #[test]
    fn listing_serde_zero_price_donation() {
        let listing = Listing {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            product_name: "Donated Zucchini".to_string(),
            quantity_kg: 50.0,
            price_per_kg: 0.0,
            available_from: 1700000000,
            status: ListingStatus::Available,
            allergen_flags: vec![],
            organic: false,
            cultural_markers: vec![],
        };
        let json = serde_json::to_string(&listing).unwrap();
        let decoded: Listing = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.price_per_kg, 0.0);
        assert_eq!(decoded.product_name, "Donated Zucchini");
    }

    // ========================================================================
    // Order struct serde roundtrip
    // ========================================================================

    #[test]
    fn order_serde_roundtrip_pending() {
        let order = Order {
            listing_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            buyer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            quantity_kg: 5.0,
            status: OrderStatus::Pending,
        };
        let json = serde_json::to_string(&order).unwrap();
        let decoded: Order = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity_kg, 5.0);
        assert_eq!(decoded.status, OrderStatus::Pending);
    }

    #[test]
    fn order_serde_all_statuses() {
        for status in [
            OrderStatus::Pending,
            OrderStatus::Confirmed,
            OrderStatus::Fulfilled,
            OrderStatus::Cancelled,
        ] {
            let order = Order {
                listing_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                buyer: AgentPubKey::from_raw_36(vec![0xab; 36]),
                quantity_kg: 3.0,
                status: status.clone(),
            };
            let json = serde_json::to_string(&order).unwrap();
            let decoded: Order = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ========================================================================
    // Boundary condition tests
    // ========================================================================

    #[test]
    fn bridge_event_signal_empty_payload() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            source_zome: "food_distribution".to_string(),
            payload: "".to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payload, "");
    }

    #[test]
    fn bridge_event_signal_very_long_payload() {
        let long_payload = "x".repeat(10_000);
        let signal = BridgeEventSignal {
            event_type: "large_event".to_string(),
            source_zome: "food_distribution".to_string(),
            payload: long_payload.clone(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payload.len(), 10_000);
    }

    #[test]
    fn market_serde_empty_schedule() {
        let market = Market {
            id: "mkt-empty-sched".to_string(),
            name: "No Schedule Market".to_string(),
            location_lat: 0.0,
            location_lon: 0.0,
            market_type: MarketType::CoOp,
            steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
            schedule: "".to_string(),
        };
        let json = serde_json::to_string(&market).unwrap();
        let decoded: Market = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.schedule, "");
    }

    #[test]
    fn market_serde_extreme_coordinates() {
        let market = Market {
            id: "mkt-extreme".to_string(),
            name: "Extreme Coords".to_string(),
            location_lat: 90.0,
            location_lon: -180.0,
            market_type: MarketType::Farmers,
            steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
            schedule: "Always".to_string(),
        };
        let json = serde_json::to_string(&market).unwrap();
        let decoded: Market = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location_lat, 90.0);
        assert_eq!(decoded.location_lon, -180.0);
    }

    #[test]
    fn market_serde_zero_coordinates() {
        let market = Market {
            id: "mkt-zero".to_string(),
            name: "Origin Market".to_string(),
            location_lat: 0.0,
            location_lon: 0.0,
            market_type: MarketType::FoodBank,
            steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
            schedule: "Weekdays".to_string(),
        };
        let json = serde_json::to_string(&market).unwrap();
        let decoded: Market = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location_lat, 0.0);
        assert_eq!(decoded.location_lon, 0.0);
    }

    #[test]
    fn listing_serde_very_large_quantity() {
        let listing = Listing {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            product_name: "Bulk Wheat".to_string(),
            quantity_kg: 999_999.99,
            price_per_kg: 0.50,
            available_from: u64::MAX,
            status: ListingStatus::Available,
            allergen_flags: vec![],
            organic: false,
            cultural_markers: vec![],
        };
        let json = serde_json::to_string(&listing).unwrap();
        let decoded: Listing = serde_json::from_str(&json).unwrap();
        assert!((decoded.quantity_kg - 999_999.99).abs() < 0.001);
        assert_eq!(decoded.available_from, u64::MAX);
    }

    #[test]
    fn listing_serde_empty_product_name() {
        let listing = Listing {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            product_name: "".to_string(),
            quantity_kg: 1.0,
            price_per_kg: 1.0,
            available_from: 0,
            status: ListingStatus::Available,
            allergen_flags: vec![],
            organic: false,
            cultural_markers: vec![],
        };
        let json = serde_json::to_string(&listing).unwrap();
        let decoded: Listing = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.product_name, "");
    }

    #[test]
    fn order_serde_zero_quantity() {
        let order = Order {
            listing_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            buyer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            quantity_kg: 0.0,
            status: OrderStatus::Pending,
        };
        let json = serde_json::to_string(&order).unwrap();
        let decoded: Order = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity_kg, 0.0);
    }

    #[test]
    fn order_serde_very_small_quantity() {
        let order = Order {
            listing_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            buyer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            quantity_kg: 0.001,
            status: OrderStatus::Confirmed,
        };
        let json = serde_json::to_string(&order).unwrap();
        let decoded: Order = serde_json::from_str(&json).unwrap();
        assert!((decoded.quantity_kg - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn update_order_status_input_serde_pending() {
        let input = UpdateOrderStatusInput {
            order_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: OrderStatus::Pending,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOrderStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, OrderStatus::Pending);
    }

    #[test]
    fn update_order_status_input_serde_confirmed() {
        let input = UpdateOrderStatusInput {
            order_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: OrderStatus::Confirmed,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateOrderStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, OrderStatus::Confirmed);
    }

    #[test]
    fn bridge_event_signal_clone() {
        let signal = BridgeEventSignal {
            event_type: "clone_test".to_string(),
            source_zome: "food_distribution".to_string(),
            payload: r#"{"key":"value"}"#.to_string(),
        };
        let cloned = signal.clone();
        assert_eq!(signal.event_type, cloned.event_type);
        assert_eq!(signal.source_zome, cloned.source_zome);
        assert_eq!(signal.payload, cloned.payload);
    }

    #[test]
    fn market_serde_field_access() {
        let market = Market {
            id: "mkt-access".to_string(),
            name: "Field Access Market".to_string(),
            location_lat: 51.5074,
            location_lon: -0.1278,
            market_type: MarketType::CSA,
            steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
            schedule: "Tue/Thu 10am-2pm".to_string(),
        };
        let json = serde_json::to_string(&market).unwrap();
        let decoded: Market = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "mkt-access");
        assert_eq!(decoded.name, "Field Access Market");
        assert!((decoded.location_lat - 51.5074).abs() < f64::EPSILON);
        assert!((decoded.location_lon - (-0.1278)).abs() < f64::EPSILON);
        assert_eq!(decoded.market_type, MarketType::CSA);
        assert_eq!(decoded.schedule, "Tue/Thu 10am-2pm");
    }

    #[test]
    fn listing_serde_available_from_zero() {
        let listing = Listing {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            product_name: "Epoch Start Produce".to_string(),
            quantity_kg: 10.0,
            price_per_kg: 3.0,
            available_from: 0,
            status: ListingStatus::Reserved,
            allergen_flags: vec![],
            organic: false,
            cultural_markers: vec![],
        };
        let json = serde_json::to_string(&listing).unwrap();
        let decoded: Listing = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.available_from, 0);
        assert_eq!(decoded.status, ListingStatus::Reserved);
    }

    // ========================================================================
    // AllergenSearchInput serde roundtrip
    // ========================================================================

    #[test]
    fn allergen_search_input_serde_roundtrip() {
        let input = AllergenSearchInput {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            exclude_allergens: vec!["gluten".to_string(), "dairy".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AllergenSearchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.exclude_allergens, vec!["gluten", "dairy"]);
    }

    #[test]
    fn allergen_search_input_serde_empty_exclusions() {
        let input = AllergenSearchInput {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            exclude_allergens: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AllergenSearchInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.exclude_allergens.is_empty());
    }

    #[test]
    fn listing_serde_with_all_new_fields_populated() {
        let listing = Listing {
            market_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            producer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            product_name: "Organic Kimchi".to_string(),
            quantity_kg: 8.0,
            price_per_kg: 12.00,
            available_from: 1700000000,
            status: ListingStatus::Available,
            allergen_flags: vec!["soy".to_string(), "shellfish".to_string()],
            organic: true,
            cultural_markers: vec!["korean".to_string(), "fermented".to_string()],
        };
        let json = serde_json::to_string(&listing).unwrap();
        let decoded: Listing = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.allergen_flags, vec!["soy", "shellfish"]);
        assert!(decoded.organic);
        assert_eq!(decoded.cultural_markers, vec!["korean", "fermented"]);
        assert_eq!(decoded.product_name, "Organic Kimchi");
    }
}
