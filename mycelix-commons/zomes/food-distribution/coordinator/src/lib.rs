//! Food Distribution Coordinator Zome
//! Business logic for markets, listings, and order fulfillment.

use food_distribution_integrity::*;
use hdk::prelude::*;

// ============================================================================
// BRIDGE SIGNAL (for cross-domain UI notification)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub source_zome: String,
    pub payload: String,
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
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
    let action_hash = create_entry(&EntryTypes::Market(market.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_markets".to_string())))?;
    create_link(anchor_hash("all_markets")?, action_hash.clone(), LinkTypes::AllMarkets, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created market".into())))
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
    let _market = get(listing.market_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;

    let action_hash = create_entry(&EntryTypes::Listing(listing.clone()))?;
    create_link(listing.market_hash, action_hash.clone(), LinkTypes::MarketToListing, ())?;
    create_link(listing.producer, action_hash.clone(), LinkTypes::ProducerToListing, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created listing".into())))
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
    let listing_record = get(order.listing_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Listing not found".into())))?;

    let action_hash = create_entry(&EntryTypes::Order(order.clone()))?;
    create_link(order.listing_hash.clone(), action_hash.clone(), LinkTypes::ListingToOrder, ())?;
    create_link(order.buyer, action_hash.clone(), LinkTypes::BuyerToOrder, ())?;

    // Cross-domain: if the listing's market is a FoodBank, notify mutualaid
    let listing: Option<Listing> = listing_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    if let Some(listing) = listing {
        if let Some(market_record) = get(listing.market_hash, GetOptions::default())? {
            let market: Option<Market> = market_record.entry()
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created order".into())))
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
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            extern_io.decode::<Vec<Record>>().ok()
        }
        _ => None,
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateOrderStatusInput {
    pub order_hash: ActionHash,
    pub new_status: OrderStatus,
}

#[hdk_extern]
pub fn fulfill_order(input: UpdateOrderStatusInput) -> ExternResult<Record> {
    let record = get(input.order_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Order not found".into())))?;
    let mut order: Order = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid order entry".into())))?;

    order.status = OrderStatus::Fulfilled;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::Order(order))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated order".into())))
}

#[hdk_extern]
pub fn cancel_order(order_hash: ActionHash) -> ExternResult<Record> {
    let record = get(order_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Order not found".into())))?;
    let mut order: Order = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid order entry".into())))?;

    order.status = OrderStatus::Cancelled;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::Order(order))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated order".into())))
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
}
