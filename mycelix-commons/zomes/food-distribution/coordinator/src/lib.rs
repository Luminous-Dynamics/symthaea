//! Food Distribution Coordinator Zome
//! Business logic for markets, listings, and order fulfillment.

use food_distribution_integrity::*;
use hdk::prelude::*;

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
    let _listing = get(order.listing_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Listing not found".into())))?;

    let action_hash = create_entry(&EntryTypes::Order(order.clone()))?;
    create_link(order.listing_hash, action_hash.clone(), LinkTypes::ListingToOrder, ())?;
    create_link(order.buyer, action_hash.clone(), LinkTypes::BuyerToOrder, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created order".into())))
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
