//! Food Distribution Integrity Zome
//! Entry types and validation for local food markets, listings, and orders.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// MARKET
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketType {
    Farmers,
    CSA,
    FoodBank,
    CoOp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Market {
    pub id: String,
    pub name: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub market_type: MarketType,
    pub steward: AgentPubKey,
    pub schedule: String,
}

// ============================================================================
// LISTING
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ListingStatus {
    Available,
    Reserved,
    Sold,
    Expired,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Listing {
    pub market_hash: ActionHash,
    pub producer: AgentPubKey,
    pub product_name: String,
    pub quantity_kg: f64,
    pub price_per_kg: f64,
    pub available_from: u64,
    pub status: ListingStatus,
}

// ============================================================================
// ORDER
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OrderStatus {
    Pending,
    Confirmed,
    Fulfilled,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Order {
    pub listing_hash: ActionHash,
    pub buyer: AgentPubKey,
    pub quantity_kg: f64,
    pub status: OrderStatus,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Market(Market),
    Listing(Listing),
    Order(Order),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllMarkets,
    MarketToListing,
    ProducerToListing,
    BuyerToOrder,
    ListingToOrder,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Market(m) => validate_market(m),
                EntryTypes::Listing(l) => validate_listing(l),
                EntryTypes::Order(o) => validate_order(o),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Market(m) => validate_market(m),
                EntryTypes::Listing(l) => validate_listing(l),
                EntryTypes::Order(o) => validate_order(o),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_market(m: Market) -> ExternResult<ValidateCallbackResult> {
    if m.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Market ID cannot be empty".into()));
    }
    if m.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Market name cannot be empty".into()));
    }
    if m.location_lat < -90.0 || m.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if m.location_lon < -180.0 || m.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_listing(l: Listing) -> ExternResult<ValidateCallbackResult> {
    if l.product_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Product name cannot be empty".into()));
    }
    if l.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Quantity must be positive".into()));
    }
    if l.price_per_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Price cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_order(o: Order) -> ExternResult<ValidateCallbackResult> {
    if o.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Order quantity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}
