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

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }
    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn valid_market() -> Market {
        Market {
            id: "mkt-1".into(),
            name: "Richardson Farmers Market".into(),
            location_lat: 32.95,
            location_lon: -96.73,
            market_type: MarketType::Farmers,
            steward: fake_agent(),
            schedule: "Saturdays 8am-1pm".into(),
        }
    }

    #[test]
    fn valid_market_passes() {
        assert_eq!(validate_market(valid_market()).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn market_empty_id_rejected() {
        let mut m = valid_market();
        m.id = String::new();
        assert!(matches!(validate_market(m).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn market_empty_name_rejected() {
        let mut m = valid_market();
        m.name = String::new();
        assert!(matches!(validate_market(m).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn market_invalid_lat_rejected() {
        let mut m = valid_market();
        m.location_lat = -91.0;
        assert!(matches!(validate_market(m).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn market_all_types_valid() {
        for mt in [MarketType::Farmers, MarketType::CSA, MarketType::FoodBank, MarketType::CoOp] {
            let mut m = valid_market();
            m.market_type = mt;
            assert_eq!(validate_market(m).unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn valid_listing_passes() {
        let l = Listing {
            market_hash: fake_action_hash(),
            producer: fake_agent(),
            product_name: "Heirloom Tomatoes".into(),
            quantity_kg: 10.0,
            price_per_kg: 5.50,
            available_from: 1700000000,
            status: ListingStatus::Available,
        };
        assert_eq!(validate_listing(l).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn listing_empty_product_rejected() {
        let l = Listing {
            market_hash: fake_action_hash(),
            producer: fake_agent(),
            product_name: String::new(),
            quantity_kg: 10.0,
            price_per_kg: 5.50,
            available_from: 1700000000,
            status: ListingStatus::Available,
        };
        assert!(matches!(validate_listing(l).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn listing_zero_quantity_rejected() {
        let l = Listing {
            market_hash: fake_action_hash(),
            producer: fake_agent(),
            product_name: "Basil".into(),
            quantity_kg: 0.0,
            price_per_kg: 3.00,
            available_from: 1700000000,
            status: ListingStatus::Available,
        };
        assert!(matches!(validate_listing(l).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn listing_negative_price_rejected() {
        let l = Listing {
            market_hash: fake_action_hash(),
            producer: fake_agent(),
            product_name: "Basil".into(),
            quantity_kg: 5.0,
            price_per_kg: -1.0,
            available_from: 1700000000,
            status: ListingStatus::Available,
        };
        assert!(matches!(validate_listing(l).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn listing_zero_price_valid() {
        let l = Listing {
            market_hash: fake_action_hash(),
            producer: fake_agent(),
            product_name: "Donated Produce".into(),
            quantity_kg: 20.0,
            price_per_kg: 0.0,
            available_from: 1700000000,
            status: ListingStatus::Available,
        };
        assert_eq!(validate_listing(l).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn valid_order_passes() {
        let o = Order {
            listing_hash: fake_action_hash(),
            buyer: fake_agent(),
            quantity_kg: 5.0,
            status: OrderStatus::Pending,
        };
        assert_eq!(validate_order(o).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn order_zero_quantity_rejected() {
        let o = Order {
            listing_hash: fake_action_hash(),
            buyer: fake_agent(),
            quantity_kg: 0.0,
            status: OrderStatus::Pending,
        };
        assert!(matches!(validate_order(o).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
}
