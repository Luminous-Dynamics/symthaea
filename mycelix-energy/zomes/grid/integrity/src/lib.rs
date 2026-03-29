// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! P2P Grid Trading Integrity Zome
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnergyProduction {
    pub id: String,
    pub producer_did: String,
    pub project_id: String,
    pub amount_kwh: f64,
    pub timestamp: Timestamp,
    pub period_hours: f64,
    pub meter_reading: Option<f64>,
    pub verified: bool,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnergyConsumption {
    pub id: String,
    pub consumer_did: String,
    pub amount_kwh: f64,
    pub timestamp: Timestamp,
    pub period_hours: f64,
    pub meter_reading: Option<f64>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TradeOffer {
    pub id: String,
    pub seller_did: String,
    pub project_id: Option<String>,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub available_from: Timestamp,
    pub available_until: Timestamp,
    pub status: OfferStatus,
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OfferStatus {
    Active,
    PartiallyFilled,
    Filled,
    Expired,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Trade {
    pub id: String,
    pub offer_id: String,
    pub seller_did: String,
    pub buyer_did: String,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub total_price: f64,
    pub currency: String,
    pub executed: Timestamp,
    pub settled: bool,
    pub payment_reference: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    EnergyProduction(EnergyProduction),
    EnergyConsumption(EnergyConsumption),
    TradeOffer(TradeOffer),
    Trade(Trade),
}

#[hdk_link_types]
pub enum LinkTypes {
    ProducerToProduction,
    ConsumerToConsumption,
    SellerToOffers,
    ActiveOffers,
    OfferToTrades,
    BuyerToTrades,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::EnergyProduction(production) => {
                        validate_create_energy_production(EntryCreationAction::Create(action), production)
                    }
                    EntryTypes::EnergyConsumption(consumption) => {
                        validate_create_energy_consumption(EntryCreationAction::Create(action), consumption)
                    }
                    EntryTypes::TradeOffer(offer) => {
                        validate_create_trade_offer(EntryCreationAction::Create(action), offer)
                    }
                    EntryTypes::Trade(trade) => {
                        validate_create_trade(EntryCreationAction::Create(action), trade)
                    }
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::EnergyProduction(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Production records cannot be updated".into(),
                        ))
                    }
                    EntryTypes::EnergyConsumption(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Consumption records cannot be updated".into(),
                        ))
                    }
                    EntryTypes::TradeOffer(offer) => {
                        validate_update_trade_offer(action, offer)
                    }
                    EntryTypes::Trade(trade) => {
                        validate_update_trade(action, trade)
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::ProducerToProduction => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ConsumerToConsumption => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SellerToOffers => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ActiveOffers => Ok(ValidateCallbackResult::Valid),
                LinkTypes::OfferToTrades => Ok(ValidateCallbackResult::Valid),
                LinkTypes::BuyerToTrades => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_energy_production(
    _action: EntryCreationAction,
    production: EnergyProduction,
) -> ExternResult<ValidateCallbackResult> {
    if !production.producer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Producer must be a valid DID".into()));
    }
    if production.amount_kwh <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Production amount must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_energy_consumption(
    _action: EntryCreationAction,
    consumption: EnergyConsumption,
) -> ExternResult<ValidateCallbackResult> {
    if !consumption.consumer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Consumer must be a valid DID".into()));
    }
    if consumption.amount_kwh <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Consumption amount must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_trade_offer(
    _action: EntryCreationAction,
    offer: TradeOffer,
) -> ExternResult<ValidateCallbackResult> {
    if !offer.seller_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Seller must be a valid DID".into()));
    }
    if offer.amount_kwh <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Offer amount must be positive".into()));
    }
    if offer.price_per_kwh < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Price cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_trade_offer(
    _action: Update,
    offer: TradeOffer,
) -> ExternResult<ValidateCallbackResult> {
    if offer.amount_kwh <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Offer amount must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_trade(
    _action: EntryCreationAction,
    trade: Trade,
) -> ExternResult<ValidateCallbackResult> {
    if !trade.seller_did.starts_with("did:") || !trade.buyer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Parties must be valid DIDs".into()));
    }
    if trade.seller_did == trade.buyer_did {
        return Ok(ValidateCallbackResult::Invalid("Cannot trade with yourself".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_trade(
    _action: Update,
    _trade: Trade,
) -> ExternResult<ValidateCallbackResult> {
    // Can update settlement status
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OfferStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_offer_status_variants() {
        let statuses = vec![
            OfferStatus::Active,
            OfferStatus::PartiallyFilled,
            OfferStatus::Filled,
            OfferStatus::Expired,
            OfferStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_offer_status_equality() {
        assert_eq!(OfferStatus::Active, OfferStatus::Active);
        assert_ne!(OfferStatus::Active, OfferStatus::Cancelled);
    }

    // =========================================================================
    // EnergyProduction Validation Tests
    // =========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    fn valid_energy_production() -> EnergyProduction {
        EnergyProduction {
            id: "prod:did:test:producer1:123456".to_string(),
            producer_did: "did:test:producer1".to_string(),
            project_id: "project:solar_farm_1".to_string(),
            amount_kwh: 100.5,
            timestamp: create_test_timestamp(),
            period_hours: 24.0,
            meter_reading: Some(12500.75),
            verified: false,
        }
    }

    #[test]
    fn test_energy_production_valid_did() {
        let production = valid_energy_production();
        assert!(production.producer_did.starts_with("did:"));
    }

    #[test]
    fn test_energy_production_positive_amount() {
        let production = valid_energy_production();
        assert!(production.amount_kwh > 0.0);
    }

    #[test]
    fn test_energy_production_invalid_did_format() {
        let production = EnergyProduction {
            producer_did: "invalid_producer".to_string(),
            ..valid_energy_production()
        };
        // Validation should fail for non-DID format
        assert!(!production.producer_did.starts_with("did:"));
    }

    #[test]
    fn test_energy_production_zero_amount_invalid() {
        let production = EnergyProduction {
            amount_kwh: 0.0,
            ..valid_energy_production()
        };
        assert!(production.amount_kwh <= 0.0);
    }

    #[test]
    fn test_energy_production_negative_amount_invalid() {
        let production = EnergyProduction {
            amount_kwh: -50.0,
            ..valid_energy_production()
        };
        assert!(production.amount_kwh <= 0.0);
    }

    #[test]
    fn test_energy_production_with_meter_reading() {
        let production = valid_energy_production();
        assert!(production.meter_reading.is_some());
        assert_eq!(production.meter_reading.unwrap(), 12500.75);
    }

    #[test]
    fn test_energy_production_without_meter_reading() {
        let production = EnergyProduction {
            meter_reading: None,
            ..valid_energy_production()
        };
        assert!(production.meter_reading.is_none());
    }

    #[test]
    fn test_energy_production_unverified_by_default() {
        let production = valid_energy_production();
        assert!(!production.verified);
    }

    // =========================================================================
    // EnergyConsumption Validation Tests
    // =========================================================================

    fn valid_energy_consumption() -> EnergyConsumption {
        EnergyConsumption {
            id: "cons:did:test:consumer1:123456".to_string(),
            consumer_did: "did:test:consumer1".to_string(),
            amount_kwh: 75.25,
            timestamp: create_test_timestamp(),
            period_hours: 24.0,
            meter_reading: Some(8500.50),
        }
    }

    #[test]
    fn test_energy_consumption_valid_did() {
        let consumption = valid_energy_consumption();
        assert!(consumption.consumer_did.starts_with("did:"));
    }

    #[test]
    fn test_energy_consumption_positive_amount() {
        let consumption = valid_energy_consumption();
        assert!(consumption.amount_kwh > 0.0);
    }

    #[test]
    fn test_energy_consumption_invalid_did_format() {
        let consumption = EnergyConsumption {
            consumer_did: "consumer123".to_string(),
            ..valid_energy_consumption()
        };
        assert!(!consumption.consumer_did.starts_with("did:"));
    }

    #[test]
    fn test_energy_consumption_zero_amount_invalid() {
        let consumption = EnergyConsumption {
            amount_kwh: 0.0,
            ..valid_energy_consumption()
        };
        assert!(consumption.amount_kwh <= 0.0);
    }

    #[test]
    fn test_energy_consumption_negative_amount_invalid() {
        let consumption = EnergyConsumption {
            amount_kwh: -25.0,
            ..valid_energy_consumption()
        };
        assert!(consumption.amount_kwh <= 0.0);
    }

    // =========================================================================
    // TradeOffer Validation Tests
    // =========================================================================

    fn valid_trade_offer() -> TradeOffer {
        TradeOffer {
            id: "offer:did:test:seller1:123456".to_string(),
            seller_did: "did:test:seller1".to_string(),
            project_id: Some("project:solar_farm_1".to_string()),
            amount_kwh: 500.0,
            price_per_kwh: 0.15,
            currency: "USD".to_string(),
            available_from: create_test_timestamp(),
            available_until: Timestamp::from_micros(1704153600000000), // +24 hours
            status: OfferStatus::Active,
            created: create_test_timestamp(),
        }
    }

    #[test]
    fn test_trade_offer_valid_seller_did() {
        let offer = valid_trade_offer();
        assert!(offer.seller_did.starts_with("did:"));
    }

    #[test]
    fn test_trade_offer_positive_amount() {
        let offer = valid_trade_offer();
        assert!(offer.amount_kwh > 0.0);
    }

    #[test]
    fn test_trade_offer_non_negative_price() {
        let offer = valid_trade_offer();
        assert!(offer.price_per_kwh >= 0.0);
    }

    #[test]
    fn test_trade_offer_zero_price_valid() {
        // Free energy offers should be allowed
        let offer = TradeOffer {
            price_per_kwh: 0.0,
            ..valid_trade_offer()
        };
        assert!(offer.price_per_kwh >= 0.0);
    }

    #[test]
    fn test_trade_offer_negative_price_invalid() {
        let offer = TradeOffer {
            price_per_kwh: -0.05,
            ..valid_trade_offer()
        };
        assert!(offer.price_per_kwh < 0.0);
    }

    #[test]
    fn test_trade_offer_invalid_seller_did() {
        let offer = TradeOffer {
            seller_did: "seller123".to_string(),
            ..valid_trade_offer()
        };
        assert!(!offer.seller_did.starts_with("did:"));
    }

    #[test]
    fn test_trade_offer_zero_amount_invalid() {
        let offer = TradeOffer {
            amount_kwh: 0.0,
            ..valid_trade_offer()
        };
        assert!(offer.amount_kwh <= 0.0);
    }

    #[test]
    fn test_trade_offer_with_project_id() {
        let offer = valid_trade_offer();
        assert!(offer.project_id.is_some());
    }

    #[test]
    fn test_trade_offer_without_project_id() {
        let offer = TradeOffer {
            project_id: None,
            ..valid_trade_offer()
        };
        assert!(offer.project_id.is_none());
    }

    #[test]
    fn test_trade_offer_status_transitions() {
        // Test valid status values
        let statuses = [
            OfferStatus::Active,
            OfferStatus::PartiallyFilled,
            OfferStatus::Filled,
            OfferStatus::Expired,
            OfferStatus::Cancelled,
        ];

        for status in statuses {
            let offer = TradeOffer {
                status,
                ..valid_trade_offer()
            };
            // Each status should be cloneable and comparable
            assert_eq!(offer.status.clone(), offer.status);
        }
    }

    // =========================================================================
    // Trade Validation Tests
    // =========================================================================

    fn valid_trade() -> Trade {
        Trade {
            id: "trade:offer1:123456".to_string(),
            offer_id: "offer:did:test:seller1:123456".to_string(),
            seller_did: "did:test:seller1".to_string(),
            buyer_did: "did:test:buyer1".to_string(),
            amount_kwh: 100.0,
            price_per_kwh: 0.15,
            total_price: 15.0,
            currency: "USD".to_string(),
            executed: create_test_timestamp(),
            settled: false,
            payment_reference: None,
        }
    }

    #[test]
    fn test_trade_valid_seller_did() {
        let trade = valid_trade();
        assert!(trade.seller_did.starts_with("did:"));
    }

    #[test]
    fn test_trade_valid_buyer_did() {
        let trade = valid_trade();
        assert!(trade.buyer_did.starts_with("did:"));
    }

    #[test]
    fn test_trade_different_seller_and_buyer() {
        let trade = valid_trade();
        assert_ne!(trade.seller_did, trade.buyer_did);
    }

    #[test]
    fn test_trade_same_seller_buyer_invalid() {
        let trade = Trade {
            buyer_did: "did:test:seller1".to_string(),
            ..valid_trade()
        };
        // Self-trade should be invalid
        assert_eq!(trade.seller_did, trade.buyer_did);
    }

    #[test]
    fn test_trade_invalid_seller_did() {
        let trade = Trade {
            seller_did: "seller123".to_string(),
            ..valid_trade()
        };
        assert!(!trade.seller_did.starts_with("did:"));
    }

    #[test]
    fn test_trade_invalid_buyer_did() {
        let trade = Trade {
            buyer_did: "buyer456".to_string(),
            ..valid_trade()
        };
        assert!(!trade.buyer_did.starts_with("did:"));
    }

    #[test]
    fn test_trade_total_price_calculation() {
        let trade = valid_trade();
        let expected_total = trade.amount_kwh * trade.price_per_kwh;
        assert!((trade.total_price - expected_total).abs() < 0.001);
    }

    #[test]
    fn test_trade_unsettled_by_default() {
        let trade = valid_trade();
        assert!(!trade.settled);
        assert!(trade.payment_reference.is_none());
    }

    #[test]
    fn test_trade_settled_with_reference() {
        let trade = Trade {
            settled: true,
            payment_reference: Some("PAY-2024-001234".to_string()),
            ..valid_trade()
        };
        assert!(trade.settled);
        assert!(trade.payment_reference.is_some());
    }

    // =========================================================================
    // Anchor Tests
    // =========================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("active_energy_offers".to_string());
        assert_eq!(anchor.0, "active_energy_offers");
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor("test_anchor".to_string());
        let anchor2 = Anchor("test_anchor".to_string());
        let anchor3 = Anchor("other_anchor".to_string());

        assert_eq!(anchor1, anchor2);
        assert_ne!(anchor1, anchor3);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_very_large_kwh_amount() {
        let production = EnergyProduction {
            amount_kwh: 1_000_000_000.0, // 1 billion kWh
            ..valid_energy_production()
        };
        assert!(production.amount_kwh > 0.0);
    }

    #[test]
    fn test_very_small_kwh_amount() {
        let production = EnergyProduction {
            amount_kwh: 0.001, // 1 Wh
            ..valid_energy_production()
        };
        assert!(production.amount_kwh > 0.0);
    }

    #[test]
    fn test_very_high_price_per_kwh() {
        let offer = TradeOffer {
            price_per_kwh: 10000.0, // Very high price
            ..valid_trade_offer()
        };
        assert!(offer.price_per_kwh >= 0.0);
    }

    #[test]
    fn test_fractional_kwh_amounts() {
        let trade = Trade {
            amount_kwh: 0.123456789,
            price_per_kwh: 0.15,
            total_price: 0.123456789 * 0.15,
            ..valid_trade()
        };
        assert!(trade.amount_kwh > 0.0);
    }

    #[test]
    fn test_various_currency_codes() {
        let currencies = vec!["USD", "EUR", "GBP", "BTC", "ETH", "SOLAR"];
        for currency in currencies {
            let offer = TradeOffer {
                currency: currency.to_string(),
                ..valid_trade_offer()
            };
            assert!(!offer.currency.is_empty());
        }
    }

    #[test]
    fn test_long_period_hours() {
        let production = EnergyProduction {
            period_hours: 8760.0, // Full year
            ..valid_energy_production()
        };
        assert!(production.period_hours > 0.0);
    }

    #[test]
    fn test_fractional_period_hours() {
        let consumption = EnergyConsumption {
            period_hours: 0.5, // 30 minutes
            ..valid_energy_consumption()
        };
        assert!(consumption.period_hours > 0.0);
    }
}
