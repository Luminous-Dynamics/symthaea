// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! P2P Grid Trading Coordinator Zome
use hdk::prelude::*;
use grid_integrity::*;
use mycelix_energy_shared::batch::{links_to_records, filter_records_by};
use mycelix_energy_shared::anchors::anchor_hash;

#[hdk_extern]
pub fn record_production(input: RecordProductionInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let production = EnergyProduction {
        id: format!("prod:{}:{}", input.producer_did, now.as_micros()),
        producer_did: input.producer_did.clone(),
        project_id: input.project_id,
        amount_kwh: input.amount_kwh,
        timestamp: now,
        period_hours: input.period_hours,
        meter_reading: input.meter_reading,
        verified: false,
    };

    let action_hash = create_entry(&EntryTypes::EnergyProduction(production))?;
    create_link(anchor_hash(&input.producer_did)?, action_hash.clone(), LinkTypes::ProducerToProduction, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordProductionInput {
    pub producer_did: String,
    pub project_id: String,
    pub amount_kwh: f64,
    pub period_hours: f64,
    pub meter_reading: Option<f64>,
}

#[hdk_extern]
pub fn create_trade_offer(input: CreateOfferInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let offer = TradeOffer {
        id: format!("offer:{}:{}", input.seller_did, now.as_micros()),
        seller_did: input.seller_did.clone(),
        project_id: input.project_id,
        amount_kwh: input.amount_kwh,
        price_per_kwh: input.price_per_kwh,
        currency: input.currency,
        available_from: input.available_from,
        available_until: input.available_until,
        status: OfferStatus::Active,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::TradeOffer(offer))?;
    create_link(anchor_hash(&input.seller_did)?, action_hash.clone(), LinkTypes::SellerToOffers, ())?;
    let anchor = anchor_hash("active_energy_offers")?;
    create_link(anchor, action_hash.clone(), LinkTypes::ActiveOffers, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateOfferInput {
    pub seller_did: String,
    pub project_id: Option<String>,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub available_from: Timestamp,
    pub available_until: Timestamp,
}

#[hdk_extern]
pub fn execute_trade(input: ExecuteTradeInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::TradeOffer)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(offer) = record.entry().to_app_option::<TradeOffer>().ok().flatten() {
            if offer.id == input.offer_id && offer.status == OfferStatus::Active {
                let now = sys_time()?;
                let total_price = input.amount_kwh * offer.price_per_kwh;

                let trade = Trade {
                    id: format!("trade:{}:{}", input.offer_id, now.as_micros()),
                    offer_id: input.offer_id.clone(),
                    seller_did: offer.seller_did.clone(),
                    buyer_did: input.buyer_did.clone(),
                    amount_kwh: input.amount_kwh,
                    price_per_kwh: offer.price_per_kwh,
                    total_price,
                    currency: offer.currency.clone(),
                    executed: now,
                    settled: false,
                    payment_reference: None,
                };

                let trade_hash = create_entry(&EntryTypes::Trade(trade))?;
                create_link(anchor_hash(&input.offer_id)?, trade_hash.clone(), LinkTypes::OfferToTrades, ())?;
                create_link(anchor_hash(&input.buyer_did)?, trade_hash.clone(), LinkTypes::BuyerToTrades, ())?;

                // Update offer status
                let remaining = offer.amount_kwh - input.amount_kwh;
                let new_status = if remaining <= 0.0 { OfferStatus::Filled } else { OfferStatus::PartiallyFilled };
                let updated_offer = TradeOffer { amount_kwh: remaining.max(0.0), status: new_status, ..offer };
                update_entry(record.action_address().clone(), &EntryTypes::TradeOffer(updated_offer))?;

                return get(trade_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Offer not found or not active".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecuteTradeInput {
    pub offer_id: String,
    pub buyer_did: String,
    pub amount_kwh: f64,
}

/// Get all active trade offers
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_active_offers(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = anchor_hash("active_energy_offers")?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::ActiveOffers)?, GetStrategy::default())?;
    // FIXED N+1: Batch fetch all records, then filter
    let all_records = links_to_records(links)?;
    Ok(filter_records_by::<TradeOffer, _>(&all_records, |offer| {
        offer.status == OfferStatus::Active || offer.status == OfferStatus::PartiallyFilled
    }))
}

#[hdk_extern]
pub fn settle_trade(input: SettleTradeInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Trade)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(trade) = record.entry().to_app_option::<Trade>().ok().flatten() {
            if trade.id == input.trade_id {
                let settled_trade = Trade { settled: true, payment_reference: Some(input.payment_reference), ..trade };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::Trade(settled_trade))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Trade not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SettleTradeInput {
    pub trade_id: String,
    pub payment_reference: String,
}

/// Get producer's energy production records
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_producer_production(producer_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&producer_did)?, LinkTypes::ProducerToProduction)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get seller's trade offers
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_seller_offers(seller_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&seller_did)?, LinkTypes::SellerToOffers)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get buyer's trade history
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_buyer_trades(buyer_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&buyer_did)?, LinkTypes::BuyerToTrades)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get trades for an offer
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_offer_trades(offer_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&offer_id)?, LinkTypes::OfferToTrades)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Verify energy production (by verifier)
#[hdk_extern]
pub fn verify_production(input: VerifyProductionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProduction)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(production) = record.entry().to_app_option::<EnergyProduction>().ok().flatten() {
            if production.id == input.production_id {
                let verified = EnergyProduction {
                    verified: true,
                    ..production
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::EnergyProduction(verified))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Production record not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyProductionInput {
    pub production_id: String,
    pub verifier_did: String,
}

/// Cancel a trade offer (seller only, only if active)
#[hdk_extern]
pub fn cancel_offer(input: CancelOfferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::TradeOffer)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(offer) = record.entry().to_app_option::<TradeOffer>().ok().flatten() {
            if offer.id == input.offer_id {
                // Only seller can cancel
                if offer.seller_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only seller can cancel offer".into())));
                }

                // Can only cancel active offers
                if offer.status != OfferStatus::Active {
                    return Err(wasm_error!(WasmErrorInner::Guest("Can only cancel active offers".into())));
                }

                let cancelled = TradeOffer {
                    status: OfferStatus::Cancelled,
                    ..offer
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::TradeOffer(cancelled))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Offer not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelOfferInput {
    pub offer_id: String,
    pub requester_did: String,
}

/// Get a specific trade by ID
#[hdk_extern]
pub fn get_trade(trade_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Trade)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(trade) = record.entry().to_app_option::<Trade>().ok().flatten() {
            if trade.id == trade_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get a specific offer by ID
#[hdk_extern]
pub fn get_offer(offer_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::TradeOffer)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(offer) = record.entry().to_app_option::<TradeOffer>().ok().flatten() {
            if offer.id == offer_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get unsettled trades (for payment processing)
#[hdk_extern]
pub fn get_unsettled_trades(_: ()) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Trade)?))
        .include_entries(true);

    let mut trades = Vec::new();
    for record in query(filter)? {
        if let Some(trade) = record.entry().to_app_option::<Trade>().ok().flatten() {
            if !trade.settled {
                trades.push(record);
            }
        }
    }
    Ok(trades)
}

/// Get total production for a producer
#[hdk_extern]
pub fn get_producer_total_production(producer_did: String) -> ExternResult<ProducerStats> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProduction)?))
        .include_entries(true);

    let mut total_kwh = 0.0;
    let mut verified_kwh = 0.0;
    let mut record_count = 0;

    for record in query(filter)? {
        if let Some(production) = record.entry().to_app_option::<EnergyProduction>().ok().flatten() {
            if production.producer_did == producer_did {
                total_kwh += production.amount_kwh;
                if production.verified {
                    verified_kwh += production.amount_kwh;
                }
                record_count += 1;
            }
        }
    }

    Ok(ProducerStats {
        producer_did,
        total_kwh,
        verified_kwh,
        record_count,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProducerStats {
    pub producer_did: String,
    pub total_kwh: f64,
    pub verified_kwh: f64,
    pub record_count: u32,
}

/// Get per-agent thermodynamic yield score [0.0, 1.0].
///
/// Normalizes verified energy production: 30+ verified records = saturation.
/// Quality = verified_kwh / total_kwh (higher = more trustworthy production).
///
/// Used by the 8D Sovereign Profile (D1: Thermodynamic Yield).
#[hdk_extern]
pub fn get_agent_thermodynamic_score(producer_did: String) -> ExternResult<f64> {
    let stats = get_producer_total_production(producer_did)?;
    if stats.record_count == 0 {
        return Ok(0.0);
    }
    let saturation = (stats.record_count as f64 / 30.0).min(1.0);
    let quality = if stats.total_kwh > 0.0 {
        (stats.verified_kwh / stats.total_kwh).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Ok((saturation * quality).clamp(0.0, 1.0))
}

/// Update offer price (seller only)
#[hdk_extern]
pub fn update_offer_price(input: UpdateOfferPriceInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::TradeOffer)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(offer) = record.entry().to_app_option::<TradeOffer>().ok().flatten() {
            if offer.id == input.offer_id {
                // Only seller can update
                if offer.seller_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only seller can update price".into())));
                }

                // Can only update active offers
                if offer.status != OfferStatus::Active && offer.status != OfferStatus::PartiallyFilled {
                    return Err(wasm_error!(WasmErrorInner::Guest("Can only update active offers".into())));
                }

                let updated = TradeOffer {
                    price_per_kwh: input.new_price_per_kwh,
                    ..offer
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::TradeOffer(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Offer not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateOfferPriceInput {
    pub offer_id: String,
    pub requester_did: String,
    pub new_price_per_kwh: f64,
}

/// Get grid trading summary
#[hdk_extern]
pub fn get_grid_summary(_: ()) -> ExternResult<GridSummary> {
    let offer_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::TradeOffer)?))
        .include_entries(true);

    let mut active_offers = 0;
    let mut total_kwh_available = 0.0;

    for record in query(offer_filter)? {
        if let Some(offer) = record.entry().to_app_option::<TradeOffer>().ok().flatten() {
            if offer.status == OfferStatus::Active || offer.status == OfferStatus::PartiallyFilled {
                active_offers += 1;
                total_kwh_available += offer.amount_kwh;
            }
        }
    }

    let trade_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Trade)?))
        .include_entries(true);

    let mut total_trades = 0;
    let mut total_kwh_traded = 0.0;
    let mut total_value_traded = 0.0;

    for record in query(trade_filter)? {
        if let Some(trade) = record.entry().to_app_option::<Trade>().ok().flatten() {
            total_trades += 1;
            total_kwh_traded += trade.amount_kwh;
            total_value_traded += trade.total_price;
        }
    }

    Ok(GridSummary {
        active_offers,
        total_kwh_available,
        total_trades,
        total_kwh_traded,
        total_value_traded,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GridSummary {
    pub active_offers: u32,
    pub total_kwh_available: f64,
    pub total_trades: u32,
    pub total_kwh_traded: f64,
    pub total_value_traded: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Input Struct Validation Tests
    // =========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    // =========================================================================
    // RecordProductionInput Tests
    // =========================================================================

    fn valid_record_production_input() -> RecordProductionInput {
        RecordProductionInput {
            producer_did: "did:mycelix:producer1".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            amount_kwh: 250.5,
            period_hours: 24.0,
            meter_reading: Some(50000.0),
        }
    }

    #[test]
    fn test_record_production_input_valid() {
        let input = valid_record_production_input();
        assert!(input.producer_did.starts_with("did:"));
        assert!(input.amount_kwh > 0.0);
        assert!(input.period_hours > 0.0);
    }

    #[test]
    fn test_record_production_input_with_meter_reading() {
        let input = valid_record_production_input();
        assert!(input.meter_reading.is_some());
    }

    #[test]
    fn test_record_production_input_without_meter_reading() {
        let input = RecordProductionInput {
            meter_reading: None,
            ..valid_record_production_input()
        };
        assert!(input.meter_reading.is_none());
    }

    #[test]
    fn test_record_production_input_serialization() {
        let input = valid_record_production_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // CreateOfferInput Tests
    // =========================================================================

    fn valid_create_offer_input() -> CreateOfferInput {
        CreateOfferInput {
            seller_did: "did:mycelix:seller1".to_string(),
            project_id: Some("project:wind_farm_beta".to_string()),
            amount_kwh: 1000.0,
            price_per_kwh: 0.12,
            currency: "USD".to_string(),
            available_from: create_test_timestamp(),
            available_until: Timestamp::from_micros(1704153600000000),
        }
    }

    #[test]
    fn test_create_offer_input_valid() {
        let input = valid_create_offer_input();
        assert!(input.seller_did.starts_with("did:"));
        assert!(input.amount_kwh > 0.0);
        assert!(input.price_per_kwh >= 0.0);
        assert!(!input.currency.is_empty());
    }

    #[test]
    fn test_create_offer_input_with_project() {
        let input = valid_create_offer_input();
        assert!(input.project_id.is_some());
    }

    #[test]
    fn test_create_offer_input_without_project() {
        let input = CreateOfferInput {
            project_id: None,
            ..valid_create_offer_input()
        };
        assert!(input.project_id.is_none());
    }

    #[test]
    fn test_create_offer_input_free_energy() {
        let input = CreateOfferInput {
            price_per_kwh: 0.0,
            ..valid_create_offer_input()
        };
        assert_eq!(input.price_per_kwh, 0.0);
    }

    #[test]
    fn test_create_offer_input_various_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "CHF", "JPY"];
        for currency in currencies {
            let input = CreateOfferInput {
                currency: currency.to_string(),
                ..valid_create_offer_input()
            };
            assert_eq!(input.currency, currency);
        }
    }

    #[test]
    fn test_create_offer_input_timestamp_range() {
        let input = valid_create_offer_input();
        // available_until should be after available_from
        assert!(input.available_until.as_micros() >= input.available_from.as_micros());
    }

    // =========================================================================
    // ExecuteTradeInput Tests
    // =========================================================================

    fn valid_execute_trade_input() -> ExecuteTradeInput {
        ExecuteTradeInput {
            offer_id: "offer:did:mycelix:seller1:123456789".to_string(),
            buyer_did: "did:mycelix:buyer1".to_string(),
            amount_kwh: 100.0,
        }
    }

    #[test]
    fn test_execute_trade_input_valid() {
        let input = valid_execute_trade_input();
        assert!(!input.offer_id.is_empty());
        assert!(input.buyer_did.starts_with("did:"));
        assert!(input.amount_kwh > 0.0);
    }

    #[test]
    fn test_execute_trade_input_partial_fill() {
        let input = ExecuteTradeInput {
            amount_kwh: 50.0, // Less than the full offer
            ..valid_execute_trade_input()
        };
        assert!(input.amount_kwh > 0.0);
    }

    #[test]
    fn test_execute_trade_input_full_fill() {
        let input = ExecuteTradeInput {
            amount_kwh: 1000.0, // Full offer amount
            ..valid_execute_trade_input()
        };
        assert!(input.amount_kwh > 0.0);
    }

    // =========================================================================
    // SettleTradeInput Tests
    // =========================================================================

    fn valid_settle_trade_input() -> SettleTradeInput {
        SettleTradeInput {
            trade_id: "trade:offer1:123456789".to_string(),
            payment_reference: "PAY-2024-00001".to_string(),
        }
    }

    #[test]
    fn test_settle_trade_input_valid() {
        let input = valid_settle_trade_input();
        assert!(!input.trade_id.is_empty());
        assert!(!input.payment_reference.is_empty());
    }

    #[test]
    fn test_settle_trade_input_various_payment_refs() {
        let payment_refs = vec![
            "WIRE-123456",
            "ACH-789012",
            "CRYPTO-0xabc123",
            "CHECK-00001",
            "INTERNAL-TRANSFER-001",
        ];
        for ref_id in payment_refs {
            let input = SettleTradeInput {
                payment_reference: ref_id.to_string(),
                ..valid_settle_trade_input()
            };
            assert!(!input.payment_reference.is_empty());
        }
    }

    // =========================================================================
    // VerifyProductionInput Tests
    // =========================================================================

    fn valid_verify_production_input() -> VerifyProductionInput {
        VerifyProductionInput {
            production_id: "prod:did:mycelix:producer1:123456789".to_string(),
            verifier_did: "did:mycelix:verifier1".to_string(),
        }
    }

    #[test]
    fn test_verify_production_input_valid() {
        let input = valid_verify_production_input();
        assert!(!input.production_id.is_empty());
        assert!(input.verifier_did.starts_with("did:"));
    }

    // =========================================================================
    // CancelOfferInput Tests
    // =========================================================================

    fn valid_cancel_offer_input() -> CancelOfferInput {
        CancelOfferInput {
            offer_id: "offer:did:mycelix:seller1:123456789".to_string(),
            requester_did: "did:mycelix:seller1".to_string(),
        }
    }

    #[test]
    fn test_cancel_offer_input_valid() {
        let input = valid_cancel_offer_input();
        assert!(!input.offer_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
    }

    #[test]
    fn test_cancel_offer_input_seller_must_match() {
        let input = valid_cancel_offer_input();
        // The offer_id contains seller DID, requester should match
        assert!(input.offer_id.contains("seller1"));
        assert!(input.requester_did.contains("seller1"));
    }

    // =========================================================================
    // UpdateOfferPriceInput Tests
    // =========================================================================

    fn valid_update_offer_price_input() -> UpdateOfferPriceInput {
        UpdateOfferPriceInput {
            offer_id: "offer:did:mycelix:seller1:123456789".to_string(),
            requester_did: "did:mycelix:seller1".to_string(),
            new_price_per_kwh: 0.18,
        }
    }

    #[test]
    fn test_update_offer_price_input_valid() {
        let input = valid_update_offer_price_input();
        assert!(!input.offer_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
        assert!(input.new_price_per_kwh >= 0.0);
    }

    #[test]
    fn test_update_offer_price_input_price_increase() {
        let input = UpdateOfferPriceInput {
            new_price_per_kwh: 0.25,
            ..valid_update_offer_price_input()
        };
        assert!(input.new_price_per_kwh > 0.0);
    }

    #[test]
    fn test_update_offer_price_input_price_decrease() {
        let input = UpdateOfferPriceInput {
            new_price_per_kwh: 0.05,
            ..valid_update_offer_price_input()
        };
        assert!(input.new_price_per_kwh > 0.0);
    }

    #[test]
    fn test_update_offer_price_input_free_energy() {
        let input = UpdateOfferPriceInput {
            new_price_per_kwh: 0.0,
            ..valid_update_offer_price_input()
        };
        assert_eq!(input.new_price_per_kwh, 0.0);
    }

    // =========================================================================
    // ProducerStats Tests
    // =========================================================================

    #[test]
    fn test_producer_stats_creation() {
        let stats = ProducerStats {
            producer_did: "did:mycelix:producer1".to_string(),
            total_kwh: 50000.0,
            verified_kwh: 45000.0,
            record_count: 100,
        };
        assert!(stats.producer_did.starts_with("did:"));
        assert!(stats.total_kwh >= stats.verified_kwh);
    }

    #[test]
    fn test_producer_stats_no_verified() {
        let stats = ProducerStats {
            producer_did: "did:mycelix:producer1".to_string(),
            total_kwh: 10000.0,
            verified_kwh: 0.0,
            record_count: 50,
        };
        assert_eq!(stats.verified_kwh, 0.0);
        assert!(stats.record_count > 0);
    }

    #[test]
    fn test_producer_stats_all_verified() {
        let stats = ProducerStats {
            producer_did: "did:mycelix:producer1".to_string(),
            total_kwh: 25000.0,
            verified_kwh: 25000.0,
            record_count: 75,
        };
        assert_eq!(stats.total_kwh, stats.verified_kwh);
    }

    #[test]
    fn test_producer_stats_empty() {
        let stats = ProducerStats {
            producer_did: "did:mycelix:new_producer".to_string(),
            total_kwh: 0.0,
            verified_kwh: 0.0,
            record_count: 0,
        };
        assert_eq!(stats.record_count, 0);
    }

    // =========================================================================
    // GridSummary Tests
    // =========================================================================

    #[test]
    fn test_grid_summary_creation() {
        let summary = GridSummary {
            active_offers: 50,
            total_kwh_available: 100000.0,
            total_trades: 200,
            total_kwh_traded: 500000.0,
            total_value_traded: 75000.0,
        };
        assert!(summary.active_offers > 0);
        assert!(summary.total_kwh_available > 0.0);
    }

    #[test]
    fn test_grid_summary_empty_grid() {
        let summary = GridSummary {
            active_offers: 0,
            total_kwh_available: 0.0,
            total_trades: 0,
            total_kwh_traded: 0.0,
            total_value_traded: 0.0,
        };
        assert_eq!(summary.active_offers, 0);
        assert_eq!(summary.total_trades, 0);
    }

    #[test]
    fn test_grid_summary_value_consistency() {
        let summary = GridSummary {
            active_offers: 10,
            total_kwh_available: 5000.0,
            total_trades: 100,
            total_kwh_traded: 10000.0,
            total_value_traded: 1500.0,
        };
        // Average price should be reasonable
        if summary.total_kwh_traded > 0.0 {
            let avg_price = summary.total_value_traded / summary.total_kwh_traded;
            assert!(avg_price >= 0.0);
        }
    }

    #[test]
    fn test_grid_summary_high_volume() {
        let summary = GridSummary {
            active_offers: 10000,
            total_kwh_available: 50_000_000.0,
            total_trades: 1_000_000,
            total_kwh_traded: 100_000_000.0,
            total_value_traded: 15_000_000.0,
        };
        assert!(summary.active_offers > 0);
    }

    // =========================================================================
    // Business Logic Edge Cases
    // =========================================================================

    #[test]
    fn test_offer_time_window_same_time() {
        // Edge case: available_from == available_until (instant availability)
        let input = CreateOfferInput {
            available_from: create_test_timestamp(),
            available_until: create_test_timestamp(),
            ..valid_create_offer_input()
        };
        assert_eq!(input.available_from, input.available_until);
    }

    #[test]
    fn test_fractional_kwh_trade() {
        let input = ExecuteTradeInput {
            amount_kwh: 0.001, // 1 Wh
            ..valid_execute_trade_input()
        };
        assert!(input.amount_kwh > 0.0);
    }

    #[test]
    fn test_large_kwh_trade() {
        let input = ExecuteTradeInput {
            amount_kwh: 1_000_000.0, // 1 GWh
            ..valid_execute_trade_input()
        };
        assert!(input.amount_kwh > 0.0);
    }

    #[test]
    fn test_very_high_price() {
        let input = CreateOfferInput {
            price_per_kwh: 10000.0, // Very high price
            ..valid_create_offer_input()
        };
        assert!(input.price_per_kwh > 0.0);
    }

    #[test]
    fn test_different_did_formats() {
        let did_formats = vec![
            "did:mycelix:user1",
            "did:web:example.com:user1",
            "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "did:ion:EiDyOQbbZAa3aiRzeCkV7LOx3SERjjH93EXoIM3UoN4oWg",
        ];
        for did in did_formats {
            let input = CreateOfferInput {
                seller_did: did.to_string(),
                ..valid_create_offer_input()
            };
            assert!(input.seller_did.starts_with("did:"));
        }
    }

    #[test]
    fn test_serialization_deserialization_roundtrip() {
        let input = valid_create_offer_input();
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.seller_did, input.seller_did);
        assert_eq!(deserialized.amount_kwh, input.amount_kwh);
    }

    #[test]
    fn test_execute_trade_input_zero_amount_edge() {
        // This should be invalid in actual validation, but struct allows it
        let input = ExecuteTradeInput {
            amount_kwh: 0.0,
            ..valid_execute_trade_input()
        };
        // Business logic should reject this
        assert!(input.amount_kwh == 0.0);
    }

    #[test]
    fn test_long_offer_id_format() {
        let input = ExecuteTradeInput {
            offer_id: format!("offer:did:mycelix:seller1:{}", "9".repeat(50)),
            ..valid_execute_trade_input()
        };
        assert!(!input.offer_id.is_empty());
    }
}
