// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use serde_json::Value;
use wasm_bindgen_futures::spawn_local;
use commons_leptos_types::*;
use crate::mock_data;
use mycelix_leptos_core::holochain_provider::use_holochain;

#[derive(Clone)]
pub struct CommonsCtx {
    pub needs: RwSignal<Vec<NeedView>>,
    pub offers: RwSignal<Vec<OfferView>>,
    pub care_circles: RwSignal<Vec<CareCircleView>>,
    pub plots: RwSignal<Vec<PlotView>>,
    pub markets: RwSignal<Vec<MarketView>>,
    pub food_listings: RwSignal<Vec<FoodListingView>>,
    pub compost_batches: RwSignal<Vec<mock_data::CompostBatchView>>,
    pub water_systems: RwSignal<Vec<WaterSystemView>>,
    pub tools: RwSignal<Vec<ToolView>>,
    pub events: RwSignal<Vec<EventView>>,
    pub my_agent_did: RwSignal<String>,
}

pub fn provide_commons_context() -> CommonsCtx {
    let ctx = CommonsCtx {
        needs: RwSignal::new(mock_data::mock_needs()),
        offers: RwSignal::new(mock_data::mock_offers()),
        care_circles: RwSignal::new(mock_data::mock_care_circles()),
        plots: RwSignal::new(mock_data::mock_plots()),
        markets: RwSignal::new(mock_data::mock_markets()),
        food_listings: RwSignal::new(mock_data::mock_food_listings()),
        compost_batches: RwSignal::new(mock_data::mock_compost_batches()),
        water_systems: RwSignal::new(mock_data::mock_water_systems()),
        tools: RwSignal::new(mock_data::mock_tools()),
        events: RwSignal::new(mock_data::mock_events()),
        my_agent_did: RwSignal::new("did:mycelix:mock-commoner".into()),
    };
    provide_context(ctx.clone());

    if let Some(set_pending) = use_context::<WriteSignal<u32>>() {
        let open = ctx.needs.get_untracked().iter().filter(|n| n.status == NeedStatus::Open).count() as u32;
        set_pending.set(open);
    }

    let c = ctx.clone();
    spawn_local(async move { gloo_timers::future::TimeoutFuture::new(4000).await; try_load(c).await; });
    ctx
}

#[derive(Clone, serde::Serialize)]
struct SearchNeedsInput {
    category: Option<Value>,
    urgency: Option<Value>,
    emergency_only: bool,
    query: Option<String>,
    limit: Option<u32>,
}

#[derive(Clone, serde::Serialize)]
struct SearchOffersInput {
    category: Option<Value>,
    query: Option<String>,
    limit: Option<u32>,
}

async fn try_load(ctx: CommonsCtx) {
    let hc = use_holochain();
    if hc.is_mock() {
        web_sys::console::log_1(&"[Commons] Running in mock mode — using simulated commons data.".into());
        return;
    }

    if let Some(did) = hc.connected_agent_did() {
        ctx.my_agent_did.set(did);
    }

    match hc
        .call_zome::<_, Vec<Value>>(
            "commons_care",
            "mutualaid-needs",
            "search_needs",
            &SearchNeedsInput {
                category: None,
                urgency: None,
                emergency_only: false,
                query: None,
                limit: Some(50),
            },
        )
        .await
    {
        Ok(records) => {
            let needs = decode_records(&records, decode_need_record);
            web_sys::console::log_1(&format!("[Commons] Loaded {} open needs", needs.len()).into());
            if !needs.is_empty() {
                if let Some(set_pending) = use_context::<WriteSignal<u32>>() {
                    set_pending.set(needs.iter().filter(|need| need.status == NeedStatus::Open).count() as u32);
                }
                ctx.needs.set(needs);
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Commons] Could not load needs: {e}").into());
        }
    }

    match hc
        .call_zome::<_, Vec<Value>>(
            "commons_care",
            "mutualaid-needs",
            "search_offers",
            &SearchOffersInput {
                category: None,
                query: None,
                limit: Some(50),
            },
        )
        .await
    {
        Ok(records) => {
            let offers = decode_records(&records, decode_offer_record);
            web_sys::console::log_1(&format!("[Commons] Loaded {} available offers", offers.len()).into());
            if !offers.is_empty() {
                ctx.offers.set(offers);
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Commons] Could not load offers: {e}").into());
        }
    }

    match hc
        .call_zome::<(), Vec<Value>>("commons_care", "care_circles", "get_all_circles", &())
        .await
    {
        Ok(records) => {
            let circles = decode_records(&records, decode_care_circle_record);
            web_sys::console::log_1(&format!("[Commons] Loaded {} care circles", circles.len()).into());
            if !circles.is_empty() {
                ctx.care_circles.set(circles);
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Commons] Could not load care circles: {e}").into());
        }
    }

    match hc
        .call_zome::<(), Vec<Value>>("commons_land", "food_production", "get_all_plots", &())
        .await
    {
        Ok(records) => {
            let plots = decode_records(&records, decode_plot_record);
            web_sys::console::log_1(&format!("[Commons] Loaded {} plots", plots.len()).into());
            if !plots.is_empty() {
                ctx.plots.set(plots);
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Commons] Could not load plots: {e}").into());
        }
    }

    match hc
        .call_zome::<(), Vec<Value>>("commons_land", "food_distribution", "get_all_markets", &())
        .await
    {
        Ok(records) => {
            let mut markets = decode_records(&records, decode_market_record);
            web_sys::console::log_1(&format!("[Commons] Loaded {} markets", markets.len()).into());

            let mut food_listings = Vec::new();
            for market in &mut markets {
                match hc
                    .call_zome::<String, Vec<Value>>(
                        "commons_land",
                        "food_distribution",
                        "get_market_listings",
                        &market.hash,
                    )
                    .await
                {
                    Ok(records) => {
                        let listings = decode_records(&records, decode_food_listing_record);
                        market.listing_count = listings.len() as u32;
                        food_listings.extend(listings);
                    }
                    Err(e) => {
                        web_sys::console::log_1(
                            &format!(
                                "[Commons] Could not load listings for market {}: {e}",
                                market.name
                            )
                            .into(),
                        );
                    }
                }
            }

            if !markets.is_empty() {
                ctx.markets.set(markets);
            }
            if !food_listings.is_empty() {
                web_sys::console::log_1(
                    &format!("[Commons] Loaded {} food listings", food_listings.len()).into(),
                );
                ctx.food_listings.set(food_listings);
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Commons] Could not load markets: {e}").into());
        }
    }
}

pub fn use_commons() -> CommonsCtx { expect_context::<CommonsCtx>() }

fn decode_records<T>(records: &[Value], decode: fn(&Value) -> Option<T>) -> Vec<T> {
    records.iter().filter_map(decode).collect()
}

fn record_entry(record: &Value) -> Option<&Value> {
    record.get("entry").and_then(|entry| entry.get("Present")).or(Some(record))
}

fn value_string(value: &Value) -> Option<String> {
    match value {
        Value::String(string) => Some(string.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(boolean) => Some(boolean.to_string()),
        Value::Object(object) if object.len() == 1 => object.values().next().and_then(value_string),
        _ => None,
    }
}

fn value_u32(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|n| u32::try_from(n).ok())
}

fn value_i64(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| value.as_u64().and_then(|n| i64::try_from(n).ok()))
}

fn timestamp_to_micros(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => value_i64(&Value::Number(number.clone())),
        Value::String(string) => string.parse().ok(),
        Value::Object(object) => object
            .get("micros")
            .and_then(value_i64)
            .or_else(|| object.get("timestamp_us").and_then(value_i64)),
        _ => None,
    }
}

fn decode_need_record(record: &Value) -> Option<NeedView> {
    let entry = record_entry(record)?;
    Some(NeedView {
        hash: action_hash(record)?,
        id: entry.get("id").and_then(value_string)?,
        title: entry.get("title").and_then(value_string)?,
        description: entry.get("description").and_then(value_string)?,
        category: parse_need_category(entry.get("category")?)?,
        requester_did: entry.get("requester").and_then(value_string)?,
        urgency: parse_urgency(entry.get("urgency")?)?,
        status: parse_need_status(entry.get("status")?)?,
        created: entry
            .get("created_at")
            .and_then(timestamp_to_micros)
            .unwrap_or_default(),
    })
}

fn decode_offer_record(record: &Value) -> Option<OfferView> {
    let entry = record_entry(record)?;
    let status = parse_offer_status(entry.get("status")?)?;
    if !matches!(status, OfferBackendStatus::Available) {
        return None;
    }
    Some(OfferView {
        hash: action_hash(record)?,
        id: entry.get("id").and_then(value_string)?,
        title: entry.get("title").and_then(value_string)?,
        description: entry.get("description").and_then(value_string)?,
        category: parse_need_category(entry.get("category")?)?,
        offerer_did: entry.get("offerer").and_then(value_string)?,
        created: entry
            .get("created_at")
            .and_then(timestamp_to_micros)
            .unwrap_or_default(),
    })
}

fn decode_care_circle_record(record: &Value) -> Option<CareCircleView> {
    let entry = record_entry(record)?;
    Some(CareCircleView {
        hash: action_hash(record)?,
        name: entry.get("name").and_then(value_string)?,
        description: entry.get("description").and_then(value_string)?,
        circle_type: parse_circle_type(entry.get("circle_type")?)?,
        member_count: entry.get("max_members").and_then(value_u32).unwrap_or_default(),
        active: entry.get("active").and_then(Value::as_bool).unwrap_or(true),
        created: entry
            .get("created_at")
            .and_then(timestamp_to_micros)
            .unwrap_or_default(),
    })
}

fn decode_plot_record(record: &Value) -> Option<PlotView> {
    let entry = record_entry(record)?;
    Some(PlotView {
        hash: action_hash(record)?,
        name: entry.get("name").and_then(value_string)?,
        area_sqm: entry.get("area_sqm").and_then(Value::as_f64)?,
        plot_type: parse_plot_type(entry.get("plot_type")?)?,
        steward_did: entry.get("steward").and_then(value_string)?,
        crop_count: 0,
    })
}

fn decode_market_record(record: &Value) -> Option<MarketView> {
    let entry = record_entry(record)?;
    Some(MarketView {
        hash: action_hash(record)?,
        name: entry.get("name").and_then(value_string)?,
        market_type: parse_market_type(entry.get("market_type")?)?,
        listing_count: 0,
    })
}

fn decode_food_listing_record(record: &Value) -> Option<FoodListingView> {
    let entry = record_entry(record)?;
    let status = parse_listing_status(entry.get("status")?)?;
    Some(FoodListingView {
        hash: action_hash(record)?,
        product_name: entry.get("product_name").and_then(value_string)?,
        quantity_kg: entry.get("quantity_kg").and_then(Value::as_f64)?,
        price_per_kg: entry.get("price_per_kg").and_then(Value::as_f64)?,
        organic: entry.get("organic").and_then(Value::as_bool).unwrap_or(false),
        producer_did: entry.get("producer").and_then(value_string)?,
        available: matches!(status, ListingBackendStatus::Available),
    })
}

fn action_hash(record: &Value) -> Option<String> {
    record
        .get("signed_action")
        .and_then(|action| action.get("hashed"))
        .and_then(|hashed| hashed.get("hash"))
        .and_then(value_string)
        .or_else(|| record.get("action_hash").and_then(value_string))
}

fn parse_need_category(value: &Value) -> Option<NeedCategory> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Housing" => NeedCategory::Housing,
        "Childcare" => NeedCategory::Childcare,
        "Transportation" | "Rides" => NeedCategory::Transportation,
        "Food" => NeedCategory::Food,
        "Healthcare" => NeedCategory::Healthcare,
        "Tools" | "Equipment" | "Skills" | "Computers" | "Books" | "SchoolSupplies" => NeedCategory::Skills,
        other => NeedCategory::Other(other.to_string()),
    })
}

fn parse_urgency(value: &Value) -> Option<Urgency> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Low" => Urgency::Low,
        "Medium" => Urgency::Medium,
        "High" => Urgency::High,
        "Urgent" | "Emergency" => Urgency::Critical,
        _ => return None,
    })
}

fn parse_need_status(value: &Value) -> Option<NeedStatus> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Open" => NeedStatus::Open,
        "Matched" | "PartiallyMet" => NeedStatus::Matched,
        "Fulfilled" | "Expired" => NeedStatus::Fulfilled,
        "Withdrawn" => NeedStatus::Withdrawn,
        _ => return None,
    })
}

enum OfferBackendStatus {
    Available,
    Reserved,
    Claimed,
    Completed,
    Withdrawn,
    Expired,
}

fn parse_offer_status(value: &Value) -> Option<OfferBackendStatus> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Available" => OfferBackendStatus::Available,
        "Reserved" => OfferBackendStatus::Reserved,
        "Claimed" => OfferBackendStatus::Claimed,
        "Completed" => OfferBackendStatus::Completed,
        "Withdrawn" => OfferBackendStatus::Withdrawn,
        "Expired" => OfferBackendStatus::Expired,
        _ => return None,
    })
}

enum ListingBackendStatus {
    Available,
    Reserved,
    Sold,
    Expired,
}

fn parse_listing_status(value: &Value) -> Option<ListingBackendStatus> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Available" => ListingBackendStatus::Available,
        "Reserved" => ListingBackendStatus::Reserved,
        "Sold" => ListingBackendStatus::Sold,
        "Expired" => ListingBackendStatus::Expired,
        _ => return None,
    })
}

fn parse_circle_type(value: &Value) -> Option<CircleType> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Neighborhood" => CircleType::Neighborhood,
        "Family" => CircleType::Family,
        "Workplace" | "School" | "Faith" => CircleType::MutualAid,
        other => CircleType::Other(other.to_string()),
    })
}

fn parse_plot_type(value: &Value) -> Option<PlotType> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Garden" | "CommunityGarden" => PlotType::CommunityGarden,
        "Rooftop" => PlotType::Rooftop,
        "Orchard" => PlotType::Orchard,
        "FoodForest" => PlotType::Orchard,
        "Greenhouse" | "Raised" => PlotType::Farm,
        other => PlotType::Other(other.to_string()),
    })
}

fn parse_market_type(value: &Value) -> Option<MarketType> {
    let string = value_string(value)?;
    Some(match string.as_str() {
        "Farmers" => MarketType::Farmers,
        "FoodBank" => MarketType::FoodBank,
        "CoOp" => MarketType::CoOp,
        "CSA" => MarketType::CoOp,
        _ => return None,
    })
}
