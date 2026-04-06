// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
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

async fn try_load(ctx: CommonsCtx) {
    let hc = use_holochain();
    if hc.is_mock() { return; }
    // Load needs
    match hc.call_zome::<(), Vec<serde_json::Value>>("commons_care", "mutualaid-needs", "get_emergency_needs", &()).await {
        Ok(records) => { web_sys::console::log_1(&format!("[Commons] Loaded {} emergency needs", records.len()).into()); }
        Err(e) => { web_sys::console::log_1(&format!("[Commons] Could not load needs: {e}").into()); }
    }
    let _ = ctx;
}

pub fn use_commons() -> CommonsCtx { expect_context::<CommonsCtx>() }
