// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Climate.
//!
//! Mock-first: loads shared mock data immediately for instant render,
//! then attempts real data from the Holochain conductor.

use leptos::prelude::*;
use climate_leptos_types::*;
use mycelix_mock_data::climate as mock;

#[derive(Clone)]
pub struct ClimateCtx {
    pub projects: RwSignal<Vec<ClimateProjectView>>,
    pub credits: RwSignal<Vec<CarbonCreditView>>,
    pub footprints: RwSignal<Vec<CarbonFootprintView>>,
    pub credit_summary: RwSignal<CreditSummaryView>,
    pub projects_summary: RwSignal<ProjectsSummaryView>,
}

pub fn provide_climate_context() {
    let state = ClimateCtx {
        projects: RwSignal::new(mock::projects()),
        credits: RwSignal::new(mock::credits()),
        footprints: RwSignal::new(mock::footprints()),
        credit_summary: RwSignal::new(mock::credit_summary()),
        projects_summary: RwSignal::new(mock::projects_summary()),
    };
    provide_context(state.clone());

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            // Wire real zome calls here (Phase 1):
            // if let Ok(projects) = hc.call_zome_default::<(), Vec<ClimateProjectView>>(
            //     "projects", "get_all_projects", &()
            // ).await {
            //     state.projects.set(projects);
            // }
            let _ = &state;
        }
    });
}

pub fn use_climate_context() -> ClimateCtx {
    expect_context::<ClimateCtx>()
}
