// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Climate.
//!
//! Three-tier data loading:
//! 1. localStorage (persisted mock/user data) — instant
//! 2. Shared mock data fallback — instant
//! 3. Real Holochain conductor — after 4s if connected

use leptos::prelude::*;
use climate_leptos_types::*;
use mycelix_mock_data::climate as mock;
use mycelix_leptos_core::{load_json, save_json};

const KEY_PROJECTS: &str = "climate_projects";
const KEY_CREDITS: &str = "climate_credits";
const KEY_FOOTPRINTS: &str = "climate_footprints";

#[derive(Clone)]
pub struct ClimateCtx {
    pub projects: RwSignal<Vec<ClimateProjectView>>,
    pub credits: RwSignal<Vec<CarbonCreditView>>,
    pub footprints: RwSignal<Vec<CarbonFootprintView>>,
    pub credit_summary: RwSignal<CreditSummaryView>,
    pub projects_summary: RwSignal<ProjectsSummaryView>,
}

pub fn provide_climate_context() {
    // Load from localStorage first, fallback to mock
    let projects = load_json::<Vec<ClimateProjectView>>(KEY_PROJECTS)
        .unwrap_or_else(mock::projects);
    let credits = load_json::<Vec<CarbonCreditView>>(KEY_CREDITS)
        .unwrap_or_else(mock::credits);
    let footprints = load_json::<Vec<CarbonFootprintView>>(KEY_FOOTPRINTS)
        .unwrap_or_else(mock::footprints);

    let state = ClimateCtx {
        projects: RwSignal::new(projects),
        credits: RwSignal::new(credits),
        footprints: RwSignal::new(footprints),
        credit_summary: RwSignal::new(mock::credit_summary()),
        projects_summary: RwSignal::new(mock::projects_summary()),
    };
    provide_context(state.clone());

    // Auto-save to localStorage when data changes
    let projects_save = state.projects;
    let credits_save = state.credits;
    let footprints_save = state.footprints;

    Effect::new(move |_| {
        save_json(KEY_PROJECTS, &projects_save.get());
    });
    Effect::new(move |_| {
        save_json(KEY_CREDITS, &credits_save.get());
    });
    Effect::new(move |_| {
        save_json(KEY_FOOTPRINTS, &footprints_save.get());
    });

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[Climate] Conductor connected — loading real data...".into());

            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "projects", "get_all_projects", &()
            ).await {
                let projects: Vec<ClimateProjectView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !projects.is_empty() {
                    web_sys::console::log_1(&format!("[Climate] Loaded {} projects", projects.len()).into());
                    state.projects.set(projects);
                }
            }

            let owner = mycelix_leptos_core::local_did();
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "carbon", "get_credits_by_owner", &owner
            ).await {
                let credits: Vec<CarbonCreditView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !credits.is_empty() {
                    state.credits.set(credits);
                }
            }

            if let Ok(summary) = hc.call_zome_default::<String, CreditSummaryView>(
                "carbon", "get_credit_summary", &owner
            ).await {
                state.credit_summary.set(summary);
            }

            if let Ok(summary) = hc.call_zome_default::<(), ProjectsSummaryView>(
                "projects", "get_projects_summary", &()
            ).await {
                state.projects_summary.set(summary);
            }

            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "carbon", "get_footprints_by_entity", &owner
            ).await {
                let footprints: Vec<CarbonFootprintView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !footprints.is_empty() {
                    state.footprints.set(footprints);
                }
            }
        }
    });
}

fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_climate_context() -> ClimateCtx {
    expect_context::<ClimateCtx>()
}
