// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Energy.

use leptos::prelude::*;
use energy_leptos_types::*;
use mycelix_mock_data::energy as mock;

#[derive(Clone)]
pub struct EnergyCtx {
    pub projects: RwSignal<Vec<EnergyProjectView>>,
    pub investments: RwSignal<Vec<InvestmentView>>,
    pub offers: RwSignal<Vec<TradeOfferView>>,
    pub trades: RwSignal<Vec<TradeView>>,
    pub contracts: RwSignal<Vec<RegenerativeContractView>>,
}

pub fn provide_energy_context() {
    let state = EnergyCtx {
        projects: RwSignal::new(mock::projects()),
        investments: RwSignal::new(mock::investments()),
        offers: RwSignal::new(mock::offers()),
        trades: RwSignal::new(vec![]),
        contracts: RwSignal::new(mock::contracts()),
    };
    provide_context(state.clone());

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[Energy] Conductor connected — loading real data...".into());

            // Load projects
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "projects", "get_all_projects", &()
            ).await {
                let projects: Vec<EnergyProjectView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !projects.is_empty() {
                    web_sys::console::log_1(&format!("[Energy] Loaded {} projects", projects.len()).into());
                    state.projects.set(projects);
                }
            }

            // Load investments
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "investments", "get_all_investments", &()
            ).await {
                let investments: Vec<InvestmentView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !investments.is_empty() {
                    web_sys::console::log_1(&format!("[Energy] Loaded {} investments", investments.len()).into());
                    state.investments.set(investments);
                }
            }

            // Load trade offers
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "grid", "get_open_offers", &()
            ).await {
                let offers: Vec<TradeOfferView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !offers.is_empty() {
                    web_sys::console::log_1(&format!("[Energy] Loaded {} trade offers", offers.len()).into());
                    state.offers.set(offers);
                }
            }

            // Load regenerative contracts
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "regenerative", "get_all_contracts", &()
            ).await {
                let contracts: Vec<RegenerativeContractView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !contracts.is_empty() {
                    web_sys::console::log_1(&format!("[Energy] Loaded {} contracts", contracts.len()).into());
                    state.contracts.set(contracts);
                }
            }
        }
    });
}

/// Extract the entry from a Holochain Record JSON value.
/// Records have structure: { "entry": { "Present": { ...fields... } } }
fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_energy_context() -> EnergyCtx {
    expect_context::<EnergyCtx>()
}
