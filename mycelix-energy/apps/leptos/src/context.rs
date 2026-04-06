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

    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            let _ = &state;
        }
    });
}

pub fn use_energy_context() -> EnergyCtx {
    expect_context::<EnergyCtx>()
}
