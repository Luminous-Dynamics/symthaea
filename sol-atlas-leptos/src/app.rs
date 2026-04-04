// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::components::globe_canvas::GlobeCanvas;
use crate::components::hud::Hud;
use crate::components::planet_nav::PlanetNav;
use crate::components::info_panel::InfoPanel;
use crate::components::layer_panel::LayerPanel;
use crate::components::legend::Legend;
use crate::components::timeline::Timeline;
use crate::components::tooltip::Tooltip;
use crate::data::static_data;
use crate::state::data_state::DataState;
use crate::state::globe_state::GlobeState;

#[component]
pub fn App() -> impl IntoView {
    // Initialize reactive state
    let globe_state = GlobeState::new();
    let data_state = DataState::new();

    // Load static data immediately (always available as fallback)
    let data_for_effect = data_state.clone();
    Effect::new(move |_| {
        let loaded = static_data::load_all();
        log::info!(
            "Static data: {} sites, {} geothermal, {} corridors, {} vaults, {} fossil deposits, {} nuclear",
            loaded.sites.len(),
            loaded.geothermal_nodes.len(),
            loaded.maglev_corridors.len(),
            loaded.resontia_vaults.len(),
            loaded.fossil_deposits.len(),
            loaded.nuclear_sites.len(),
        );
        data_for_effect.set_all(loaded);
    });

    // When holochain feature is enabled, try to fetch live data and merge
    #[cfg(feature = "holochain")]
    {
        let data_for_hc = data_state.clone();
        Effect::new(move |_| {
            let ds = data_for_hc.clone();
            wasm_bindgen_futures::spawn_local(async move {
                use crate::data::holochain;
                // Try each data source — on success, replace static data
                let sites = holochain::fetch_all_sites().await;
                if !sites.is_empty() { ds.sites.set(sites); }
                let nodes = holochain::fetch_geothermal_nodes().await;
                if !nodes.is_empty() { ds.geothermal_nodes.set(nodes); }
                let corridors = holochain::fetch_maglev_corridors().await;
                if !corridors.is_empty() { ds.maglev_corridors.set(corridors); }
                let vaults = holochain::fetch_vaults().await;
                if !vaults.is_empty() { ds.resontia_vaults.set(vaults); }
                let tl = holochain::fetch_terra_lumina_sites().await;
                if !tl.is_empty() { ds.terra_lumina_sites.set(tl); }
                let deposits = holochain::fetch_fossil_deposits().await;
                if !deposits.is_empty() { ds.fossil_deposits.set(deposits); }
            });
        });
    }

    // Provide state via context
    provide_context(globe_state.clone());
    provide_context(data_state.clone());

    view! {
        <GlobeCanvas />
        <div class="globe-title">
            <h1>"Sol Atlas"</h1>
            <p class="subtitle">"Symthaea Planetary Coordination Layer"</p>
        </div>
        <Hud />
        <LayerPanel />
        <Timeline />
        <Legend />
        <Tooltip />
        <InfoPanel />
        <PlanetNav />
    }
}
