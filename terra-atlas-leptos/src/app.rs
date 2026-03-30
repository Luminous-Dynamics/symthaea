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

    // Load static data on mount
    let data_for_effect = data_state.clone();
    Effect::new(move |_| {
        let loaded = static_data::load_all();
        data_for_effect.set_all(loaded);
        log::info!(
            "Loaded {} geothermal nodes, {} corridors, {} vaults, {} Terra Lumina sites",
            data_for_effect.geothermal_nodes.read().len(),
            data_for_effect.maglev_corridors.read().len(),
            data_for_effect.resontia_vaults.read().len(),
            data_for_effect.terra_lumina_sites.read().len(),
        );
    });

    // Provide state via context
    provide_context(globe_state.clone());
    provide_context(data_state.clone());

    view! {
        <GlobeCanvas />
        <div class="globe-title">
            <h1>"Terra Atlas"</h1>
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
