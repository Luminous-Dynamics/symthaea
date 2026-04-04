// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::state::globe_state::GlobeState;
use crate::state::data_state::DataState;

/// Planetary Homeostasis HUD — shows live consciousness and system metrics.
#[component]
pub fn Hud() -> impl IntoView {
    let globe_state = expect_context::<GlobeState>();
    let data_state = expect_context::<DataState>();

    let active_count = move || globe_state.active_layers.read().len();

    let ds1 = data_state.clone();
    let ds2 = data_state.clone();
    let ds3 = data_state.clone();

    let total_markers = move || {
        ds1.geothermal_nodes.read().len()
            + ds1.resontia_vaults.read().len()
            + ds1.terra_lumina_sites.read().len()
            + ds1.sites.read().len()
            + ds1.earth_regions.read().len()
            + ds1.climate_projects.read().len()
            + ds1.emergency_shelters.read().len()
            + ds1.health_facilities.read().len()
            + ds1.robotics_dispatch.read().len()
    };

    let total_corridors = move || {
        ds2.maglev_corridors.read().len()
            + ds2.supply_routes.read().len()
    };

    // Aggregate Phi from earth regions
    let global_phi = move || {
        let regions = ds3.earth_regions.read();
        if regions.is_empty() { return 0.0; }
        let total_pop: f64 = regions.iter().map(|r| r.population_m).sum();
        let weighted: f64 = regions.iter().map(|r| r.phi_mean * r.population_m).sum();
        weighted / total_pop
    };

    view! {
        <div class="hud">
            <div class="hud-metric">
                <span class="hud-value psi-value">{move || format!("{:.0}%", global_phi() * 100.0)}</span>
                <span class="hud-label">"Coherence"</span>
            </div>
            <div class="hud-metric">
                <span class="hud-value">{move || total_markers().to_string()}</span>
                <span class="hud-label">"Nodes"</span>
            </div>
            <div class="hud-metric">
                <span class="hud-value">{move || total_corridors().to_string()}</span>
                <span class="hud-label">"Corridors"</span>
            </div>
            <div class="hud-metric">
                <span class="hud-value">{move || active_count().to_string()}</span>
                <span class="hud-label">"Layers"</span>
            </div>
        </div>
    }
}
