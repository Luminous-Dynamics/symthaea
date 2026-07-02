// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::contexts::commons_context::use_commons;

#[component]
pub fn WaterPage() -> impl IntoView {
    let commons = use_commons();
    view! {
        <div class="water-page" data-page="water" role="main">
            <h1 class="page-title">"Water"</h1>
            <p class="page-subtitle">"capture, store, steward — every drop is sacred"</p>
            <div class="water-systems" data-section="water-systems" role="list">
                {move || commons.water_systems.get().into_iter().map(|w| {
                    let type_label = w.system_type.label().to_string();
                    let fill = w.fill_pct();
                    let owner = w.owner_did.split(':').last().unwrap_or(&w.owner_did).to_string();
                    view! {
                        <div class="water-card" data-system-hash=w.hash.clone() data-fill-pct=format!("{fill:.0}") data-capacity=w.capacity_liters.to_string() role="listitem">
                            <div class="water-header">
                                <h3 class="water-name">{w.name}</h3>
                                <span class="water-type">{type_label}</span>
                            </div>
                            <div class="water-level">
                                <div class="level-bar">
                                    <div class="level-fill" style=format!("width: {fill:.0}%")></div>
                                </div>
                                <span class="level-text" data-metric="level">
                                    {format!("{}L / {}L ({fill:.0}%)", w.current_level_liters, w.capacity_liters)}
                                </span>
                            </div>
                            <span class="water-steward">{format!("steward: {owner}")}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
