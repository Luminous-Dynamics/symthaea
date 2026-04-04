// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::data::types::EnergyType;

#[component]
pub fn Legend() -> impl IntoView {
    let types = [
        EnergyType::Solar,
        EnergyType::Wind,
        EnergyType::Hydro,
        EnergyType::Nuclear,
        EnergyType::Geothermal,
        EnergyType::Battery,
    ];

    view! {
        <div class="legend">
            <h4>"Energy Types"</h4>
            {types.into_iter().map(|t| {
                view! {
                    <div class="legend-item">
                        <span class="legend-swatch" style=format!("background: {}", t.hex_color()) />
                        <span>{t.label()}</span>
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}
