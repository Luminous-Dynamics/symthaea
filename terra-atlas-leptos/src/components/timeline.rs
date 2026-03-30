// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::state::globe_state::GlobeState;

/// 777-year Resontia simulation timeline slider.
/// Shows infrastructure construction epochs:
///   Epoch 1 (0-25):   Foundation — first geothermal taps
///   Epoch 2 (25-50):  Growth — maglev corridors begin
///   Epoch 3 (50-150): Maturation — vault construction begins
///   Epoch 4 (150-300): Hardening — full network operational
///   Epoch 5 (300-777): Deep time — multi-century resilience
#[component]
pub fn Timeline() -> impl IntoView {
    let globe_state = expect_context::<GlobeState>();
    let year = globe_state.timeline_year;

    let epoch_name = move || {
        let y = year.get();
        match y {
            0..=25 => "Foundation",
            26..=50 => "Growth",
            51..=150 => "Maturation",
            151..=300 => "Hardening",
            _ => "Deep Time",
        }
    };

    let epoch_detail = move || {
        let y = year.get();
        match y {
            0..=25 => "First geothermal taps online. Survey teams mapping vault sites.",
            26..=50 => "Maglev corridors under construction. Initial vault excavation.",
            51..=150 => "Vault network expanding. Spaceport construction after 4 vaults.",
            151..=300 => "Full maglev grid operational. All 12 vaults active.",
            _ => "Multi-century resilience. Self-sustaining civilization infrastructure.",
        }
    };

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.unchecked_into();
        let val: u32 = input.value().parse().unwrap_or(0);
        year.set(val);
    };

    view! {
        <div class="timeline">
            <div class="timeline-header">
                <span class="timeline-label">"YEAR "</span>
                <span class="timeline-year">{move || year.get().to_string()}</span>
                <span class="timeline-epoch">{epoch_name}</span>
            </div>
            <input
                type="range"
                min="0"
                max="777"
                prop:value=move || year.get().to_string()
                on:input=on_input
                class="timeline-slider"
            />
            <div class="timeline-detail">{epoch_detail}</div>
        </div>
    }
}
