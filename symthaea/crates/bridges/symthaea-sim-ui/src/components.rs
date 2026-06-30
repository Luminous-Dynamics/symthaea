// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reusable UI components for the Civilization Simulator.

use leptos::prelude::*;

/// A toggle switch with label.
#[component]
pub fn Toggle(label: &'static str, checked: RwSignal<bool>) -> impl IntoView {
    view! {
        <div class="toggle-row">
            <span class="toggle-label">{label}</span>
            <input type="checkbox"
                prop:checked=move || checked.get()
                on:change=move |ev| {
                    checked.set(event_target_checked(&ev));
                }
            />
        </div>
    }
}

/// A slider with label and value display.
#[component]
pub fn Slider(label: &'static str, value: RwSignal<f64>) -> impl IntoView {
    view! {
        <div class="slider-row">
            <label>{label}</label>
            <input type="range" min="0" max="200" step="5"
                prop:value=move || value.get().to_string()
                on:input=move |ev| {
                    if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                        value.set(v);
                    }
                }
            />
            <span class="value">{move || format!("{:.0}%", value.get())}</span>
        </div>
    }
}

/// A metric display card.
#[component]
pub fn MetricCard(label: &'static str, value: &'static str) -> impl IntoView {
    view! {
        <div class="metric-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
    }
}

/// Helper: get checked state from checkbox event.
pub fn event_target_checked(ev: &leptos::ev::Event) -> bool {
    use leptos::wasm_bindgen::JsCast;
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.checked())
        .unwrap_or(false)
}
