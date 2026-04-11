// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Floating dev panel for testing consciousness tiers and thermodynamic state.
//! Only visible when the conductor is in mock mode.

use leptos::prelude::*;
use mycelix_leptos_core::use_consciousness;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::use_thermodynamic;
use mycelix_leptos_core::use_homeostasis;

#[component]
pub fn DevPanel() -> impl IntoView {
    let hc = use_holochain();
    let consciousness = use_consciousness();
    let thermo = use_thermodynamic();
    let homeostasis = use_homeostasis();
    let pending_care = homeostasis.pending_counts.get(0).copied();
    let pending_decisions = homeostasis.pending_counts.get(1).copied();

    let (collapsed, set_collapsed) = signal(true);

    // Only show in mock mode
    let is_mock = move || hc.is_mock();

    view! {
        {move || is_mock().then(|| {
            let _profile = consciousness.profile.get();
            let set_profile = consciousness.set_profile;

            view! {
                <div class=move || format!("dev-panel {}", if collapsed.get() { "dev-collapsed" } else { "" })>
                    <button
                        class="dev-toggle"
                        on:click=move |_| set_collapsed.set(!collapsed.get())
                    >
                        {move || if collapsed.get() { "DEV" } else { "x" }}
                    </button>

                    {move || (!collapsed.get()).then(|| {
                        let p = consciousness.profile.get();
                        view! {
                            <div class="dev-content">
                                <h4>"Consciousness"</h4>

                                <DevSlider
                                    label="Epistemic"
                                    value=p.epistemic_integrity
                                    on_change=move |v| {
                                        let mut p = consciousness.profile.get();
                                        p.epistemic_integrity = v;
                                        set_profile.set(p);
                                    }
                                />
                                <DevSlider
                                    label="Civic"
                                    value=p.civic_participation
                                    on_change=move |v| {
                                        let mut p = consciousness.profile.get();
                                        p.civic_participation = v;
                                        set_profile.set(p);
                                    }
                                />
                                <DevSlider
                                    label="Stewardship"
                                    value=p.stewardship_care
                                    on_change=move |v| {
                                        let mut p = consciousness.profile.get();
                                        p.stewardship_care = v;
                                        set_profile.set(p);
                                    }
                                />
                                <DevSlider
                                    label="Competence"
                                    value=p.domain_competence
                                    on_change=move |v| {
                                        let mut p = consciousness.profile.get();
                                        p.domain_competence = v;
                                        set_profile.set(p);
                                    }
                                />

                                <div class="dev-tier">
                                    "Tier: "
                                    <span class=move || format!("tier-badge {}", consciousness.tier.get().css_class())>
                                        {move || consciousness.tier.get().label()}
                                    </span>
                                    <span class="dev-score">
                                        {move || format!(" ({:.0}%)", consciousness.profile.get().combined_score(&mycelix_leptos_core::consciousness::DimensionWeights::governance()) * 100.0)}
                                    </span>
                                </div>

                                <h4>"Thermodynamic"</h4>
                                <div class="dev-row">
                                    <span>"Battery: "</span>
                                    <span>{move || format!("{:.0}%", thermo.device_energy.get() * 100.0)}</span>
                                    {move || (!thermo.battery_available.get()).then(|| view! {
                                        <span class="dev-warn">" (API blocked)"</span>
                                    })}
                                </div>
                                <div class="dev-row">
                                    <span>"Network: "</span>
                                    <span>{move || format!("{:.0}%", thermo.network_health.get() * 100.0)}</span>
                                </div>
                                <div class="dev-row">
                                    <span>"Torpor: "</span>
                                    <span>{move || format!("{:.0}%", thermo.torpor_level.get() * 100.0)}</span>
                                </div>

                                <h4>"Homeostasis"</h4>
                                <div class="dev-row">
                                    <span>{move || if homeostasis.in_homeostasis.get() { "VOID ACTIVE" } else { "Normal" }}</span>
                                    <span>
                                        {move || format!(" (care:{}, decisions:{})",
                                            pending_care.map(|s| s.get()).unwrap_or(0),
                                            pending_decisions.map(|s| s.get()).unwrap_or(0)
                                        )}
                                    </span>
                                </div>
                            </div>
                        }
                    })}
                </div>
            }
        })}
    }
}

#[component]
fn DevSlider(
    label: &'static str,
    value: f64,
    on_change: impl Fn(f64) + Copy + 'static,
) -> impl IntoView {
    let initial = (value * 100.0) as i32;
    view! {
        <div class="dev-slider-row">
            <label>{label}</label>
            <input
                type="range"
                min="0"
                max="100"
                prop:value=initial
                on:input=move |ev| {
                    use wasm_bindgen::JsCast;
                    if let Some(target) = ev.target() {
                        if let Some(input) = JsCast::dyn_ref::<web_sys::HtmlInputElement>(&target) {
                            let v: f64 = input.value().parse().unwrap_or(50.0) / 100.0;
                            on_change(v);
                        }
                    }
                }
            />
            <span class="dev-slider-val">{format!("{:.0}", value * 100.0)}</span>
        </div>
    }
}
