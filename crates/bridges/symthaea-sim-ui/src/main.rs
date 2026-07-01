// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea Civilization Simulator UI — Leptos WASM app on port 8090.
//!
//! Interactive toggle matrix for the 2^N scientific instrument.
//! Configure parameters, run simulations, visualize results.

use leptos::prelude::*;

mod components;
use components::*;

fn main() {
    leptos::mount::mount_to_body(App);
}

/// Root application component.
#[component]
fn App() -> impl IntoView {
    // Reactive state for all toggles
    let consciousness_gating = RwSignal::new(true);
    let turchin_cycles = RwSignal::new(true);
    let fep_immiseration = RwSignal::new(true);
    let sacred_stillness = RwSignal::new(true);
    let disasters_enabled = RwSignal::new(true);
    let faction_enabled = RwSignal::new(true);
    let education_enabled = RwSignal::new(true);
    let migration_enabled = RwSignal::new(true);
    let stoichiometry = RwSignal::new(false);
    let epistemic_decay = RwSignal::new(false);

    // Disaster category sliders (0-200%)
    let solar_mult = RwSignal::new(100.0_f64);
    let impact_mult = RwSignal::new(100.0);
    let eclss_mult = RwSignal::new(100.0);
    let psych_mult = RwSignal::new(100.0);

    // Simulation parameters
    let duration_years = RwSignal::new(300_u32);
    let seed = RwSignal::new(42_u64);

    // Computed: active toggle count
    let toggle_count = Memo::new(move |_| {
        let mut count = 0;
        if consciousness_gating.get() {
            count += 1;
        }
        if turchin_cycles.get() {
            count += 1;
        }
        if fep_immiseration.get() {
            count += 1;
        }
        if sacred_stillness.get() {
            count += 1;
        }
        if disasters_enabled.get() {
            count += 1;
        }
        if faction_enabled.get() {
            count += 1;
        }
        if education_enabled.get() {
            count += 1;
        }
        if migration_enabled.get() {
            count += 1;
        }
        if stoichiometry.get() {
            count += 1;
        }
        if epistemic_decay.get() {
            count += 1;
        }
        count
    });

    view! {
        <div class="app">
            <header>
                <h1>"Symthaea Civilization Simulator"</h1>
                <div class="preset-bar">
                    <button class="preset-btn"
                        on:click=move |_| {
                            consciousness_gating.set(true);
                            turchin_cycles.set(true);
                            fep_immiseration.set(true);
                            sacred_stillness.set(true);
                            disasters_enabled.set(true);
                            faction_enabled.set(true);
                        }
                    >"Default"</button>
                    <button class="preset-btn"
                        on:click=move |_| {
                            consciousness_gating.set(false);
                            fep_immiseration.set(false);
                            sacred_stillness.set(false);
                        }
                    >"Legacy Mode"</button>
                    <button class="preset-btn"
                        on:click=move |_| {
                            disasters_enabled.set(false);
                            turchin_cycles.set(false);
                            faction_enabled.set(false);
                        }
                    >"Utopia Mode"</button>
                </div>
            </header>

            <div class="grid">
                // Toggle Panel
                <div class="panel">
                    <h2>"Governance Toggles (" {move || toggle_count.get()} " active)"</h2>
                    <Toggle label="Consciousness Gating" checked=consciousness_gating />
                    <Toggle label="Turchin Secular Cycles" checked=turchin_cycles />
                    <Toggle label="FEP Immiseration" checked=fep_immiseration />
                    <Toggle label="Sacred Stillness" checked=sacred_stillness />
                    <Toggle label="Disasters" checked=disasters_enabled />
                    <Toggle label="Factions" checked=faction_enabled />
                    <Toggle label="Education" checked=education_enabled />
                    <Toggle label="Migration" checked=migration_enabled />
                    <Toggle label="Stoichiometry" checked=stoichiometry />
                    <Toggle label="Epistemic Decay" checked=epistemic_decay />
                </div>

                // Disaster Sliders
                <div class="panel">
                    <h2>"Disaster Severity"</h2>
                    <Slider label="Solar" value=solar_mult />
                    <Slider label="Impact" value=impact_mult />
                    <Slider label="ECLSS" value=eclss_mult />
                    <Slider label="Psychological" value=psych_mult />

                    <h2 style="margin-top: 1rem">"Simulation"</h2>
                    <div class="slider-row">
                        <label>"Duration"</label>
                        <input type="range" min="100" max="1000" step="50"
                            prop:value=move || duration_years.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse() {
                                    duration_years.set(v);
                                }
                            }
                        />
                        <span class="value">{move || format!("{}yr", duration_years.get())}</span>
                    </div>
                    <div class="slider-row">
                        <label>"Seed"</label>
                        <input type="number" style="width: 80px; background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 0.3rem;"
                            prop:value=move || seed.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse() {
                                    seed.set(v);
                                }
                            }
                        />
                    </div>
                </div>

                // Metrics Display
                <div class="panel full-width">
                    <h2>"Live Metrics (configure and run to populate)"</h2>
                    <div class="metrics-grid">
                        <MetricCard label="CVS" value="—" />
                        <MetricCard label="Population" value="—" />
                        <MetricCard label="Phi" value="—" />
                        <MetricCard label="Civil Wars" value="—" />
                        <MetricCard label="Turchin Phase" value="—" />
                        <MetricCard label="P(war)/tick" value="—" />
                    </div>
                </div>
            </div>

            // Run Bar
            <div class="run-bar">
                <button class="run-btn primary">
                    {move || format!("Run {}yr Simulation (seed {})", duration_years.get(), seed.get())}
                </button>
                <button class="run-btn secondary">"Export TOML"</button>
                <button class="run-btn secondary">"Load JSONL Results"</button>
            </div>
        </div>
    }
}
