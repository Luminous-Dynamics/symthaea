// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Water Stewardship Sandbox — Interactive Greywater Loop Simulation.
//! Teaches students to balance filtration, storage, and agricultural output.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct WaterSystemState {
    pub raw_greywater_liters: f32,
    pub filtered_reservoir_liters: f32,
    pub filter_clog_permille: u16,
    pub soil_moisture_pct: u8,
}

#[component]
pub fn WaterStewardshipGame() -> impl IntoView {
    let (state, set_state) = signal(WaterSystemState {
        raw_greywater_liters: 15.0,
        filtered_reservoir_liters: 5.0,
        filter_clog_permille: 100, // 10% clogged
        soil_moisture_pct: 45,
    });

    // Tick-based simulation
    let _ = use_interval(1000, move || {
        set_state.update(|s| {
            // Natural evaporation and plant uptake
            s.soil_moisture_pct = s.soil_moisture_pct.saturating_sub(1);
            
            // Automatic filtration flow
            if s.raw_greywater_liters > 0.1 && s.filter_clog_permille < 950 {
                let flow = 0.5 * (1.0 - (s.filter_clog_permille as f32 / 1000.0));
                s.raw_greywater_liters -= flow;
                s.filtered_reservoir_liters += flow;
                // Debris accumulation in filter
                s.filter_clog_permille = (s.filter_clog_permille + 5).min(1000);
            }
        });
    });

    view! {
        <div class="water-sandbox">
            <header class="sandbox-header">
                <h3>"VOC-H2O-101: The Greywater Loop"</h3>
                <div class="system-status">
                    <span class=move || if state.get().soil_moisture_pct > 30 { "status-safe" } else { "status-danger" }>
                        "Soil Health: " {move || state.get().soil_moisture_pct}"%"
                    </span>
                </div>
            </header>

            <div class="water-layout">
                // 1. Storage Visualization
                <div class="storage-columns">
                    <div class="tank">
                        <label>"Raw Greywater"</label>
                        <div class="tank-visual greywater">
                            <div class="water-level" style=move || format!("height: {}%", (state.get().raw_greywater_liters * 2.0).min(100.0))></div>
                        </div>
                        <span class="tank-label">{move || format!("{:.1}L", state.get().raw_greywater_liters)}</span>
                        <button class="btn-sm" on:click=move |_| set_state.update(|s| s.raw_greywater_liters += 10.0)>"Add Waste"</button>
                    </div>

                    <div class="filter-pipe">
                        <div class="clog-indicator" style=move || format!("opacity: {}", state.get().filter_clog_permille as f32 / 1000.0)>
                            "\u{26A0} Clogged"
                        </div>
                        <div class="pipe-arrow">"\u{2192}"</div>
                        <button class="btn-sm btn-outline" on:click=move |_| set_state.update(|s| s.filter_clog_permille = 0)>"Backwash Filter"</button>
                    </div>

                    <div class="tank">
                        <label>"Filtered Reservoir"</label>
                        <div class="tank-visual clean">
                            <div class="water-level" style=move || format!("height: {}%", (state.get().filtered_reservoir_liters * 2.0).min(100.0))></div>
                        </div>
                        <span class="tank-label">{move || format!("{:.1}L", state.get().filtered_reservoir_liters)}</span>
                        <button 
                            class="btn-sm btn-primary"
                            disabled=move || state.get().filtered_reservoir_liters < 5.0
                            on:click=move |_| {
                                set_state.update(|s| {
                                    s.filtered_reservoir_liters -= 5.0;
                                    s.soil_moisture_pct = (s.soil_moisture_pct + 25).min(100);
                                });
                            }
                        >"Irrigate Crops"</button>
                    </div>
                </div>

                // 2. The Logic Panel
                <div class="water-logic">
                    <h4>"Stewardship Challenge"</h4>
                    <p>"Maintain soil moisture above 40% without letting the raw greywater tank overflow or the filter clog completely."</p>
                    
                    <div class="proof-of-dirt-box">
                        <h5>"Proof of Dirt (IoT Forecast):"</h5>
                        {move || {
                            let s = state.get();
                            if s.soil_moisture_pct > 60 && s.filter_clog_permille < 300 {
                                view! { <div class="status-pass">"\u{2705} High Stewardship Efficiency"</div> }.into_any()
                            } else if s.soil_moisture_pct < 40 {
                                view! { <div class="status-fail">"\u{274C} Crops are wilting"</div> }.into_any()
                            } else {
                                view! { <div class="status-warn">"\u{26A0} Maintain Balance"</div> }.into_any()
                            }
                        }}
                    </div>
                </div>
            </div>

            <footer class="sandbox-footer">
                <p style="font-size: 0.7rem; color: var(--text-tertiary)">
                    "Simulation grounded in standard IBC-tote sand/charcoal filtration flow rates."
                </p>
            </footer>
        </div>
    }
}
