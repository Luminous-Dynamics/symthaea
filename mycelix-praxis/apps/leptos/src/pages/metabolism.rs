// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metabolism Page — Holographic X-Ray of the Civilizational Flow.

use leptos::prelude::*;
use crate::components::gl_canvas::GardenCanvas;

#[component]
pub fn MetabolismPage() -> impl IntoView {
    view! {
        <div class="metabolism-page">
            <header class="metabolic-header">
                <h2>"Holographic X-Ray"</h2>
                <p class="subtitle">"Visualizing the flow of atoms, energy, and value."</p>
            </header>

            <div class="xray-container" style="height: 60vh; background: var(--surface-low); border-radius: 12px; position: relative">
                <div style="position: absolute; top: 20px; left: 20px; z-index: 10">
                    <div class="flow-stat">
                        <span class="label">"Energy Flow:"</span>
                        <span class="value" style="color: var(--primary)">"4.2 kW (Incoming)"</span>
                    </div>
                    <div class="flow-stat">
                        <span class="label">"Pathogen Load:"</span>
                        <span class="value" style="color: var(--error)">"Alert: E. coli detected (Florida Lake)"</span>
                    </div>
                    <div class="flow-stat">
                        <span class="label">"Air Quality (PM2.5):"</span>
                        <span class="value" style="color: var(--warning)">"142 \u{03BC}g/m\u{00B3} (Scrubbing Active)"</span>
                    </div>
                </div>
                
                <div class="gl-xray-view" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center">
                    <p style="color: var(--text-tertiary); font-family: monospace">"BIOLOGICAL IMMUNE SYSTEM VIEW [HEATMAP ACTIVE]"</p>
                    <GardenCanvas node_count=0 />
                </div>
            </div>

            <section class="metabolic-control" style="margin-top: 2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem">
                <div class="control-card">
                    <h4>"Bio-Defense Protocol"</h4>
                    <p style="font-size: 0.8rem">"Deploy Mycoremediation Grids into active sewage spills."</p>
                    <button class="btn-sm btn-outline" style="width: 100%; border-color: var(--success); color: var(--success)">"Deploy Spore-Matrix"</button>
                </div>
                <div class="control-card">
                    <h4>"Atmospheric Arbitrage"</h4>
                    <p style="font-size: 0.8rem">"Current Carbon Black yield: 450mg. Process for conductive ink."</p>
                    <button class="btn-sm btn-outline" style="width: 100%">"Refine Particulates"</button>
                </div>
            </section>
        </div>
    }
}
