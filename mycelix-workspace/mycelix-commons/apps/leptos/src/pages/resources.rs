// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[component]
pub fn ResourcesPage() -> impl IntoView {
    view! {
        <div class="page resources-page">
            <h1>"Resource Mesh"</h1>
            <p class="page-desc">"Community infrastructure — water, food, energy, and shared tools."</p>

            <section class="resource-sections">
                <div class="resource-section">
                    <h2>"Water Infrastructure"</h2>
                    <div class="resource-grid">
                        <div class="resource-card">
                            <h3>"Main Well #1"</h3>
                            <p>"Capacity: 10,000L/day"</p>
                            <p>"Current: 7,200L available"</p>
                            <span class="status-badge good">"Good"</span>
                        </div>
                        <div class="resource-card">
                            <h3>"Rainwater Cistern A"</h3>
                            <p>"Capacity: 5,000L"</p>
                            <p>"Current: 3,100L stored"</p>
                            <span class="status-badge good">"Good"</span>
                        </div>
                    </div>
                </div>

                <div class="resource-section">
                    <h2>"Food Network"</h2>
                    <div class="resource-grid">
                        <div class="resource-card">
                            <h3>"Community Garden South"</h3>
                            <p>"34 plots — 28 active growers"</p>
                            <p>"Season: Autumn harvest"</p>
                            <span class="status-badge good">"Producing"</span>
                        </div>
                        <div class="resource-card">
                            <h3>"Food Bank Central"</h3>
                            <p>"Stock: 2,300 items"</p>
                            <p>"Served: 145 families this week"</p>
                            <span class="status-badge warning">"Low Stock"</span>
                        </div>
                    </div>
                </div>

                <div class="resource-section">
                    <h2>"Tools Library"</h2>
                    <div class="resource-grid">
                        <div class="resource-card">
                            <h3>"Power Tools"</h3>
                            <p>"45 items — 12 checked out"</p>
                        </div>
                        <div class="resource-card">
                            <h3>"Garden Equipment"</h3>
                            <p>"78 items — 23 checked out"</p>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    }
}
