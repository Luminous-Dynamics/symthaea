// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[component]
pub fn TransportPage() -> impl IntoView {
    view! {
        <div class="page transport-page">
            <h1>"Community Transport"</h1>
            <p class="page-desc">"Shared vehicles and routes — coordinated on the mesh."</p>

            <section class="transport-stats">
                <div class="stat-card">
                    <span class="stat-value">"67"</span>
                    <span class="stat-label">"Shared Vehicles"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"12"</span>
                    <span class="stat-label">"Active Routes"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"89%"</span>
                    <span class="stat-label">"Fleet Available"</span>
                </div>
            </section>

            <section class="vehicle-list">
                <h2>"Available Vehicles"</h2>
                <div class="vehicle-grid">
                    <div class="vehicle-card">
                        <h3>"Electric Van #3"</h3>
                        <p>"Range: 280km — Battery: 84%"</p>
                        <span class="status-badge good">"Available"</span>
                    </div>
                    <div class="vehicle-card">
                        <h3>"Cargo Bike #7"</h3>
                        <p>"Max load: 150kg"</p>
                        <span class="status-badge good">"Available"</span>
                    </div>
                    <div class="vehicle-card">
                        <h3>"Shuttle Bus"</h3>
                        <p>"Route: Central — 12 seats"</p>
                        <span class="status-badge in-use">"In Use"</span>
                    </div>
                </div>
            </section>
        </div>
    }
}
