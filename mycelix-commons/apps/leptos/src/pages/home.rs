// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[component]
pub fn HomePage() -> impl IntoView {
    view! {
        <div class="page home-page">
            <section class="hero">
                <h1 class="hero-title">"The Commons"</h1>
                <p class="hero-subtitle">
                    "Community-owned resources managed on Holochain. "
                    "Property, housing, care, water, food, and transport — coordinated without middlemen."
                </p>
            </section>

            <section class="dashboard-grid">
                <div class="stat-card">
                    <span class="stat-value">"2,847"</span>
                    <span class="stat-label">"Properties Registered"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"156"</span>
                    <span class="stat-label">"Housing Cooperatives"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"1,203"</span>
                    <span class="stat-label">"Care Commitments Active"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"89%"</span>
                    <span class="stat-label">"Resource Mesh Uptime"</span>
                </div>
            </section>

            <section class="resource-overview">
                <h2>"Resource Mesh Status"</h2>
                <div class="mesh-grid">
                    <div class="mesh-card water">
                        <h3>"Water"</h3>
                        <p>"12 community wells monitored"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                    <div class="mesh-card food">
                        <h3>"Food"</h3>
                        <p>"34 community gardens, 8 food banks"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                    <div class="mesh-card transport">
                        <h3>"Transport"</h3>
                        <p>"67 shared vehicles, 12 routes"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                    <div class="mesh-card mutual-aid">
                        <h3>"Mutual Aid"</h3>
                        <p>"423 active commitments"</p>
                        <span class="mesh-status online">"Online"</span>
                    </div>
                </div>
            </section>

            <section class="consciousness-info">
                <h2>"Consciousness-Gated Access"</h2>
                <p>
                    "Resource access is weighted by your participation tier. "
                    "Contribute to the commons to deepen your engagement level."
                </p>
                <div class="tier-cards">
                    <div class="tier observer">"Observer — View resources"</div>
                    <div class="tier participant">"Participant — Request resources"</div>
                    <div class="tier citizen">"Citizen — Manage allocations"</div>
                    <div class="tier steward">"Steward — Govern policies"</div>
                    <div class="tier guardian">"Guardian — Constitutional changes"</div>
                </div>
            </section>
        </div>
    }
}
