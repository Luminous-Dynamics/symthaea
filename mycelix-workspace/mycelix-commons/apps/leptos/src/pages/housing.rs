// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[component]
pub fn HousingPage() -> impl IntoView {
    view! {
        <div class="page housing-page">
            <h1>"Housing Cooperatives"</h1>
            <p class="page-desc">"Collectively managed housing — from waitlists to maintenance."</p>

            <div class="coop-grid">
                <div class="coop-card">
                    <h3>"Hillside Co-op"</h3>
                    <div class="coop-stats">
                        <span>"24 units"</span>
                        <span>"2 available"</span>
                        <span>"Waitlist: 5"</span>
                    </div>
                    <div class="maintenance-bar">
                        <span>"Maintenance Fund"</span>
                        <div class="bar"><div class="bar-fill" style="width: 78%"></div></div>
                        <span>"78%"</span>
                    </div>
                    <span class="tier-badge">"Requires: Participant"</span>
                </div>
                <div class="coop-card">
                    <h3>"Valley Gardens"</h3>
                    <div class="coop-stats">
                        <span>"16 units"</span>
                        <span>"0 available"</span>
                        <span>"Waitlist: 12"</span>
                    </div>
                    <div class="maintenance-bar">
                        <span>"Maintenance Fund"</span>
                        <div class="bar"><div class="bar-fill" style="width: 92%"></div></div>
                        <span>"92%"</span>
                    </div>
                    <span class="tier-badge">"Requires: Participant"</span>
                </div>
            </div>

            <section class="maintenance-requests">
                <h2>"Open Maintenance Requests"</h2>
                <div class="request-list">
                    <div class="request-item">
                        <span class="priority high">"High"</span>
                        <span>"Hillside B3 — Roof leak"</span>
                        <span class="date">"2026-03-28"</span>
                    </div>
                    <div class="request-item">
                        <span class="priority medium">"Medium"</span>
                        <span>"Valley G7 — Window seal"</span>
                        <span class="date">"2026-03-25"</span>
                    </div>
                </div>
            </section>
        </div>
    }
}
